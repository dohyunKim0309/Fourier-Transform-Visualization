"""
ftvis.core — Layer 1: Pure mathematical primitives.

이 레이어는 numpy 외 의존성이 없고, 어떤 렌더러에 대해서도 무지하다.
모든 함수는 결정론적이며, 동일 입력에 동일 출력을 보장한다.

자세한 시그니처 명세는 동봉된 ``core.pyi`` 참조.

Conventions
-----------
- 시간 t: 단위 second.
- 주파수: angular frequency ω (rad/s)로 통일. Hz는 사용자가 직접 ω = 2π·f로 변환.
- 복소수 표현은 numpy의 complex128.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

TimeArray = NDArray[np.float64]
RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
RealSignalFn = Callable[[TimeArray], NDArray]
ComplexSignalFn = Callable[[TimeArray], NDArray]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_TWO_PI = 2.0 * math.pi


def _ensure_time_array(t: NDArray) -> TimeArray:
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError(f"time array must be 1-D, got shape {t_arr.shape}")
    return t_arr


def _broadcast_real(values: Any, shape: tuple) -> RealArray:
    """fn(t) 결과가 스칼라일 수 있으므로 broadcast로 t shape에 맞춰주고 float64로 캐스팅."""
    arr = np.broadcast_to(np.asarray(values), shape)
    return np.asarray(arr, dtype=np.float64)


def _broadcast_complex(values: Any, shape: tuple) -> ComplexArray:
    arr = np.broadcast_to(np.asarray(values), shape)
    return np.asarray(arr, dtype=np.complex128)


def _is_complex_scalar(z: Any) -> bool:
    """파이썬 complex 또는 numpy complex 스칼라이고, 허수부가 0이 아니면 True."""
    if isinstance(z, complex):
        return z.imag != 0
    if isinstance(z, np.complexfloating):
        return complex(z).imag != 0
    return False


def _format_scalar(z: complex) -> str:
    if z.imag == 0:
        return f"{z.real:g}"
    if z.real == 0:
        return f"{z.imag:g}j"
    return f"({z.real:g}{z.imag:+g}j)"


# ─────────────────────────────────────────────────────────────────────────────
# Signal — 실수 시간 함수 (기본 클래스)
# ─────────────────────────────────────────────────────────────────────────────

class Signal:
    """실수 시간 신호. 시그니처는 core.pyi 참조."""

    __slots__ = ("_fn", "name")

    def __init__(self, fn: RealSignalFn, name: str = "<lambda>") -> None:
        if not callable(fn):
            raise TypeError(f"Signal expects a callable, got {type(fn).__name__}")
        self._fn: RealSignalFn = fn
        self.name: str = name

    @classmethod
    def from_lambda(cls, fn: RealSignalFn, *, name: str = "<lambda>") -> "Signal":
        return cls(fn, name=name)

    def sample(self, t: TimeArray) -> RealArray:
        t_arr = _ensure_time_array(t)
        return _broadcast_real(self._fn(t_arr), t_arr.shape)

    # ── 합성 연산 ─────────────────────────────────────────────────────────
    @staticmethod
    def _coerce(other: object) -> "Signal":
        """피연산자를 적절한 Signal/ComplexSignal로 강제 변환."""
        if isinstance(other, Signal):  # ComplexSignal/WoundSignal 포함
            return other
        if isinstance(other, (int, float, np.integer, np.floating)):
            value = float(other)
            return Signal(
                lambda t, _v=value: np.full_like(t, _v, dtype=np.float64),
                name=_format_scalar(complex(value)),
            )
        if _is_complex_scalar(other):
            value_c = complex(other)
            return ComplexSignal(
                lambda t, _v=value_c: np.full_like(t, _v, dtype=np.complex128),
                name=_format_scalar(value_c),
            )
        if isinstance(other, complex):
            # 허수부 0인 파이썬 complex (위 _is_complex_scalar에서 걸리지 않은)도 실수로.
            value = float(other.real)
            return Signal(
                lambda t, _v=value: np.full_like(t, _v, dtype=np.float64),
                name=_format_scalar(complex(value)),
            )
        raise TypeError(f"cannot combine Signal with {type(other).__name__}")

    @staticmethod
    def _is_complex_signal(s: "Signal") -> bool:
        return isinstance(s, ComplexSignal)

    def _binop(self, other: object, op: str) -> "Signal":
        rhs = Signal._coerce(other)
        lhs_complex = Signal._is_complex_signal(self)
        rhs_complex = Signal._is_complex_signal(rhs)
        result_complex = lhs_complex or rhs_complex
        lhs_fn, rhs_fn = self._fn, rhs._fn

        if op == "+":
            symbol = " + "
            combined = lambda t, _l=lhs_fn, _r=rhs_fn: np.asarray(_l(t)) + np.asarray(_r(t))
        elif op == "-":
            symbol = " - "
            combined = lambda t, _l=lhs_fn, _r=rhs_fn: np.asarray(_l(t)) - np.asarray(_r(t))
        elif op == "*":
            symbol = " · "
            combined = lambda t, _l=lhs_fn, _r=rhs_fn: np.asarray(_l(t)) * np.asarray(_r(t))
        else:
            raise ValueError(f"unknown op {op!r}")

        new_name = f"({self.name}{symbol}{rhs.name})"
        if result_complex:
            return ComplexSignal(combined, name=new_name)
        return Signal(combined, name=new_name)

    def __add__(self, other: object) -> "Signal":
        return self._binop(other, "+")

    def __radd__(self, other: object) -> "Signal":
        return Signal._coerce(other)._binop(self, "+")

    def __sub__(self, other: object) -> "Signal":
        return self._binop(other, "-")

    def __rsub__(self, other: object) -> "Signal":
        return Signal._coerce(other)._binop(self, "-")

    def __mul__(self, other: object) -> "Signal":
        return self._binop(other, "*")

    def __rmul__(self, other: object) -> "Signal":
        return Signal._coerce(other)._binop(self, "*")

    def __neg__(self) -> "Signal":
        fn = self._fn
        if isinstance(self, ComplexSignal):
            return ComplexSignal(lambda t: -np.asarray(fn(t)), name=f"-{self.name}")
        return Signal(lambda t: -np.asarray(fn(t)), name=f"-{self.name}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


# ─────────────────────────────────────────────────────────────────────────────
# ComplexSignal — 복소 시간 함수
# ─────────────────────────────────────────────────────────────────────────────

class ComplexSignal(Signal):
    """복소 시간 신호. Signal의 자식. 시그니처는 core.pyi 참조."""

    __slots__ = ()

    @classmethod
    def from_lambda(  # type: ignore[override]
        cls, fn: ComplexSignalFn, *, name: str = "<lambda>"
    ) -> "ComplexSignal":
        return cls(fn, name=name)

    def sample(self, t: TimeArray) -> ComplexArray:  # type: ignore[override]
        t_arr = _ensure_time_array(t)
        return _broadcast_complex(self._fn(t_arr), t_arr.shape)


# ─────────────────────────────────────────────────────────────────────────────
# WoundSignal — x(t)·e^(-jωt), ComplexSignal의 자식
# ─────────────────────────────────────────────────────────────────────────────

class WoundSignal(ComplexSignal):
    """winding 결과. ComplexSignal의 자식 + 사전 분석 데이터. 시그니처는 core.pyi 참조."""

    __slots__ = (
        "base",
        "omega",
        "t",
        "signal_values",
        "wound_values",
        "running_integral",
    )

    def __init__(
        self,
        *,
        base: Signal,
        omega: float,
        t: TimeArray,
        signal_values: ComplexArray,
        wound_values: ComplexArray,
        running_integral: ComplexArray,
    ) -> None:
        # 평가 함수: x(t)·e^(-jωt)를 닫힌 형태로 계산.
        omega_f = float(omega)
        base_fn = base._fn

        def fn(tau: TimeArray, _w=omega_f, _b=base_fn) -> ComplexArray:
            x = np.asarray(_b(tau))
            return np.asarray(x, dtype=np.complex128) * np.exp(-1j * _w * tau)

        # ComplexSignal 부모의 __init__는 Signal.__init__이고 fn/name만 받는다.
        super().__init__(fn, name=f"{base.name}·exp(-j{omega_f:g}t)")

        # __slots__ 필드 채우기 (object.__setattr__로 ComplexSignal/Signal __slots__ 충돌 회피)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "omega", omega_f)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "signal_values", signal_values)
        object.__setattr__(self, "wound_values", wound_values)
        object.__setattr__(self, "running_integral", running_integral)

    @property
    def final_integral(self) -> complex:
        return complex(self.running_integral[-1])

    def __repr__(self) -> str:
        return (
            f"WoundSignal(base={self.base.name!r}, omega={self.omega:g}, "
            f"N={len(self.t)}, final={self.final_integral:.4g})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# signals — 표준 신호 팩토리
# ─────────────────────────────────────────────────────────────────────────────

class signals:
    """표준 신호 팩토리들의 namespace. 시그니처는 core.pyi 참조."""

    @staticmethod
    def cosine(omega: float, amp: float = 1.0, phase: float = 0.0) -> Signal:
        omega_f = float(omega)

        def fn(t: TimeArray) -> NDArray:
            return amp * np.cos(omega_f * t + phase)

        return Signal(fn, name=f"{amp:g}·cos({omega_f:g}t{_phase_label(phase)})")

    @staticmethod
    def sine(omega: float, amp: float = 1.0, phase: float = 0.0) -> Signal:
        omega_f = float(omega)

        def fn(t: TimeArray) -> NDArray:
            return amp * np.sin(omega_f * t + phase)

        return Signal(fn, name=f"{amp:g}·sin({omega_f:g}t{_phase_label(phase)})")

    @staticmethod
    def complex_exp(omega: float, amp: complex = 1.0 + 0j) -> ComplexSignal:
        omega_f = float(omega)
        amp_c = complex(amp)

        def fn(t: TimeArray) -> NDArray:
            return amp_c * np.exp(1j * omega_f * t)

        return ComplexSignal(fn, name=f"{_format_scalar(amp_c)}·exp(j·{omega_f:g}t)")

    @staticmethod
    def gaussian(sigma: float = 1.0, center: float = 0.0, amp: float = 1.0) -> Signal:
        if sigma <= 0:
            raise ValueError(f"gaussian sigma must be > 0, got {sigma}")
        two_sigma_sq = 2.0 * sigma * sigma

        def fn(t: TimeArray) -> NDArray:
            return amp * np.exp(-((t - center) ** 2) / two_sigma_sq)

        return Signal(fn, name=f"{amp:g}·gauss(σ={sigma:g}, c={center:g})")

    @staticmethod
    def rect(width: float = 1.0, center: float = 0.0, amp: float = 1.0) -> Signal:
        if width <= 0:
            raise ValueError(f"rect width must be > 0, got {width}")
        half = width / 2.0

        def fn(t: TimeArray) -> NDArray:
            mask = (t >= center - half) & (t <= center + half)
            out = np.zeros_like(t, dtype=np.float64)
            out[mask] = amp
            return out

        return Signal(fn, name=f"{amp:g}·rect(w={width:g}, c={center:g})")

    @staticmethod
    def step(t0: float = 0.0, amp: float = 1.0) -> Signal:
        def fn(t: TimeArray) -> NDArray:
            out = np.zeros_like(t, dtype=np.float64)
            out[t >= t0] = amp
            return out

        return Signal(fn, name=f"{amp:g}·u(t-{t0:g})")

    @staticmethod
    def decaying_exp(rate: float = 1.0, amp: float = 1.0, t0: float = 0.0) -> Signal:
        if rate <= 0:
            raise ValueError(f"decaying_exp rate must be > 0, got {rate}")

        def fn(t: TimeArray) -> NDArray:
            out = np.zeros_like(t, dtype=np.float64)
            mask = t >= t0
            out[mask] = amp * np.exp(-rate * (t[mask] - t0))
            return out

        return Signal(fn, name=f"{amp:g}·exp(-{rate:g}(t-{t0:g}))·u(t-{t0:g})")

    @staticmethod
    def two_sided_exp(rate: float = 1.0, amp: float = 1.0,
                      center: float = 0.0) -> Signal:
        """A·exp(-rate·|t - center|). 짝함수, ∫|x|dt = 2A/rate < ∞.

        FT: X(jω) = 2·A·rate / (rate² + ω²) — 순수 실수, 짝함수성 직접 확인 가능.
        """
        if rate <= 0:
            raise ValueError(f"two_sided_exp rate must be > 0, got {rate}")

        def fn(t: TimeArray) -> NDArray:
            return amp * np.exp(-rate * np.abs(t - center))

        return Signal(fn, name=f"{amp:g}·exp(-{rate:g}·|t-{center:g}|)")


def _phase_label(phase: float) -> str:
    if phase == 0:
        return ""
    return f" + {phase:g}"


# ─────────────────────────────────────────────────────────────────────────────
# FourierAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class FourierAnalyzer:
    """winding 계산의 진입점. 시그니처는 core.pyi 참조."""

    def __init__(
        self,
        signal: Signal,
        *,
        t_min: float = 0.0,
        t_max: float = 4.0,
        n_samples: int = 500,
    ) -> None:
        if not isinstance(signal, Signal):
            raise TypeError(f"FourierAnalyzer expects a Signal, got {type(signal).__name__}")
        if t_max <= t_min:
            raise ValueError(f"t_max ({t_max}) must be > t_min ({t_min})")
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")

        self.signal: Signal = signal
        self.t_min: float = float(t_min)
        self.t_max: float = float(t_max)
        self.n_samples: int = int(n_samples)

        self._t_cache: Optional[TimeArray] = None
        self._signal_cache: Optional[ComplexArray] = None

    @property
    def t(self) -> TimeArray:
        if self._t_cache is None:
            self._t_cache = np.linspace(
                self.t_min, self.t_max, self.n_samples, dtype=np.float64
            )
        return self._t_cache

    def _signal_samples(self) -> ComplexArray:
        """원본 신호의 샘플을 complex128로 캐싱."""
        if self._signal_cache is None:
            x = self.signal.sample(self.t)
            self._signal_cache = np.asarray(x, dtype=np.complex128)
        return self._signal_cache

    # ── 핵심: winding ─────────────────────────────────────────────────────
    def wound_at_omega(self, omega: float) -> WoundSignal:
        t = self.t
        x = self._signal_samples()
        carrier = np.exp(-1j * float(omega) * t)
        wound_values = x * carrier
        running = _running_integral(t, wound_values)
        return WoundSignal(
            base=self.signal,
            omega=float(omega),
            t=t,
            signal_values=x,
            wound_values=wound_values,
            running_integral=running,
        )

    # ── Mode B: spectrum & inverse accumulation ───────────────────────────
    def spectrum(
        self,
        omega_min: float,
        omega_max: float,
        n_omega: int = 200,
    ) -> tuple[NDArray[np.float64], ComplexArray]:
        """[omega_min, omega_max] 범위에서 X(jω)를 n_omega개 점에 대해 계산.

        각 ω에 대해 wound_at_omega(ω).final_integral. 양/음 ω 모두 가능.
        """
        if n_omega < 2:
            raise ValueError(f"n_omega must be >= 2, got {n_omega}")
        if omega_max <= omega_min:
            raise ValueError(
                f"omega_max ({omega_max}) must be > omega_min ({omega_min})"
            )
        omegas = np.linspace(omega_min, omega_max, int(n_omega), dtype=np.float64)
        # 효율 향상: signal samples를 한 번만 평가하고 reuse.
        t = self.t
        x = self._signal_samples()
        # wound[k, i] = x(t_i) · e^(-jω_k t_i). 메모리 vs 시간 trade-off:
        # n_omega × n_samples 행렬. n_omega=200, n_samples=2000이면 400k complex = 6.4 MB.
        # 충분히 작으므로 한 번에 처리.
        # X[k] = ∫ x(t)·e^(-jω_k t) dt ≈ trapezoidal sum.
        X = np.empty(omegas.size, dtype=np.complex128)
        for k, omega_k in enumerate(omegas):
            wound_k = x * np.exp(-1j * float(omega_k) * t)
            # trapezoidal: (1/2)(wound[0:-1] + wound[1:]) · dt 합.
            dt = np.diff(t)
            X[k] = np.sum(0.5 * (wound_k[:-1] + wound_k[1:]) * dt)
        return omegas, X

    def inverse_accumulation(
        self,
        t_fix: float,
        omega_min: float,
        omega_max: float,
        n_omega: int = 200,
        order: str = "monotonic",
    ) -> tuple[NDArray[np.float64], ComplexArray, ComplexArray]:
        """고정 시점 t_fix에서, ω를 누적하며 역변환 화살표 시퀀스 반환.

        Parameters
        ----------
        t_fix : float
            역변환 결과를 보고 싶은 시점.
        omega_min, omega_max : float
            적분 ω 범위. 양/음 모두 가능.
        n_omega : int
            ω 샘플 수.
        order : 'monotonic' | 'paired_by_abs'
            화살표 누적 순서.
            * 'monotonic': omega_min → omega_max로 단조 증가 순서.
            * 'paired_by_abs': |ω| 작은 것부터 큰 것 순. ω 와 -ω를 짝으로 인접 누적.
              실수 신호의 *허수부 cancel*이 가장 명료하게 보임.

        Returns
        -------
        omegas : NDArray[float64], shape (n_omega,)
            누적 *순서*에 따라 정렬된 ω 배열 (order에 따라 다름).
        arrows : ComplexArray, shape (n_omega,)
            각 ω의 기여 (1/(2π))·X(jω)·e^(jωt_fix)·dω.
        accumulated : ComplexArray, shape (n_omega,)
            머리 잇기 누적합. accumulated[-1] ≈ x(t_fix) (Riemann 근사).

        Notes
        -----
        omega_min/omega_max를 크게 잡고 n_omega를 충분히 크게 하면 accumulated[-1]이
        실제 x(t_fix)에 수렴. 비감쇠 신호는 발산하므로 무한 적분 가능 신호에서만
        의미 있는 결과.
        """
        if order not in ("monotonic", "paired_by_abs"):
            raise ValueError(
                f"order must be 'monotonic' or 'paired_by_abs', got {order!r}"
            )

        # spectrum 계산: omega_min→omega_max 단조 순서로 X(jω) 얻기.
        omegas_mono, X_mono = self.spectrum(omega_min, omega_max, n_omega)
        # ω 간격 (uniform 가정)
        d_omega = float(omegas_mono[1] - omegas_mono[0])

        # 역변환 화살표 (1/(2π)) · X(jω) · e^(jωt_fix) · dω
        arrows_mono = (1.0 / (2.0 * math.pi)) * X_mono * np.exp(
            1j * omegas_mono * float(t_fix)
        ) * d_omega

        if order == "monotonic":
            omegas_out = omegas_mono
            arrows_out = arrows_mono
        else:  # 'paired_by_abs'
            # |ω| 오름차순으로 재정렬. tie-breaker로 ω 자체.
            idx = np.lexsort((omegas_mono, np.abs(omegas_mono)))
            omegas_out = omegas_mono[idx]
            arrows_out = arrows_mono[idx]

        accumulated = np.cumsum(arrows_out)
        return omegas_out, arrows_out, accumulated


# ─────────────────────────────────────────────────────────────────────────────
# 적분 누적합 — trapezoidal cumulative integration
# ─────────────────────────────────────────────────────────────────────────────

def _running_integral(t: TimeArray, y: ComplexArray) -> ComplexArray:
    """∫_{t[0]}^{t[k]} y(τ) dτ를 사다리꼴 규칙으로 누적 계산. 길이 보존, [0]=0."""
    if y.shape != t.shape:
        raise ValueError(f"shape mismatch: t {t.shape} vs y {y.shape}")
    if t.size < 2:
        return np.zeros_like(y)
    dt = np.diff(t)
    segment = 0.5 * (y[:-1] + y[1:]) * dt
    running = np.empty_like(y, dtype=np.complex128)
    running[0] = 0.0
    np.cumsum(segment, out=running[1:])
    return running
