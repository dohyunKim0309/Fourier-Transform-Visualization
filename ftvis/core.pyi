"""
ftvis.core — Layer 1: Pure mathematical primitives.

이 레이어는 numpy 외 의존성이 없고, 어떤 렌더러에 대해서도 무지하다.
모든 함수는 결정론적이며, 동일 입력에 동일 출력을 보장한다.

Conventions
-----------
- 시간 t: 단위 second.
- 주파수: angular frequency ω (rad/s)로 통일. Hz는 사용자가 직접 ω = 2π·f로
  변환해 입력. API에 freq/omega 두 갈래를 두지 않는다 (DESIGN §3.1).
- 복소수 표현은 numpy의 complex128. 실수 결과도 broadcast 시 복소 배열로 승격됨.

Class hierarchy
---------------
::

    Signal             # 실수 시간 함수의 표현 (기본)
     └── ComplexSignal   # 복소 시간 함수
          └── WoundSignal  # x(t)·e^(-jωt). 평가 가능 + 분석 결과 들고 있음.

합성 시 타입 승격은 DESIGN §4.1.2 표를 따른다.
"""

from __future__ import annotations

from typing import Callable, overload

import numpy as np
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

TimeArray = NDArray[np.float64]
"""1-D array of time samples, shape (N,)."""

RealArray = NDArray[np.float64]
"""1-D array of real samples, shape (N,)."""

ComplexArray = NDArray[np.complex128]
"""1-D array of complex samples, shape (N,)."""

RealSignalFn = Callable[[TimeArray], NDArray]
"""실수 신호 함수의 시그니처. t 배열을 받아 실수(또는 호환 가능한) 배열을 반환."""

ComplexSignalFn = Callable[[TimeArray], NDArray]
"""복소 신호 함수의 시그니처. t 배열을 받아 복소(또는 호환 가능한) 배열을 반환."""


# ─────────────────────────────────────────────────────────────────────────────
# Signal — 실수 시간 함수 (기본 클래스)
# ─────────────────────────────────────────────────────────────────────────────

class Signal:
    """
    실수 연속 시간 신호의 추상 표현.

    ``signals.cosine(...)`` / ``gaussian(...)`` 등 표준 팩토리, 또는
    ``Signal.from_lambda(...)``로 생성.

    합성 연산 (`+`, `-`, `*`, unary `-`)은 새로운 ``Signal`` 또는
    ``ComplexSignal``을 반환. 승격 규칙은 DESIGN §4.1.2 참조.

    Examples
    --------
    >>> sig = Signal.from_lambda(lambda t: np.cos(2*np.pi*2*t))
    >>> samples = sig.sample(np.linspace(0, 1, 100))
    >>> assert samples.dtype == np.float64
    """

    name: str
    """디버깅·UI 라벨용. 합성 시 자동으로 합성된 이름이 붙는다."""

    @classmethod
    def from_lambda(cls, fn: RealSignalFn, *, name: str = "<lambda>") -> "Signal":
        """Python callable을 실수 Signal로 감싼다.

        Notes
        -----
        ``fn``이 복소 배열을 반환할 가능성이 있다면 ``ComplexSignal.from_lambda``를 쓸 것.
        본 메서드는 결과를 float64로 캐스팅한다.
        """
        ...

    def sample(self, t: TimeArray) -> RealArray:
        """주어진 시간 배열에서 신호를 평가. float64 반환."""
        ...

    # ── 합성 연산 ─────────────────────────────────────────────────────────
    @overload
    def __add__(self, other: "ComplexSignal") -> "ComplexSignal": ...
    @overload
    def __add__(self, other: "Signal | float") -> "Signal": ...
    @overload
    def __add__(self, other: complex) -> "ComplexSignal": ...
    def __add__(self, other): ...

    def __radd__(self, other: float | complex) -> "Signal | ComplexSignal": ...
    def __sub__(self, other: "Signal | float | complex") -> "Signal | ComplexSignal": ...
    def __rsub__(self, other: float | complex) -> "Signal | ComplexSignal": ...
    def __mul__(self, other: "Signal | float | complex") -> "Signal | ComplexSignal": ...
    def __rmul__(self, other: float | complex) -> "Signal | ComplexSignal": ...
    def __neg__(self) -> "Signal": ...

    def __repr__(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# ComplexSignal — 복소 시간 함수
# ─────────────────────────────────────────────────────────────────────────────

class ComplexSignal(Signal):
    """
    복소 연속 시간 신호. ``Signal``의 자식이므로 모든 ``Signal`` 자리에 들어갈 수 있다
    (LSP).

    ``signals.complex_exp(omega=...)``가 대표적인 인스턴스 생성처. ``Signal`` 인스턴스에
    복소 스칼라를 곱하거나 ``ComplexSignal``을 더하면 합성 결과로도 만들어진다.

    Examples
    --------
    >>> z = signals.complex_exp(omega=2*np.pi)
    >>> assert isinstance(z, ComplexSignal)
    >>> samples = z.sample(np.linspace(0, 1, 100))
    >>> assert samples.dtype == np.complex128
    """

    @classmethod
    def from_lambda(cls, fn: ComplexSignalFn, *, name: str = "<lambda>") -> "ComplexSignal":  # type: ignore[override]
        """Python callable을 복소 Signal로 감싼다."""
        ...

    def sample(self, t: TimeArray) -> ComplexArray:  # type: ignore[override]
        """주어진 시간 배열에서 신호를 평가. complex128 반환."""
        ...

    def __neg__(self) -> "ComplexSignal": ...  # type: ignore[override]


# ─────────────────────────────────────────────────────────────────────────────
# WoundSignal — x(t)·e^(-jωt), ComplexSignal의 자식
# ─────────────────────────────────────────────────────────────────────────────

class WoundSignal(ComplexSignal):
    """
    하나의 winding 주파수에서 신호를 감은 결과.

    ``ComplexSignal``을 상속하므로 임의의 t 배열에 대해 다시 ``.sample(t)`` 호출 가능
    (닫힌 형태 ``x(t)·e^(-jωt)``를 그대로 평가). 동시에 ``FourierAnalyzer``가 사전
    샘플링한 분석 결과(아래 attribute들)를 그대로 들고 있어, viz 레이어가 다시 계산
    없이 바로 그릴 수 있다.

    Attributes
    ----------
    base : Signal
        원본 신호 x(t).
    omega : float
        winding angular frequency (rad/s).
    t : TimeArray
        FourierAnalyzer가 샘플링한 시간 배열, shape (N,).
    signal_values : ComplexArray
        x(t)의 표본, shape (N,).
    wound_values : ComplexArray
        x(t)·e^(-jωt)의 표본, shape (N,). DESIGN §2.1 (c).
    running_integral : ComplexArray
        ∫_{t[0]}^{t[k]} x(τ)·e^(-jωτ) dτ의 사다리꼴 누적합, shape (N,). DESIGN §2.1 (d).
        running_integral[0] == 0.
    """

    base: Signal
    omega: float
    t: TimeArray
    signal_values: ComplexArray
    wound_values: ComplexArray
    running_integral: ComplexArray

    @property
    def final_integral(self) -> complex:
        """``running_integral[-1]``. 한 ω에서의 X(jω) 기여 (Mode B의 한 점)."""
        ...

    def sample(self, t: TimeArray) -> ComplexArray:  # type: ignore[override]
        """임의의 t 배열에 대해 ``base(t) · e^(-jωt)``를 다시 평가. 보간 아님."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# signals — 표준 신호 팩토리
# ─────────────────────────────────────────────────────────────────────────────
#
# 모든 팩토리는 omega를 rad/s로 받는다. Hz가 익숙한 경우 omega = 2π·f로 직접 변환.

class signals:
    """표준 신호 팩토리들의 namespace."""

    @staticmethod
    def cosine(omega: float, amp: float = 1.0, phase: float = 0.0) -> Signal:
        """A·cos(ω·t + phase)."""
        ...

    @staticmethod
    def sine(omega: float, amp: float = 1.0, phase: float = 0.0) -> Signal:
        """A·sin(ω·t + phase)."""
        ...

    @staticmethod
    def complex_exp(omega: float, amp: complex = 1.0 + 0j) -> ComplexSignal:
        """A·exp(j·ω·t). ω는 양/음 모두 가능. 항상 ComplexSignal 반환."""
        ...

    @staticmethod
    def gaussian(sigma: float = 1.0, center: float = 0.0, amp: float = 1.0) -> Signal:
        """A·exp(-(t-center)² / (2σ²))."""
        ...

    @staticmethod
    def rect(width: float = 1.0, center: float = 0.0, amp: float = 1.0) -> Signal:
        """폭 width, 중심 center, 높이 amp인 사각 펄스."""
        ...

    @staticmethod
    def step(t0: float = 0.0, amp: float = 1.0) -> Signal:
        """t < t0 은 0, t ≥ t0 은 amp."""
        ...

    @staticmethod
    def decaying_exp(rate: float = 1.0, amp: float = 1.0, t0: float = 0.0) -> Signal:
        """t < t0 은 0, t ≥ t0 은 amp·exp(-rate·(t-t0)). (causal)

        FT: X(jω) = amp · e^(-jωt0) / (rate + jω). 무한 적분 가능.
        """
        ...

    @staticmethod
    def two_sided_exp(rate: float = 1.0, amp: float = 1.0,
                      center: float = 0.0) -> Signal:
        """A·exp(-rate·|t - center|). 짝함수, ∫|x|dt = 2A/rate < ∞.

        FT: X(jω) = 2·A·rate / (rate² + ω²) — 순수 실수, 짝함수성 직접 확인 가능.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# FourierAnalyzer — 모든 적분/감기 계산의 진입점
# ─────────────────────────────────────────────────────────────────────────────

class FourierAnalyzer:
    """
    한 신호와 한 적분 구간을 묶어, 임의의 winding angular frequency에 대해 분석을 제공.

    Examples
    --------
    >>> sig = signals.cosine(omega=2*np.pi)
    >>> analyzer = FourierAnalyzer(sig, t_min=0.0, t_max=10.0, n_samples=2000)
    >>> ws = analyzer.wound_at_omega(2*np.pi)
    >>> assert isinstance(ws, WoundSignal)
    >>> ws.final_integral.real  # ≈ T/2 = 5 (살아남는 성분)
    """

    signal: Signal
    t_min: float
    t_max: float
    n_samples: int

    def __init__(
        self,
        signal: Signal,
        *,
        t_min: float = 0.0,
        t_max: float = 4.0,
        n_samples: int = 500,
    ) -> None: ...

    @property
    def t(self) -> TimeArray:
        """샘플링된 시간 배열 (캐시됨)."""
        ...

    def wound_at_omega(self, omega: float) -> WoundSignal:
        """angular frequency ω(rad/s)로 신호를 감은 결과를 ``WoundSignal``로 반환."""
        ...

    # ── Mode B (v0.2): Spectrum & Inverse accumulation ───────────────────
    def spectrum(
        self,
        omega_min: float,
        omega_max: float,
        n_omega: int = 200,
    ) -> tuple[NDArray[np.float64], ComplexArray]:
        """
        [omega_min, omega_max] 범위(양/음 모두 가능)의 ω에 대해 X(jω)를 계산.

        Returns
        -------
        omegas : NDArray[float64], shape (n_omega,)
        X : ComplexArray, shape (n_omega,)
        """
        ...

    def inverse_accumulation(
        self,
        t_fix: float,
        omega_min: float,
        omega_max: float,
        n_omega: int = 200,
        order: str = "monotonic",
    ) -> tuple[NDArray[np.float64], ComplexArray, ComplexArray]:
        """
        고정 시점 ``t_fix``에서, ω를 누적하며 역변환 화살표 시퀀스 반환.

        Parameters
        ----------
        t_fix : float
        omega_min, omega_max : float
        n_omega : int, default 200
        order : 'monotonic' | 'paired_by_abs', default 'monotonic'
            'monotonic'은 omega_min→omega_max 순서.
            'paired_by_abs'는 |ω| 작은 것부터 큰 것 순으로 누적해 켤레쌍이
            인접하게 더해짐 → 실수 신호의 *허수부 cancel*이 가장 명료하게 보임.

        Returns
        -------
        omegas : NDArray[float64], shape (n_omega,)  (누적 순서로 정렬됨)
        arrows : ComplexArray, shape (n_omega,)
            각 ω의 기여 ``(1/2π) · X(jω) · e^(jωt_fix) · dω``.
        accumulated : ComplexArray, shape (n_omega,)
            머리 잇기 누적합. ``accumulated[-1] ≈ x(t_fix)``.
        """
        ...
