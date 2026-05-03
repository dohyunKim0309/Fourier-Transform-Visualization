"""
ftvis.core — Layer 1: Pure mathematical primitives.

이 레이어는 numpy 외 의존성이 없고, 어떤 렌더러에 대해서도 무지하다.
모든 함수는 결정론적이며, 동일 입력에 동일 출력을 보장한다.

Conventions
-----------
- 시간 t: 단위 second.
- 주파수 f: 단위 Hz (cycles/second). 내부에서 ω = 2π·f로 변환되어 사용.
- 사용자 facing API는 freq (Hz)를 기본으로 하되, 명시적으로 omega를 받는
  메서드는 이름에 'omega'를 명시한다. 둘을 섞지 않는다.
- 복소수 표현은 numpy의 complex128. (Re, Im) 분리가 필요한 경우는
  consumer 쪽에서 .real / .imag 으로 처리.
"""

from __future__ import annotations

from typing import Callable, Sequence, overload

import numpy as np
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

TimeArray = NDArray[np.float64]
"""1-D array of time samples, shape (N,)."""

ComplexArray = NDArray[np.complex128]
"""1-D array of complex samples, shape (N,)."""

SignalFn = Callable[[TimeArray], NDArray[np.complex128]]
"""사용자 정의 신호 함수의 시그니처. t 배열을 받아 복소(또는 실수) 배열을 반환."""


# ─────────────────────────────────────────────────────────────────────────────
# Signal — 시간 도메인 신호 표현
# ─────────────────────────────────────────────────────────────────────────────

class Signal:
    """
    연속 시간 신호의 추상 표현.

    내부적으로는 sample-able한 callable을 들고 있다. SymPy 심볼릭 표현은
    v0.1에서 지원하지 않는다(Open Question §8).

    Examples
    --------
    >>> sig = Signal.from_lambda(lambda t: np.cos(2*np.pi*2*t))
    >>> samples = sig.sample(np.linspace(0, 1, 100))

    Composition은 자연스럽게 동작한다:

    >>> combined = signals.cosine(freq=2) + signals.cosine(freq=3)
    >>> scaled = 0.5 * signals.gaussian(sigma=0.3)
    """

    name: str
    """디버깅·UI 라벨용. 합성 시 자동으로 'cos(2Hz) + cos(3Hz)'처럼 만들어진다."""

    @classmethod
    def from_lambda(cls, fn: SignalFn, *, name: str = "<lambda>") -> "Signal":
        """Python 람다(또는 임의 callable)를 Signal로 감싼다."""
        ...

    def sample(self, t: TimeArray) -> ComplexArray:
        """주어진 시간 배열에서 신호를 평가한다. 항상 complex128로 반환."""
        ...

    # ── 합성 연산 ──────────────────────────────────────────────────────────
    def __add__(self, other: "Signal | float | complex") -> "Signal": ...
    def __radd__(self, other: float | complex) -> "Signal": ...
    def __sub__(self, other: "Signal | float | complex") -> "Signal": ...
    def __mul__(self, other: "Signal | float | complex") -> "Signal": ...
    def __rmul__(self, other: float | complex) -> "Signal": ...
    def __neg__(self) -> "Signal": ...

    def __repr__(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# signals — 표준 신호 팩토리
# ─────────────────────────────────────────────────────────────────────────────
#
# 모든 팩토리는 freq를 Hz로 받는다. 사용자가 ω로 입력하고 싶다면 freq=omega/(2π)로
# 직접 변환해서 넣어야 한다 — 이 단계의 명시성이 학습 도구로서는 의도된 마찰이다.

class signals:
    """표준 신호 팩토리들의 namespace."""

    @staticmethod
    def cosine(freq: float, amp: float = 1.0, phase: float = 0.0) -> Signal:
        """A·cos(2π·freq·t + phase). freq는 Hz, phase는 radian."""
        ...

    @staticmethod
    def sine(freq: float, amp: float = 1.0, phase: float = 0.0) -> Signal:
        """A·sin(2π·freq·t + phase)."""
        ...

    @staticmethod
    def complex_exp(freq: float, amp: complex = 1.0 + 0j) -> Signal:
        """A·exp(j·2π·freq·t). 양/음의 freq 모두 가능."""
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
        """t < t0 은 0, t ≥ t0 은 amp인 단위 계단."""
        ...

    @staticmethod
    def decaying_exp(rate: float = 1.0, amp: float = 1.0, t0: float = 0.0) -> Signal:
        """t < t0 은 0, t ≥ t0 은 amp·exp(-rate·(t-t0)). (causal)"""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# WoundSignal — x(t)·e^(-jωt)와 그 누적 적분 결과를 담는 데이터 객체
# ─────────────────────────────────────────────────────────────────────────────

class WoundSignal:
    """
    하나의 winding 주파수에서 신호를 감았을 때의 결과 묶음.

    이 객체는 데이터 컨테이너이고, 어떤 플롯도 그리지 않는다. viz 레이어가 이걸
    소비한다.

    Attributes
    ----------
    t : TimeArray
        샘플링된 시간 배열, shape (N,).
    omega : float
        winding 시 사용된 angular frequency (rad/s).
    freq : float
        omega / (2π). 편의용.
    signal_values : ComplexArray
        x(t), shape (N,). 원본 신호의 표본.
    wound : ComplexArray
        x(t)·e^(-jωt), shape (N,). §2.1 (c)에 해당.
    running_integral : ComplexArray
        ∫₀ᵗ x(τ)·e^(-jωτ) dτ의 누적합 근사, shape (N,). §2.1 (d).
        running_integral[k]는 t[0]부터 t[k]까지의 적분.
    """

    t: TimeArray
    omega: float
    freq: float
    signal_values: ComplexArray
    wound: ComplexArray
    running_integral: ComplexArray

    @property
    def final_integral(self) -> complex:
        """running_integral[-1]. Mode B에서 X(ω)의 한 점이 될 값."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# FourierAnalyzer — 모든 적분/감기 계산의 진입점
# ─────────────────────────────────────────────────────────────────────────────

class FourierAnalyzer:
    """
    한 신호와 한 적분 구간을 묶어, 임의의 winding 주파수에 대해 분석을 제공.

    적분 구간과 샘플 수는 생성자에서 고정된다 (사용자가 슬라이더로 바꿀 때마다
    재생성). 같은 (signal, t_min, t_max, n_samples) 조합에 대해 wound() 호출은
    여러 번 가능하다 — 각 호출이 다른 ω에 대응.

    Examples
    --------
    >>> sig = signals.cosine(freq=2)
    >>> analyzer = FourierAnalyzer(sig, t_min=0.0, t_max=4.0, n_samples=500)
    >>> ws = analyzer.wound_at_freq(2.0)  # f=2Hz로 감기
    >>> ws.final_integral  # 큰 값 (살아남는 성분)
    >>> ws2 = analyzer.wound_at_freq(2.5)
    >>> ws2.final_integral  # 작은 값 (cancel out)
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

    def wound_at_freq(self, freq: float) -> WoundSignal:
        """주파수 f(Hz)로 신호를 감은 결과를 반환. 내부적으로 omega=2π·freq."""
        ...

    def wound_at_omega(self, omega: float) -> WoundSignal:
        """angular frequency ω(rad/s)로 감은 결과. wound_at_freq의 저수준 버전."""
        ...

    # v0.2 — Mode B에서 활성화될 메서드. v0.1에서는 NotImplementedError.
    def spectrum(
        self,
        freq_min: float,
        freq_max: float,
        n_freq: int = 200,
    ) -> tuple[NDArray[np.float64], ComplexArray]:
        """
        [freq_min, freq_max] 범위의 주파수에 대해 X(f)를 계산.

        Returns
        -------
        freqs : NDArray[float64], shape (n_freq,)
        X : ComplexArray, shape (n_freq,)

        Notes
        -----
        v0.1에서는 NotImplementedError를 발생시킨다. v0.2에서 활성화.
        """
        ...
