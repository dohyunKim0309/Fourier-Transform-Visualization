"""
ftvis.viz — Layer 2: Visualization primitives.

각 플롯 클래스는 DESIGN §2.1의 여섯 시각 요소 중 하나에 대응한다.
모든 플롯은 동일한 갱신 인터페이스를 따른다::

    plot.update(wound: WoundSignal, t_index: int) -> None
    plot.figure -> plotly.graph_objects.Figure
    # Jupyter 인터랙티브: go.FigureWidget(plot.figure)로 감싸 사용

Layer 1과 달리 Plotly 의존성이 있다. 향후 백엔드 교체 가능성을 위해
백엔드 중립 추상 베이스 (PlotPanel)를 두고, Plotly 구현은 그 서브클래스로 둔다.
v0.1에서는 Plotly 구현만 제공.

v0.1 구현 대상: SignalPlot, WindingHelixPlot, WoundSignalPlot, ForwardIntegralPlot.
v0.2 placeholder: SpectrumPlot, InverseAccumulationPlot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import plotly.graph_objects as go

from ftvis.core import WoundSignal


# ─────────────────────────────────────────────────────────────────────────────
# Backend-agnostic protocol
# ─────────────────────────────────────────────────────────────────────────────

class PlotPanel(ABC):
    """모든 viz 컴포넌트의 추상 베이스.

    이 인터페이스를 통해서만 app 레이어가 viz 레이어를 만진다.
    """

    @property
    @abstractmethod
    def figure(self) -> go.Figure:
        """Plotly Figure. 첫 호출 시 lazy 생성. ipywidgets/Jupyter 인터랙티브가
        필요하면 사용자가 ``go.FigureWidget(panel.figure)``로 감싸면 된다."""
        ...

    @abstractmethod
    def update(self, wound: WoundSignal, t_index: int) -> None:
        """현재 시점 t_index와 wound 데이터로 패널 내용을 갱신.

        Parameters
        ----------
        wound : WoundSignal
            현재 ω에서의 감긴 신호와 누적 적분 (Mode A의 모든 데이터).
        t_index : int
            현재 시간 위치 (0 ≤ t_index < len(wound.t)).
            예: 재생 중인 t = wound.t[t_index].
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# (a) Signal curve — 시간 도메인 곡선
# ─────────────────────────────────────────────────────────────────────────────

class SignalPlot(PlotPanel):
    """
    `x(t)`의 2D 곡선. 복소 신호인 경우 Re/Im을 별도 트레이스로 표시.

    Parameters
    ----------
    show_imag : bool
        신호가 복소수일 때 Im 부분도 그릴지 여부. 기본 True.
    cursor_color : str
        현재 t 위치를 표시하는 수직선 색상.
    """

    def __init__(
        self,
        *,
        show_imag: bool = True,
        cursor_color: str = "white",
    ) -> None: ...

    @property
    def figure(self) -> go.Figure: ...
    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# (b) Winding helix — 단위 헬릭스
# ─────────────────────────────────────────────────────────────────────────────

class WindingHelixPlot(PlotPanel):
    """
    e^(-jωt)의 3D 단위 헬릭스. 시간축이 한 축, 복소평면이 나머지 두 축.

    교육적으로는 (c) WoundSignalPlot과 같은 화면에 겹쳐 보여주는 게 더 효과적이다.
    하지만 별도 패널로도 쓸 수 있도록 단독 클래스로 둔다.
    """

    def __init__(self) -> None: ...

    @property
    def figure(self) -> go.Figure: ...
    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# (c) Wound signal — 변조된 헬릭스
# ─────────────────────────────────────────────────────────────────────────────

class WoundSignalPlot(PlotPanel):
    """
    x(t)·e^(-jωt)의 3D 곡선.

    시간 축 하나 + 복소평면 두 축으로 이루어진 3D 씬에서, x(t)가 envelope, helix가
    carrier인 곡선을 그린다. 현재 시점 t_index는 작은 마커로 표시.
    """

    def __init__(
        self,
        *,
        show_envelope: bool = True,
    ) -> None:
        """
        show_envelope : bool
            x(t)의 절댓값을 envelope tube로 함께 그릴지. 시각적으로 도움 됨.
        """
        ...

    @property
    def figure(self) -> go.Figure: ...
    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# (d) Forward Running Integral — *Mode A의 메인 디시*
# ─────────────────────────────────────────────────────────────────────────────

class ForwardIntegralPlot(PlotPanel):
    """
    Wound signal의 누적 적분을 3D 누적 벡터로 보여주는 플롯.

    좌표계는 (c) WoundSignalPlot과 동일한 시간 축 + 복소평면 3D. 각 시점 t에서
    원점에서 출발해 ``running_integral[k]``로 향하는 화살표를 그리고, 화살표 끝점이
    그리는 궤적(시간이 흐를수록 길어지는 trail)을 함께 표시한다.

    학습 메시지 (DESIGN §2.1 (d)):

    * ω가 신호 성분과 일치 → 정지 성분이 살아남아 화살표가 한 방향으로 자라남.
    * ω ≠ 신호 성분 → 회전 성분만 남아 화살표 끝이 작은 원을 그리며 제자리를 맴돔.

    Parameters
    ----------
    show_complex_plane_inset : bool, default True
        오른쪽 위에 작은 2D 복소평면 inset을 띄워 누적 벡터의 끝점 궤적을 위에서
        본 모습을 함께 보여줄지. 직교성 직관에 도움.
    trail_alpha : float, default 0.4
        끝점 궤적의 투명도.
    """

    def __init__(
        self,
        *,
        show_complex_plane_inset: bool = True,
        trail_alpha: float = 0.4,
    ) -> None: ...

    @property
    def figure(self) -> go.Figure: ...
    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# (e) Spectrum — Mode B 본격 구현
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumPlot(PlotPanel):
    """
    X(jω)를 ω축 + 복소평면 3D로 시각화. Mode A의 wound signal과 좌표계 일치.

    Mode A의 ``update(wound, t_index)``는 no-op이고, 데이터는
    ``show_spectrum(omegas, X)``로 직접 박는다.

    Parameters
    ----------
    view : 'mag_phase' | 're_im', default 're_im'
        2D 인셋의 표시 방식.
    show_inset : bool, default True
        오른쪽에 2D 보조 플롯을 띄울지.
    """

    def __init__(
        self,
        *,
        view: Literal["re_im", "mag_phase"] = "re_im",
        show_inset: bool = True,
    ) -> None: ...

    @property
    def figure(self) -> go.Figure: ...
    def show_spectrum(self, omegas: np.ndarray, X: np.ndarray,
                      *, flatten_noise: bool = True) -> None:
        """spectrum 데이터를 받아 그린다.

        Parameters
        ----------
        omegas, X : 데이터.
        flatten_noise : bool, default True
            Re/Im 한 쪽이 max|X| 대비 1e-9 이하일 때 그 축 범위를 다른 쪽과
            같게 강제해 평면처럼 누이기. 짝/홀 함수의 ``X(jω)`` 순수 실수/허수
            메시지를 시각에 보존.
        """
        ...
    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# (f) Inverse Accumulation — Mode B 본격 구현
# ─────────────────────────────────────────────────────────────────────────────

class InverseAccumulationPlot(PlotPanel):
    """
    역변환 누적의 2D 복소평면 시각화. 화살표 머리 잇기.

    데이터는 ``show_accumulation(omegas, arrows, accumulated, target,
    progress_index)``로 직접 박는다.

    Parameters
    ----------
    show_target : bool, default True
        목표값 ``x(t_fix)``를 X 마커로 표시.
    arrow_alpha : float, default 0.5
        개별 화살표 투명도.
    """

    def __init__(
        self,
        *,
        show_target: bool = True,
        arrow_alpha: float = 0.5,
    ) -> None: ...

    @property
    def figure(self) -> go.Figure: ...
    def show_accumulation(
        self,
        omegas: np.ndarray,
        arrows: np.ndarray,
        accumulated: np.ndarray,
        target: complex,
        progress_index: int | None = None,
    ) -> None: ...
    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        ...
