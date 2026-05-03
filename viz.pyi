"""
ftvis.viz — Layer 2: Visualization primitives.

각 플롯 클래스는 §2.1의 다섯 시각 요소 중 하나에 대응한다.
모든 플롯은 동일한 갱신 인터페이스를 따른다:
    plot.update(wound: WoundSignal, t_index: int) -> None
    plot.figure -> plotly.graph_objects.FigureWidget

Layer 1과 달리 Plotly 의존성이 있다. 향후 백엔드 교체 가능성을 위해
백엔드 중립 추상 베이스 (PlotPanel)를 두고, Plotly 구현은 그 서브클래스로 둔다.
v0.1에서는 Plotly 구현만 제공.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol

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
    def figure(self) -> go.FigureWidget:
        """ipywidgets에 박을 수 있는 FigureWidget. 첫 호출 시 lazy 생성."""
        ...

    @abstractmethod
    def update(self, wound: WoundSignal, t_index: int) -> None:
        """현재 시점 t_index와 wound 데이터로 패널 내용을 갱신.

        Parameters
        ----------
        wound : WoundSignal
            현재 ω에서의 감긴 신호와 누적 적분.
        t_index : int
            모드 A의 현재 시간 위치 (0 ≤ t_index < len(wound.t)).
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
    def figure(self) -> go.FigureWidget: ...
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
    def figure(self) -> go.FigureWidget: ...
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
    def figure(self) -> go.FigureWidget: ...
    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# (d) Running integral vector — *메인 디시*
# ─────────────────────────────────────────────────────────────────────────────

class RunningIntegralPlot(PlotPanel):
    """
    누적 적분 벡터 ∫₀ᵗ x(τ)·e^(-jωτ) dτ를 보여주는 플롯.

    두 가지 view를 동시에 제공한다 (subplot으로 묶음):

    * 3D view: 시간 축 + 복소평면. 시점 t에서 원점에서 출발해 누적값으로 향하는
      벡터를 그리고, 그 끝점이 그리는 궤적도 함께 표시.
    * 2D projection: 복소평면(Re-Im)에 누적 벡터의 끝점만 투영해 보여줌. 직교성을
      가장 직관적으로 드러내는 view.

    이 플롯이 §2.1 (d)의 시각화이며, 이 도구의 핵심 메시지를 담는다.

    Parameters
    ----------
    show_3d : bool, default True
    show_complex_plane : bool, default True
        둘 다 False면 에러. 둘 다 True면 좌우 또는 상하 분할 레이아웃.
    trail_alpha : float, default 0.4
        끝점 궤적의 투명도.
    """

    def __init__(
        self,
        *,
        show_3d: bool = True,
        show_complex_plane: bool = True,
        trail_alpha: float = 0.4,
    ) -> None: ...

    @property
    def figure(self) -> go.FigureWidget: ...
    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# (e) Spectrum — Mode B placeholder (v0.1에서는 'coming soon' 박스)
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumPlot(PlotPanel):
    """
    X(f) 스펙트럼 곡선. v0.2 (Mode B)에서 활성화.

    v0.1에서는 update() 호출 시 "Mode B coming soon" 텍스트만 표시하는
    placeholder로 동작한다. UI 레이아웃 자리를 미리 확보해두는 게 목적.
    """

    def __init__(
        self,
        *,
        view: Literal["re_im", "mag_phase"] = "mag_phase",
    ) -> None:
        """
        view : 'mag_phase'는 두 줄(magnitude, phase), 're_im'은 두 줄(Re, Im).
        """
        ...

    @property
    def figure(self) -> go.FigureWidget: ...
    def update(self, wound: WoundSignal, t_index: int) -> None:
        """v0.1에서는 no-op (placeholder)."""
        ...
