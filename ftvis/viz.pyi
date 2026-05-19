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


# ─────────────────────────────────────────────────────────────────────────────
# 역변환용 (b') Inverse winding helix — ω축 위의 단위 헬릭스
# ─────────────────────────────────────────────────────────────────────────────

class InverseWindingHelixPlot(PlotPanel):
    """
    e^(+jωt_fix)의 3D 단위 헬릭스 — 역변환 winding 회전자.

    순변환에서 `e^(-jωt)`가 *t축 위의 헬릭스*였듯이, 역변환에서는
    `e^(+jωt_fix)`가 *ω축 위의 헬릭스*다. 각속도가 `t_fix`인 단위 헬릭스.

    Mode A의 ``update(wound, t_index)``는 no-op이고, 데이터는
    ``show_helix(omegas, t_fix)``로 직접 박는다.
    """

    def __init__(self) -> None: ...

    @property
    def figure(self) -> go.Figure: ...

    def show_helix(self, omegas: np.ndarray, t_fix: float) -> None:
        """
        Parameters
        ----------
        omegas : NDArray[float64]
            ω 범위.
        t_fix : float
            고정 시간 위치. 헬릭스의 각속도.
        """
        ...

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# 역변환용 (d') Inverse integral 3D — ω축 위의 누적 적분 trail
# ─────────────────────────────────────────────────────────────────────────────

class InverseIntegral3DPlot(PlotPanel):
    """
    역변환 누적 적분 ``(1/2π)·∫ X(jω)·e^(jωt_fix) dω``를 3D 누적 trail로.

    순변환 ``ForwardIntegralPlot``의 ω-domain 대응물. 좌표계: ω축 + 복소평면.
    각 ω에서 원점에서 누적 끝점으로 향하는 화살표와 trail.

    Parameters
    ----------
    show_complex_plane_inset : bool, default True
        오른쪽에 작은 2D 복소평면 inset 표시.
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

    def show_accumulation(
        self,
        omegas: np.ndarray,
        accumulated: np.ndarray,
        *,
        progress_index: int | None = None,
        flatten_noise: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        omegas : NDArray[float64]
            누적 *순서*에 따라 정렬된 ω 배열 (monotonic 권장).
        accumulated : ComplexArray
            머리 잇기 누적합. accumulated[-1] ≈ x(t_fix).
        progress_index : int, optional
            None이면 전부. 정수면 처음 k+1개만.
        flatten_noise : bool, default True
            trail의 Re/Im 한 쪽이 max|r| 대비 1e-9 이하면 그 축 범위를 살아있는
            축과 같게 강제해 평면에 누여 보이게 함. SpectrumPlot의
            ``flatten_noise``와 동일 패턴.
        """
        ...

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# PartitionedAccumulationPlot — 단위벡터 + 누적 호 시각화
# ─────────────────────────────────────────────────────────────────────────────

class PartitionedAccumulationPlot(PlotPanel):
    """
    한 신호 구간의 적분을 *단위원 위 단위벡터들의 합 × dt* 로 보여주는 2D 패널.

    rect→sinc 같은 적분의 기하적 유도에 쓰임. ``boundary`` 인자로 "한 주기에
    cancel되는 묶음(dim) + 잔여 호(bright)" 분기 가능.

    Parameters
    ----------
    show_unit_circle : bool, default True
        점선 단위원 표시.
    show_chord : bool, default True
        잔여 호의 양 끝점을 잇는 chord. 길이 = 2sin(잔여각/2).
    arrow_width : float, default 1.0
        단위벡터 화살표 선 굵기.
    """

    def __init__(
        self,
        *,
        show_unit_circle: bool = True,
        show_chord: bool = True,
        arrow_width: float = 1.0,
    ) -> None: ...

    @property
    def figure(self) -> go.Figure: ...

    def show_partition(
        self,
        unit_vectors: np.ndarray,
        dt: float,
        *,
        boundary: int = 0,
        label: str | None = None,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        unit_vectors : ComplexArray, shape (N,)
            단위원 위 N개 점 ``e^(-jωt_k)``.
        dt : float
            적분 가중치. ``arrows = unit_vectors * dt``.
        boundary : int, default 0
            dim/bright 경계 인덱스. 0이면 전부 bright (Phase 1).
        label : str, optional
            패널 제목.
        x_range, y_range : (float, float), optional
            축 범위.
        """
        ...

    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# WoundSpectrumPlot — X(jω)·e^(-jωα) 3D helix + |X| 회전체 envelope
# ─────────────────────────────────────────────────────────────────────────────

class WoundSpectrumPlot(PlotPanel):
    """
    ``X(jω)·e^(-jωα)``의 3D 시각화. 시간 도메인 ``WoundSignalPlot``에 대응되는
    주파수 도메인 버전. helix 본체 + ``|X(jω)|`` 회전체 envelope.

    Parameters
    ----------
    show_envelope : bool, default True
        ``|X(jω)|`` 회전체 envelope 표시.
    envelope_alpha : float, default 0.15
        envelope 투명도.
    """

    def __init__(
        self,
        *,
        show_envelope: bool = True,
        envelope_alpha: float = 0.15,
    ) -> None: ...

    @property
    def figure(self) -> go.Figure: ...

    def show_wound_spectrum(
        self,
        omegas: np.ndarray,
        X: np.ndarray,
        alpha: float = 0.0,
        *,
        label: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        omegas : NDArray[float64]
        X : ComplexArray
            원본 spectrum.
        alpha : float, default 0.0
            linear phase. ``X·e^(-jωα)``. 0이면 X 그대로.
        label : str, optional
            제목.
        """
        ...

    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# PartialIntegralComparisonPlot — 여러 partial 적분 trail을 한 3D scene에 비교
# ─────────────────────────────────────────────────────────────────────────────

class PartialIntegralComparisonPlot(PlotPanel):
    """
    여러 partial 적분 trail을 한 3D scene에 함께 그려 비교.

    좌표: x = ω, (y, z) = (Re partial, Im partial). Time shift property 시각
    증명, 다양한 적분 핵의 비교 등에 적합.
    """

    def __init__(self) -> None: ...

    @property
    def figure(self) -> go.Figure: ...

    def show_trails(
        self,
        omegas: np.ndarray,
        trails: list[dict],
        *,
        label: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        omegas : NDArray[float64]
            균등 ω 분할.
        trails : list of dict
            각 dict 키:

            - ``'kernel'`` : ComplexArray — 적분 핵.
            - ``'t'`` : float — ``e^(jωt)`` 곱.
            - ``'label'`` : str — legend.
            - ``'color'`` : str — 선 색.
            - ``'width'`` : float (optional, default 4.0).
            - ``'dash'`` : str (optional) — plotly dash style.
            - ``'marker_symbol'`` : str (optional, default 'circle').
        label : str, optional
            전체 제목.
        """
        ...

    def update(self, wound: WoundSignal, t_index: int) -> None: ...
