"""
ftvis.viz — Layer 2: Plotly-based visualization primitives.

각 클래스는 DESIGN §2.1의 시각 요소 하나에 대응. 시그니처는 viz.pyi 참조.
v0.1: SignalPlot, WindingHelixPlot, WoundSignalPlot, ForwardIntegralPlot.
v0.2 placeholder: SpectrumPlot, InverseAccumulationPlot.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import plotly.graph_objects as go

from ftvis.core import ComplexSignal, WoundSignal


# ─────────────────────────────────────────────────────────────────────────────
# Theme — 3Blue1Brown-ish dark
# ─────────────────────────────────────────────────────────────────────────────

# 한 곳에 모아둬서 나중에 라이트 테마 토글이 쉬움.
_BG = "#0d1117"            # 거의 검정, GitHub dark 느낌
_PAPER = "#0d1117"
_FG = "#e6edf3"            # 글자·축
_GRID = "#30363d"           # 격자
_AXIS = "#7d8590"           # 축선

_GREEN = "#7ec699"          # signal·wound·forward integral 본체
_BLUE = "#58a6ff"           # winding helix
_ORANGE = "#ffa657"         # 강조·envelope·forward 화살표
_PURPLE = "#bf91f3"         # imag part
_WHITE = "#f0f6fc"          # 마커·커서

_DEFAULT_LAYOUT = dict(
    paper_bgcolor=_PAPER,
    plot_bgcolor=_BG,
    font=dict(color=_FG, family="serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    showlegend=False,
)

_2D_AXIS = dict(
    showgrid=True,
    gridcolor=_GRID,
    zeroline=True,
    zerolinecolor=_AXIS,
    linecolor=_AXIS,
    color=_FG,
)

_3D_AXIS = dict(
    backgroundcolor=_BG,
    gridcolor=_GRID,
    zerolinecolor=_AXIS,
    showbackground=True,
    color=_FG,
)


def _new_2d_figure(title: str = "", height: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_DEFAULT_LAYOUT,
        title=dict(text=title, x=0.02, xanchor="left", font=dict(color=_FG)),
        height=height,
        xaxis=_2D_AXIS,
        yaxis=_2D_AXIS,
    )
    return fig


def _new_3d_figure(title: str = "", height: int = 480) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_DEFAULT_LAYOUT,
        title=dict(text=title, x=0.02, xanchor="left", font=dict(color=_FG)),
        height=height,
        scene=dict(
            xaxis=dict(_3D_AXIS, title="t"),
            yaxis=dict(_3D_AXIS, title="Re"),
            zaxis=dict(_3D_AXIS, title="Im"),
            aspectmode="manual",
            aspectratio=dict(x=2.0, y=1.0, z=1.0),
            camera=dict(eye=dict(x=2.0, y=1.6, z=1.0)),
        ),
    )
    return fig


def _make_central_axis_trace(scene: str = "scene") -> go.Scatter3d:
    """3D 씬에 (t, 0, 0) 라인을 굵고 진하게 그어주는 trace.

    helix·wound·forward integral 모두에서 회전의 기준축이 한눈에 보이게 한다.
    실제 데이터는 update에서 채우고, 여기서는 빈 trace를 만들어둔다.
    """
    return go.Scatter3d(
        x=[], y=[], z=[],
        mode="lines",
        line=dict(color=_FG, width=5),  # _AXIS → _FG, width 2→5
        opacity=1.0,
        name="t-axis",
        showlegend=False,
        scene=scene,
    )


def _set_central_axis_data(trace: go.Scatter3d, t: np.ndarray) -> None:
    """trace에 (t, 0, 0) 데이터를 박는다. 양 끝점만 있으면 충분."""
    trace.x = [float(t[0]), float(t[-1])]
    trace.y = [0.0, 0.0]
    trace.z = [0.0, 0.0]


def _omega_pi_label(omega: float, tol: float = 1e-3) -> str:
    """ω 값을 π 배수 형식으로 라벨. 단순한 배수 (정수, 0.5 step)만 'kπ' 표기.

    예:
       4π     → '4π'
       3.5π   → '3.5π'
       2π·1.7 → '3.4π'
       7.0    → '7.0'    (π에서 동떨어진 값은 그냥 숫자)
       0      → '0'
    """
    omega_f = float(omega)
    if abs(omega_f) < 1e-9:
        return "0"
    ratio = omega_f / math.pi
    # 0.1 단위까지 깔끔한 ratio만 π 표기 (0.5 step 같은 자연스러운 값까지)
    rounded_one_decimal = round(ratio, 1)
    if abs(ratio - rounded_one_decimal) < tol:
        if rounded_one_decimal == int(rounded_one_decimal):
            n = int(rounded_one_decimal)
            if n == 1:
                return "π"
            if n == -1:
                return "-π"
            return f"{n}π"
        return f"{rounded_one_decimal:g}π"
    return f"{omega_f:.3g}"


def _pi_tickvals_and_labels(t_min: float, t_max: float, omega: float,
                            max_ticks: int = 16) -> tuple[list[float], list[str]]:
    """winding ω 기준으로 ωt = nπ가 되는 t 위치에 눈금 + 라벨 생성.

    라벨은 *그 위치에서의 ωt 값*을 π 배수로 표기. 사용자 입장에서는
    "이 눈금은 ωt = 2π인 곳" → '2π'. 시간 단위가 아니라 위상 단위.

    예: ω=4π, t∈[0,5]
       t=0      → ωt=0      → '0'
       t=0.25   → ωt=π      → 'π'
       t=0.5    → ωt=2π     → '2π'
       t=0.75   → ωt=3π     → '3π'
       ...

    ω가 0이거나 너무 작으면 빈 결과 반환 (caller가 plotly 자동 눈금 사용).
    """
    if omega is None or abs(omega) < 1e-9:
        return [], []

    abs_omega = abs(float(omega))
    period_half = math.pi / abs_omega

    n_min = math.ceil(t_min / period_half)
    n_max = math.floor(t_max / period_half)
    if n_max < n_min:
        return [], []

    n_count = n_max - n_min + 1
    step = max(1, math.ceil(n_count / max_ticks))

    vals: list[float] = []
    labels: list[str] = []
    for n in range(n_min, n_max + 1, step):
        vals.append(n * period_half)
        if n == 0:
            labels.append("0")
        elif n == 1:
            labels.append("π")
        elif n == -1:
            labels.append("-π")
        else:
            labels.append(f"{n}π")
    return vals, labels




# ─────────────────────────────────────────────────────────────────────────────
# Backend-agnostic abstract base
# ─────────────────────────────────────────────────────────────────────────────

class PlotPanel(ABC):
    """모든 viz 컴포넌트의 추상 베이스. 시그니처는 viz.pyi 참조."""

    def __init__(self) -> None:
        self._fig: Optional[go.Figure] = None

    @property
    def figure(self) -> go.Figure:
        """ipywidgets/Jupyter에서 인터랙티브하게 쓸 때는 사용자가 별도로
        ``go.FigureWidget(panel.figure)``로 감쌀 수 있다. v0.1 정적 사용에는 Figure로 충분."""
        if self._fig is None:
            self._fig = self._build_figure()
        return self._fig

    @abstractmethod
    def _build_figure(self) -> go.Figure:
        """첫 호출 시 빈 트레이스로 figure를 구성한다 (lazy)."""
        ...

    @abstractmethod
    def update(self, wound: WoundSignal, t_index: int) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# (a) SignalPlot
# ─────────────────────────────────────────────────────────────────────────────

class SignalPlot(PlotPanel):
    """x(t)의 2D 곡선. 복소 신호면 Re/Im 분리."""

    def __init__(self, *, show_imag: bool = True, cursor_color: str = _WHITE) -> None:
        super().__init__()
        self.show_imag = show_imag
        self.cursor_color = cursor_color

    def _build_figure(self) -> go.Figure:
        fig = _new_2d_figure(title="(a) x(t)", height=240)
        # trace 0: real, trace 1: imag, trace 2: cursor (수직선은 shape으로 처리)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line=dict(color=_GREEN, width=2),
                                 name="Re x(t)"))
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line=dict(color=_PURPLE, width=2, dash="dot"),
                                 name="Im x(t)",
                                 visible=False))
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker=dict(color=self.cursor_color, size=8),
                                 name="t cursor"))
        return fig

    def update(self, wound: WoundSignal, t_index: int) -> None:
        fig = self.figure
        x = wound.signal_values
        t = wound.t

        is_complex = bool(np.any(np.abs(x.imag) > 1e-12))

        fig.data[0].x = t
        fig.data[0].y = x.real
        if is_complex and self.show_imag:
            fig.data[1].x = t
            fig.data[1].y = x.imag
            fig.data[1].visible = True
        else:
            fig.data[1].visible = False
        i = int(np.clip(t_index, 0, len(t) - 1))
        cursor_y = float(x[i].real)
        fig.data[2].x = [float(t[i])]
        fig.data[2].y = [cursor_y]
        # t축 눈금: ωt = nπ 위치
        vals, labels = _pi_tickvals_and_labels(float(t[0]), float(t[-1]),
                                               wound.omega)
        if vals:
            fig.layout.xaxis.tickvals = vals
            fig.layout.xaxis.ticktext = labels


# ─────────────────────────────────────────────────────────────────────────────
# (b) WindingHelixPlot
# ─────────────────────────────────────────────────────────────────────────────

class WindingHelixPlot(PlotPanel):
    """e^(-jωt)의 단위 헬릭스 3D."""

    def __init__(self) -> None:
        super().__init__()

    def _build_figure(self) -> go.Figure:
        fig = _new_3d_figure(title="(b) e^(-jωt)", height=400)
        # trace 0: 중심축 (t, 0, 0)
        fig.add_trace(_make_central_axis_trace())
        # trace 1: helix
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(color=_BLUE, width=4),
                                   name="helix"))
        # trace 2: cursor
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                   marker=dict(color=_WHITE, size=4),
                                   name="cursor"))
        return fig

    def update(self, wound: WoundSignal, t_index: int) -> None:
        fig = self.figure
        t = wound.t
        helix = np.exp(-1j * wound.omega * t)
        i = int(np.clip(t_index, 0, len(t) - 1))
        _set_central_axis_data(fig.data[0], t)
        fig.data[1].x = t
        fig.data[1].y = helix.real
        fig.data[1].z = helix.imag
        fig.data[2].x = [float(t[i])]
        fig.data[2].y = [float(helix.real[i])]
        fig.data[2].z = [float(helix.imag[i])]
        fig.layout.title.text = f"(b) e^(-j·{_omega_pi_label(wound.omega)}·t)"
        vals, labels = _pi_tickvals_and_labels(float(t[0]), float(t[-1]),
                                               wound.omega)
        if vals:
            fig.layout.scene.xaxis.tickvals = vals
            fig.layout.scene.xaxis.ticktext = labels


# ─────────────────────────────────────────────────────────────────────────────
# (c) WoundSignalPlot
# ─────────────────────────────────────────────────────────────────────────────

class WoundSignalPlot(PlotPanel):
    """x(t)·e^(-jωt) 3D 변조 헬릭스 + envelope 회전체.

    envelope은 *원래 신호 x(t)를 t축에 대해 회전시킨 회전체*. 반지름은 |x(t)|.
    zero crossing에서 회전체가 t축에 정확히 닿는 것이 의도된 시각화.
    """

    # envelope 회전체의 둘레 분할 수. 48이면 위에서 봤을 때 거의 원.
    ENVELOPE_THETA_N = 48
    # envelope의 t축 방향 다운샘플 한도. 시간이 짧으면 다 쓰고, 길면 다운샘플.
    ENVELOPE_T_MAX = 500

    def __init__(self, *, show_envelope: bool = True) -> None:
        super().__init__()
        self.show_envelope = show_envelope

    def _build_figure(self) -> go.Figure:
        fig = _new_3d_figure(title="(c) x(t)·e^(-jωt)", height=400)
        # trace 0: 중심축 (t, 0, 0)
        fig.add_trace(_make_central_axis_trace())
        # trace 1: wound 본체
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(color=_GREEN, width=4),
                                   name="wound"))
        # trace 2 (옵션): envelope 회전체 — |x(t)|를 t축에 대해 회전시킨 surface.
        # show_envelope=False면 trace 자체를 만들지 않는다 (HTML 출력 시 빈 surface
        # placeholder가 일부 브라우저/plotly 조합에서 렌더링 막힘을 일으키므로).
        if self.show_envelope:
            fig.add_trace(go.Surface(
                x=[[0, 0], [0, 0]], y=[[0, 0], [0, 0]], z=[[0, 0], [0, 0]],
                colorscale=[[0, _ORANGE], [1, _ORANGE]],
                showscale=False,
                opacity=0.15,
                hoverinfo="skip",
                name="envelope",
                lighting=dict(ambient=0.7, diffuse=0.4, specular=0.05),
            ))
        # cursor: show_envelope에 따라 trace index가 2 또는 3
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                   marker=dict(color=_WHITE, size=5),
                                   name="cursor"))
        return fig

    def _envelope_mesh(self, t: np.ndarray, amp: np.ndarray):
        """|x(t)|를 t축에 대해 회전시킨 surface 메시 (X, Y, Z) 반환.

        X[i, j] = t[i_ds]            (t 방향)
        Y[i, j] = amp[i_ds] * cos(θ_j)   (Re 방향 반지름 성분)
        Z[i, j] = amp[i_ds] * sin(θ_j)   (Im 방향 반지름 성분)
        """
        # 다운샘플 (surface가 너무 무거워지지 않게)
        n = len(t)
        if n > self.ENVELOPE_T_MAX:
            stride = max(1, n // self.ENVELOPE_T_MAX)
            t_ds = t[::stride]
            amp_ds = amp[::stride]
        else:
            t_ds = t
            amp_ds = amp

        theta = np.linspace(0, 2 * np.pi, self.ENVELOPE_THETA_N + 1)
        T_grid, Th_grid = np.meshgrid(t_ds, theta, indexing="ij")
        A_grid, _ = np.meshgrid(amp_ds, theta, indexing="ij")
        X = T_grid
        Y = A_grid * np.cos(Th_grid)
        Z = A_grid * np.sin(Th_grid)
        return X, Y, Z

    def update(self, wound: WoundSignal, t_index: int) -> None:
        fig = self.figure
        t = wound.t
        w = wound.wound_values
        i = int(np.clip(t_index, 0, len(t) - 1))
        _set_central_axis_data(fig.data[0], t)
        fig.data[1].x = t
        fig.data[1].y = w.real
        fig.data[1].z = w.imag
        # trace index: show_envelope에 따라
        #   envelope=True  → [0:axis, 1:wound, 2:surface, 3:cursor]
        #   envelope=False → [0:axis, 1:wound, 2:cursor]
        if self.show_envelope:
            # envelope = 원래 신호를 t축에 대해 회전시킨 회전체. 반지름은 |x(t)|.
            # zero crossing에서 회전체가 t축에 정확히 닿는 게 의도된 그림.
            amp = np.abs(wound.signal_values)
            X, Y, Z = self._envelope_mesh(t, amp)
            fig.data[2].x = X
            fig.data[2].y = Y
            fig.data[2].z = Z
            cursor_idx = 3
        else:
            cursor_idx = 2
        fig.data[cursor_idx].x = [float(t[i])]
        fig.data[cursor_idx].y = [float(w.real[i])]
        fig.data[cursor_idx].z = [float(w.imag[i])]
        fig.layout.title.text = f"(c) x(t)·e^(-j·{_omega_pi_label(wound.omega)}·t)"
        vals, labels = _pi_tickvals_and_labels(float(t[0]), float(t[-1]),
                                               wound.omega)
        if vals:
            fig.layout.scene.xaxis.tickvals = vals
            fig.layout.scene.xaxis.ticktext = labels


# ─────────────────────────────────────────────────────────────────────────────
# (d) ForwardIntegralPlot — *Mode A의 메인 디시*
# ─────────────────────────────────────────────────────────────────────────────

class ForwardIntegralPlot(PlotPanel):
    """
    Wound signal의 누적 적분 ∫₀ᵗ x(τ)·e^(-jωτ) dτ를 3D 누적 벡터로 본다.

    구성:
      메인 3D 씬 (subplot 'scene'):
        - 시간 축 위에 누적 벡터의 끝점이 그리는 trail (시간 진행에 따라 길어짐)
        - 현재 시점 t_index에서 (t, 0, 0) 평면에 화살표 한 개 (원점 → running_integral[i])
      옵션 2D inset (subplot 'xy'):
        - 복소평면(Re-Im)에 trail의 위에서 본 모습
    """

    def __init__(self, *, show_complex_plane_inset: bool = True,
                 trail_alpha: float = 0.4) -> None:
        super().__init__()
        self.show_inset = show_complex_plane_inset
        self.trail_alpha = trail_alpha

    def _build_figure(self) -> go.Figure:
        fig = go.Figure()
        # subplot 레이아웃: 좌측 큰 3D, 우측 작은 2D inset
        if self.show_inset:
            fig.update_layout(
                **_DEFAULT_LAYOUT,
                title=dict(text="(d) ∫ x(τ)·e^(-jωτ) dτ — running vector",
                           x=0.02, xanchor="left", font=dict(color=_FG)),
                height=520,
                scene=dict(
                    domain=dict(x=[0.0, 0.7], y=[0.0, 1.0]),
                    xaxis=dict(_3D_AXIS, title="t"),
                    yaxis=dict(_3D_AXIS, title="Re"),
                    zaxis=dict(_3D_AXIS, title="Im"),
                    aspectmode="manual",
                    aspectratio=dict(x=2.0, y=1.0, z=1.0),
                    camera=dict(eye=dict(x=2.2, y=1.5, z=0.9)),
                ),
                xaxis=dict(_2D_AXIS, title="Re", domain=[0.74, 1.0],
                           anchor="y", scaleanchor="y", scaleratio=1),
                yaxis=dict(_2D_AXIS, title="Im", domain=[0.0, 1.0],
                           anchor="x"),
            )
        else:
            fig.update_layout(
                **_DEFAULT_LAYOUT,
                title=dict(text="(d) ∫ x(τ)·e^(-jωτ) dτ — running vector",
                           x=0.02, xanchor="left", font=dict(color=_FG)),
                height=480,
                scene=dict(
                    xaxis=dict(_3D_AXIS, title="t"),
                    yaxis=dict(_3D_AXIS, title="Re"),
                    zaxis=dict(_3D_AXIS, title="Im"),
                    aspectmode="manual",
                    aspectratio=dict(x=2.0, y=1.0, z=1.0),
                    camera=dict(eye=dict(x=2.2, y=1.5, z=0.9)),
                ),
            )

        # trace 0: 중심축 (t, 0, 0) — 회전·맴돌이의 기준선
        fig.add_trace(_make_central_axis_trace())
        # trace 1: 3D trail (지나온 끝점들)
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines",
            line=dict(color=_GREEN, width=3),
            opacity=self.trail_alpha,
            name="trail",
        ))
        # trace 2: 3D 화살표 줄기 (origin → tip at current t)
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines+markers",
            line=dict(color=_ORANGE, width=6),
            marker=dict(color=_ORANGE, size=[3, 7]),
            name="vector",
        ))
        # trace 3 (옵션): 2D inset trail
        if self.show_inset:
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="lines",
                line=dict(color=_GREEN, width=2),
                opacity=self.trail_alpha,
                xaxis="x", yaxis="y",
                name="inset trail",
            ))
            # trace 4: 2D inset 현재 벡터
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="lines+markers",
                line=dict(color=_ORANGE, width=3),
                marker=dict(color=_ORANGE, size=[5, 9]),
                xaxis="x", yaxis="y",
                name="inset vector",
            ))
        return fig

    def update(self, wound: WoundSignal, t_index: int) -> None:
        fig = self.figure
        t = wound.t
        r = wound.running_integral
        i = int(np.clip(t_index, 0, len(t) - 1))

        # trail: 0..i까지의 누적 끝점들. t를 x축으로 (3D는 시간 축 사용).
        trail_t = t[: i + 1]
        trail_re = r.real[: i + 1]
        trail_im = r.imag[: i + 1]

        # 현재 화살표: 시간 t[i]에서 (Re=0, Im=0)에서 (Re=r[i].real, Im=r[i].imag)로.
        current_t = float(t[i])
        tip_re = float(r.real[i])
        tip_im = float(r.imag[i])

        _set_central_axis_data(fig.data[0], t)
        fig.data[1].x = trail_t
        fig.data[1].y = trail_re
        fig.data[1].z = trail_im
        fig.data[2].x = [current_t, current_t]
        fig.data[2].y = [0.0, tip_re]
        fig.data[2].z = [0.0, tip_im]
        if self.show_inset:
            fig.data[3].x = trail_re
            fig.data[3].y = trail_im
            fig.data[4].x = [0.0, tip_re]
            fig.data[4].y = [0.0, tip_im]

            # 인셋 축 범위를 trail의 실제 범위에 맞게 자동 조정.
            # 매칭 케이스(큰 직선)와 미스매치(작은 원)가 시각적으로 구분되게.
            # 전체 running_integral 범위 기준 (현재까지가 아니라 풀 trajectory).
            rmax = float(max(np.max(np.abs(r.real)), np.max(np.abs(r.imag)), 1e-3))
            pad = rmax * 0.15
            lim = rmax + pad
            fig.layout.xaxis.range = [-lim, lim]
            fig.layout.yaxis.range = [-lim, lim]

        fig.layout.title.text = (
            f"(d) ∫₀ᵗ x(τ)·e^(-j·{_omega_pi_label(wound.omega)}·τ) dτ "
            f"— |·| = {abs(complex(tip_re, tip_im)):.3g}"
        )
        vals, labels = _pi_tickvals_and_labels(float(t[0]), float(t[-1]),
                                               wound.omega)
        if vals:
            fig.layout.scene.xaxis.tickvals = vals
            fig.layout.scene.xaxis.ticktext = labels


# ─────────────────────────────────────────────────────────────────────────────
# (e) SpectrumPlot — Mode B 본격 구현
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumPlot(PlotPanel):
    """
    X(jω)를 ω축 + 복소평면 3D로 시각화. Mode A의 wound signal과 좌표계 일치.

    좌표:
      x = ω,   y = Re X(jω),   z = Im X(jω)

    Trail이 ω축을 따라 휘감기는 모양으로 X(jω)의 *전체 모양*이 한 눈에 보임.
    오른쪽에 옵션 2D 인셋: Re-Im (기본) 또는 magnitude/phase 두 줄.

    update()는 Mode A 인터페이스를 위해 no-op. ``show_spectrum(omegas, X)``로
    데이터를 직접 박는다.
    """

    def __init__(self, *,
                 view: Literal["re_im", "mag_phase"] = "re_im",
                 show_inset: bool = True) -> None:
        super().__init__()
        self.view = view
        self.show_inset = show_inset

    def _build_figure(self) -> go.Figure:
        fig = go.Figure()
        if self.show_inset:
            fig.update_layout(
                **_DEFAULT_LAYOUT,
                title=dict(text="(e) X(jω) — spectrum",
                           x=0.02, xanchor="left", font=dict(color=_FG)),
                height=520,
                scene=dict(
                    domain=dict(x=[0.0, 0.7], y=[0.0, 1.0]),
                    xaxis=dict(_3D_AXIS, title="ω"),
                    yaxis=dict(_3D_AXIS, title="Re X"),
                    zaxis=dict(_3D_AXIS, title="Im X"),
                    aspectmode="manual",
                    aspectratio=dict(x=2.0, y=1.0, z=1.0),
                    camera=dict(eye=dict(x=2.2, y=1.5, z=0.9)),
                ),
                xaxis=dict(_2D_AXIS, title="ω", domain=[0.74, 1.0], anchor="y"),
                yaxis=dict(_2D_AXIS, title="", domain=[0.0, 1.0], anchor="x"),
            )
        else:
            fig.update_layout(
                **_DEFAULT_LAYOUT,
                title=dict(text="(e) X(jω) — spectrum",
                           x=0.02, xanchor="left", font=dict(color=_FG)),
                height=480,
                scene=dict(
                    xaxis=dict(_3D_AXIS, title="ω"),
                    yaxis=dict(_3D_AXIS, title="Re X"),
                    zaxis=dict(_3D_AXIS, title="Im X"),
                    aspectmode="manual",
                    aspectratio=dict(x=2.0, y=1.0, z=1.0),
                    camera=dict(eye=dict(x=2.2, y=1.5, z=0.9)),
                ),
            )

        # trace 0: 중심축 (ω, 0, 0)
        fig.add_trace(_make_central_axis_trace())
        # trace 1: 3D X(jω) trail
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines",
            line=dict(color=_GREEN, width=4),
            name="X(jω)",
        ))
        # trace 2: ω=0 마커 (DC)
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="markers",
            marker=dict(color=_WHITE, size=5),
            name="ω=0",
        ))
        # 인셋: view에 따라 다른 trace 셋
        if self.show_inset:
            if self.view == "re_im":
                # 두 곡선: Re X(ω), Im X(ω)
                fig.add_trace(go.Scatter(
                    x=[], y=[], mode="lines", line=dict(color=_GREEN, width=2),
                    xaxis="x", yaxis="y", name="Re X",
                ))
                fig.add_trace(go.Scatter(
                    x=[], y=[], mode="lines", line=dict(color=_PURPLE, width=2,
                                                         dash="dot"),
                    xaxis="x", yaxis="y", name="Im X",
                ))
            else:  # mag_phase
                fig.add_trace(go.Scatter(
                    x=[], y=[], mode="lines", line=dict(color=_GREEN, width=2),
                    xaxis="x", yaxis="y", name="|X|",
                ))
                fig.add_trace(go.Scatter(
                    x=[], y=[], mode="lines", line=dict(color=_PURPLE, width=2,
                                                         dash="dot"),
                    xaxis="x", yaxis="y", name="arg X",
                ))
        return fig

    def show_spectrum(self, omegas: np.ndarray, X: np.ndarray,
                      *, flatten_noise: bool = True) -> None:
        """spectrum 데이터를 받아 3D + 인셋을 갱신.

        Parameters
        ----------
        omegas : NDArray[float64], shape (n,)
        X : ComplexArray, shape (n,)
        flatten_noise : bool, default True
            Re X 또는 Im X 한 쪽이 ``max|X|`` 대비 1e-9 이하일 때, 즉
            *수학적으로 0이어야 하는데 부동소수점 노이즈만 남은* 경우, 그 축의
            범위를 살아 있는 쪽과 동일하게 강제해 데이터가 평면(Im=0 또는 Re=0)에
            누워 보이도록 한다. 짝함수의 X(jω)가 순수 실수, 홀함수는 순수 허수
            라는 메시지가 시각으로 직접 전달되도록 하기 위함.

            False면 Plotly의 자동 축 범위가 그대로 사용되고, ε ~ 1e-16 노이즈가
            축 전체를 차지해 평면 메시지가 묻혀 보인다. 디버깅이나 *진짜로* 작은
            허수성을 보고 싶을 때만 끈다.
        """
        fig = self.figure
        # 중심축
        _set_central_axis_data(fig.data[0], omegas)
        # 3D trail
        fig.data[1].x = omegas
        fig.data[1].y = X.real
        fig.data[1].z = X.imag
        # DC 마커 (ω=0에 가장 가까운 점)
        idx0 = int(np.argmin(np.abs(omegas)))
        fig.data[2].x = [float(omegas[idx0])]
        fig.data[2].y = [float(X.real[idx0])]
        fig.data[2].z = [float(X.imag[idx0])]
        # 인셋
        if self.show_inset:
            if self.view == "re_im":
                fig.data[3].x = omegas
                fig.data[3].y = X.real
                fig.data[4].x = omegas
                fig.data[4].y = X.imag
            else:
                fig.data[3].x = omegas
                fig.data[3].y = np.abs(X)
                fig.data[4].x = omegas
                fig.data[4].y = np.angle(X)
        fig.layout.title.text = (
            f"(e) X(jω) — ω ∈ [{omegas[0]:.3g}, {omegas[-1]:.3g}], "
            f"max|X|={float(np.max(np.abs(X))):.3g}"
        )

        # ── Re/Im 축 범위 결정 ───────────────────────────────────────────────
        # 한 축이 max|X| 대비 NOISE_THRESHOLD 이하인 경우, 그 축은 "수학적으로
        # 0인데 노이즈만 남은 축". 두 축을 같은 범위로 묶어 평면처럼 누이는 게
        # 짝/홀 함수 메시지를 보존한다.
        if flatten_noise:
            xmag = float(np.max(np.abs(X)))
            re_amp = float(np.max(np.abs(X.real)))
            im_amp = float(np.max(np.abs(X.imag)))
            NOISE_THRESHOLD = 1e-9  # 가우시안 등에서 1e-15 ~ 1e-12 노이즈를 잡는 안전 임계
            re_is_noise = re_amp < xmag * NOISE_THRESHOLD
            im_is_noise = im_amp < xmag * NOISE_THRESHOLD
            if re_is_noise or im_is_noise:
                # 살아 있는 쪽 + 작은 fallback (전부 0인 극단 케이스 대비)
                live = max(re_amp, im_amp, xmag * 1e-3)
                lim = live * 1.15  # 15% 패딩
                fig.layout.scene.yaxis.range = [-lim, lim]
                fig.layout.scene.zaxis.range = [-lim, lim]
                # 인셋의 y축도 같은 범위로 (re_im 뷰일 때만)
                if self.show_inset and self.view == "re_im":
                    fig.layout.yaxis.range = [-lim, lim]

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A의 PlotPanel 인터페이스 — Spectrum은 ω를 훑은 결과라 Mode A에서는
        no-op. show_spectrum()을 직접 호출."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# (f) InverseAccumulationPlot — Mode B 본격 구현
# ─────────────────────────────────────────────────────────────────────────────

class InverseAccumulationPlot(PlotPanel):
    """
    역변환 누적 시각화. 2D 복소평면.

    각 ω의 기여 (1/(2π))·X(jω)·e^(jωt_fix)·dω를 화살표로 *머리 잇기*. 누적 끝점이
    x(t_fix)에 수렴하는 모습.

    update()는 Mode A 인터페이스용 no-op. ``show_accumulation(...)``으로 데이터
    직접 박음.

    Parameters
    ----------
    show_target : bool, default True
        목표값 x(t_fix)을 별도 마커로 표시.
    arrow_alpha : float, default 0.5
        개별 화살표 투명도 (촘촘하게 그어지므로).
    """

    def __init__(self, *, show_target: bool = True,
                 arrow_alpha: float = 0.5) -> None:
        super().__init__()
        self.show_target = show_target
        self.arrow_alpha = arrow_alpha

    def _build_figure(self) -> go.Figure:
        fig = _new_2d_figure(title="(f) Σ X(jω)·e^(jωt_fix)·dω/(2π)", height=480)
        # 같은 스케일 유지 (실수성 cancel을 보려면 정사각이어야)
        fig.update_layout(
            xaxis=dict(_2D_AXIS, title="Re", scaleanchor="y", scaleratio=1),
            yaxis=dict(_2D_AXIS, title="Im"),
        )
        # trace 0: 화살표 줄기들 (단일 trace로 None separator 사용)
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="lines",
            line=dict(color=_ORANGE, width=1.5),
            opacity=self.arrow_alpha,
            name="arrows",
        ))
        # trace 1: 누적 끝점 trail
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="lines",
            line=dict(color=_GREEN, width=2.5),
            name="trail",
        ))
        # trace 2: 현재 누적 끝점
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="markers",
            marker=dict(color=_WHITE, size=8, symbol="circle"),
            name="current end",
        ))
        # trace 3 (옵션): 목표값 x(t_fix)
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="markers",
            marker=dict(color=_BLUE, size=12, symbol="x"),
            name="target x(t_fix)",
            visible=self.show_target,
        ))
        return fig

    def show_accumulation(self,
                          omegas: np.ndarray,
                          arrows: np.ndarray,
                          accumulated: np.ndarray,
                          target: complex,
                          progress_index: Optional[int] = None) -> None:
        """역변환 누적 데이터를 받아 그림.

        progress_index=k면 처음 k+1개 화살표만 그리고 누적도 거기까지. None이면
        전부.
        """
        fig = self.figure
        n = arrows.size
        if progress_index is None:
            k = n - 1
        else:
            k = int(np.clip(progress_index, 0, n - 1))

        # 화살표 줄기들. 각 화살표 i는 origin = accumulated[i] - arrows[i],
        # tip = accumulated[i]. None separator로 한 trace에 모음.
        starts = accumulated[: k + 1] - arrows[: k + 1]
        ends = accumulated[: k + 1]
        xs = np.empty(3 * (k + 1))
        ys = np.empty(3 * (k + 1))
        xs[0::3] = starts.real
        xs[1::3] = ends.real
        xs[2::3] = np.nan
        ys[0::3] = starts.imag
        ys[1::3] = ends.imag
        ys[2::3] = np.nan
        fig.data[0].x = xs
        fig.data[0].y = ys

        # trail: 누적 끝점들 연결
        fig.data[1].x = accumulated.real[: k + 1]
        fig.data[1].y = accumulated.imag[: k + 1]

        # 현재 끝점
        fig.data[2].x = [float(accumulated.real[k])]
        fig.data[2].y = [float(accumulated.imag[k])]

        # 목표값
        if self.show_target:
            fig.data[3].x = [float(np.real(target))]
            fig.data[3].y = [float(np.imag(target))]

        # 축 자동 조정 (목표값 + 누적 trail 모두 보이도록)
        all_re = np.concatenate([accumulated.real, [np.real(target)]])
        all_im = np.concatenate([accumulated.imag, [np.imag(target)]])
        rmax = float(max(np.max(np.abs(all_re)), np.max(np.abs(all_im)), 1e-3))
        pad = rmax * 0.2
        lim = rmax + pad
        fig.layout.xaxis.range = [-lim, lim]
        fig.layout.yaxis.range = [-lim, lim]

        cur = complex(accumulated[k])
        fig.layout.title.text = (
            f"(f) inverse accum — ω={float(omegas[k]):.3g}  "
            f"current=({cur.real:+.3g},{cur.imag:+.3g})  "
            f"target=({np.real(target):+.3g},{np.imag(target):+.3g})"
        )

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PartitionedAccumulationPlot — 단위벡터 + 누적 호 시각화
# ─────────────────────────────────────────────────────────────────────────────
#
# 한 신호 구간의 적분을 *단위원 위 단위벡터들의 합 × dt* 로 분해해 보여주는
# 2D 패널. rect→sinc 같은 적분의 기하적 유도에 쓰인다.
#
#   unit_vectors[k] = e^(-jωt_k)   (단위원 위 점)
#   arrows[k]       = unit_vectors[k] * dt   (실제 누적 항)
#   cum[k]          = Σ arrows[0..k-1]       (Re축 위 점에 도달하는 trail)
#
# boundary 인자로 "한 주기에 cancel되는 묶음(dim) + 잔여 호(bright)" 분기 가능.
# Phase 1처럼 분기 없는 경우 boundary=0.

class PartitionedAccumulationPlot(PlotPanel):
    """
    단위원 위 단위벡터 화살표 + 누적 호를 한 복소평면에 그린다.

    Examples
    --------
    >>> import numpy as np
    >>> from ftvis import PartitionedAccumulationPlot
    >>> N, omega, W = 24, np.pi/12, 2.0
    >>> dt = 2*W / N
    >>> t = -W + np.arange(N) * dt + dt/2
    >>> uv = np.exp(-1j * omega * t)
    >>> p = PartitionedAccumulationPlot()
    >>> p.show_partition(uv, dt, label='θ = π/6')
    >>> p.figure  # doctest: +SKIP

    Parameters
    ----------
    show_unit_circle : bool, default True
        점선 단위원 표시.
    show_chord : bool, default True
        잔여 호의 양 끝(단위원 위)을 잇는 chord 표시. 길이 = 2sin(잔여각/2).
    arrow_width : float, default 1.0
        단위벡터 화살표 선 굵기.
    """

    # 색상은 _GREEN (잔여/메인), _DIM (한 주기 묶음), _BLUE (chord), _ORANGE (끝점) 사용

    def __init__(
        self,
        *,
        show_unit_circle: bool = True,
        show_chord: bool = True,
        arrow_width: float = 1.0,
    ) -> None:
        super().__init__()
        self.show_unit_circle = show_unit_circle
        self.show_chord = show_chord
        self.arrow_width = arrow_width

    def _build_figure(self) -> go.Figure:
        # 가로로 길어 누적 호 끝점이 들어오게. 데이터에 맞춰 show_partition에서 갱신.
        fig = _new_2d_figure(title="(partitioned accumulation)", height=320)
        # 정사각 픽셀 비율 (Re/Im 같은 스케일)
        fig.update_layout(
            xaxis=dict(_2D_AXIS, title="Re", scaleanchor="y", scaleratio=1),
            yaxis=dict(_2D_AXIS, title="Im"),
        )
        return fig

    def show_partition(
        self,
        unit_vectors: np.ndarray,
        dt: float,
        *,
        boundary: int = 0,
        label: Optional[str] = None,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        unit_vectors : ComplexArray, shape (N,)
            단위원 위 단위벡터들. ``e^(-jωt_k)`` 형태.
        dt : float
            적분 가중치. ``arrows = unit_vectors * dt``, 누적합이 적분값.
        boundary : int, default 0
            dim(앞)/bright(뒤) 경계 인덱스. 한 주기당 단위벡터 수. 0이면 전부 bright.
        label : str, optional
            패널 제목. None이면 적분값 자동 표기.
        x_range, y_range : (float, float), optional
            축 범위 강제. None이면 데이터에서 자동.
        """
        fig = self.figure
        fig.data = ()  # 데이터 비우고 다시 그림 (여러 번 show_partition 호출 시)

        unit_vectors = np.asarray(unit_vectors, dtype=np.complex128)
        N = len(unit_vectors)
        boundary = int(np.clip(boundary, 0, N))
        arrows = unit_vectors * float(dt)
        cum = np.concatenate([[0+0j], np.cumsum(arrows)])

        # 단위원
        if self.show_unit_circle:
            th = np.linspace(0, 2*np.pi, 200)
            fig.add_trace(go.Scatter(
                x=np.cos(th), y=np.sin(th), mode="lines",
                line=dict(color=_AXIS, width=1, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))

        # 한 주기 묶음 단위벡터 화살표 (회색 dim)
        if boundary > 0:
            uv_dim = unit_vectors[:boundary]
            self._add_radial_arrows(fig, uv_dim, color=_GRID, opacity=0.6,
                                     width=self.arrow_width, marker_size=3)

        # 잔여 단위벡터 화살표 (강조 초록)
        uv_res = unit_vectors[boundary:]
        if len(uv_res) > 0:
            self._add_radial_arrows(fig, uv_res, color=_GREEN, opacity=0.9,
                                     width=self.arrow_width * 1.2,
                                     marker_size=4)

        # chord (잔여 호의 양 끝점)
        if self.show_chord and len(uv_res) > 0:
            p0, p1 = complex(uv_res[0]), complex(uv_res[-1])
            fig.add_trace(go.Scatter(
                x=[p0.real, p1.real], y=[p0.imag, p1.imag],
                mode="lines+markers",
                line=dict(color=_BLUE, width=3),
                marker=dict(color=_BLUE, size=8),
                hoverinfo="skip", showlegend=False,
            ))

        # dim 누적 호 (한 주기 묶음)
        if boundary > 0:
            fig.add_trace(go.Scatter(
                x=cum.real[:boundary+1], y=cum.imag[:boundary+1],
                mode="lines",
                line=dict(color=_GRID, width=2.5),
                hoverinfo="skip", showlegend=False,
            ))

        # bright 누적 호 (잔여)
        fig.add_trace(go.Scatter(
            x=cum.real[boundary:], y=cum.imag[boundary:],
            mode="lines",
            line=dict(color=_GREEN, width=3.5),
            hoverinfo="skip", showlegend=False,
        ))

        # 끝점 = 적분값
        fig.add_trace(go.Scatter(
            x=[float(cum.real[-1])], y=[float(cum.imag[-1])],
            mode="markers",
            marker=dict(color=_ORANGE, size=10, symbol="circle"),
            hoverinfo="skip", showlegend=False,
        ))

        # 제목
        integ = complex(cum[-1])
        if label is None:
            label = f"∫ = {integ.real:+.4f}{integ.imag:+.4f}j"
        fig.layout.title.text = label

        # 축 범위
        if x_range is not None:
            fig.layout.xaxis.range = list(x_range)
        if y_range is not None:
            fig.layout.yaxis.range = list(y_range)

    @staticmethod
    def _add_radial_arrows(
        fig: go.Figure,
        unit_vectors: np.ndarray,
        *,
        color: str,
        opacity: float,
        width: float,
        marker_size: int,
    ) -> None:
        """원점에서 unit_vectors 끝점까지 가는 N개 화살표를 한 trace로 추가."""
        n = len(unit_vectors)
        xs = np.empty(3*n)
        ys = np.empty(3*n)
        xs[0::3] = 0
        xs[1::3] = unit_vectors.real
        xs[2::3] = np.nan
        ys[0::3] = 0
        ys[1::3] = unit_vectors.imag
        ys[2::3] = np.nan
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            hoverinfo="skip", showlegend=False,
        ))
        # 끝점 마커
        fig.add_trace(go.Scatter(
            x=unit_vectors.real, y=unit_vectors.imag,
            mode="markers",
            marker=dict(color=color, size=marker_size),
            opacity=opacity,
            hoverinfo="skip", showlegend=False,
        ))

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PartialIntegralComparisonPlot — 여러 partial 적분 trail을 한 3D scene에 비교
# ─────────────────────────────────────────────────────────────────────────────
#
# rect→sinc 노트북 5.2의 Trail A/B/C 비교를 일반화. 각 trail은
# `(1/2π) ∫_{-Ω}^{ω} kernel(ω')·e^(jω't) dω'` 누적.
# 사용자가 [(kernel, t, label, color, ...), ...] 형태로 trail들을 명세.

class PartialIntegralComparisonPlot(PlotPanel):
    """
    여러 partial 적분 trail을 한 3D scene에 함께 그려 비교.

    좌표: x = ω, (y, z) = (Re partial, Im partial). ω가 진행하면서 trail이
    어떻게 자라는지 본다. Time shift property 등 적분 동등성 시각 증명에 적합.

    Examples
    --------
    >>> import numpy as np
    >>> from ftvis import signals, FourierAnalyzer, PartialIntegralComparisonPlot
    >>> sig = signals.rect(width=4)
    >>> an = FourierAnalyzer(sig, t_min=-3, t_max=3)
    >>> omegas, X = an.spectrum(omega_min=-30, omega_max=30, n_omega=601)
    >>> Xs = X * np.exp(-1j * omegas * 1.5)   # X·e^(-jωα)
    >>> p = PartialIntegralComparisonPlot()
    >>> p.show_trails(omegas, [
    ...     {'kernel': X,  't': 0.0, 'label': 'A: X, t=0',         'color': '#7ec699'},
    ...     {'kernel': Xs, 't': 0.0, 'label': 'B: X·phase, t=0',   'color': '#ffa657'},
    ...     {'kernel': Xs, 't': 1.5, 'label': 'C: X·phase, t=α',   'color': '#f0f6fc', 'dash': 'dash'},
    ... ])
    >>> p.figure  # doctest: +SKIP
    """

    def __init__(self) -> None:
        super().__init__()

    def _build_figure(self) -> go.Figure:
        fig = _new_3d_figure(title="(partial integral comparison)", height=560)
        fig.update_layout(
            scene=dict(
                xaxis=dict(_3D_AXIS, title="ω"),
                yaxis=dict(_3D_AXIS, title="Re partial"),
                zaxis=dict(_3D_AXIS, title="Im partial"),
                aspectmode="manual",
                aspectratio=dict(x=2.5, y=1.0, z=1.0),
                camera=dict(eye=dict(x=2.0, y=1.6, z=1.0)),
            ),
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(13,17,23,0.7)",
                        bordercolor=_GRID, borderwidth=1),
        )
        return fig

    def show_trails(
        self,
        omegas: np.ndarray,
        trails: list[dict],
        *,
        label: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        omegas : NDArray[float64], shape (n,)
            균등 ω 분할.
        trails : list of dict
            각 dict는 다음 키:

            - ``'kernel'`` (ComplexArray, shape (n,)) — 적분 핵 X(jω) 또는
              X(jω)·e^(-jωα) 같은 변형.
            - ``'t'`` (float) — 시간 ``e^(jωt)`` 곱하기.
            - ``'label'`` (str) — legend 라벨.
            - ``'color'`` (str) — line/marker 색.
            - ``'width'`` (float, optional, default 4.0) — 선 굵기.
            - ``'dash'`` (str, optional) — plotly dash style ('dash', 'dot', etc).
            - ``'marker_symbol'`` (str, optional) — 끝점 마커 심볼.
        label : str, optional
            전체 figure 제목.
        """
        fig = self.figure
        fig.data = ()

        omegas = np.asarray(omegas, dtype=np.float64)
        domega = float(omegas[1] - omegas[0])

        # 중심축 (ω, 0, 0)
        fig.add_trace(go.Scatter3d(
            x=[float(omegas[0]), float(omegas[-1])], y=[0, 0], z=[0, 0],
            mode="lines",
            line=dict(color=_FG, width=5),
            hoverinfo="skip", showlegend=False,
        ))

        # 각 trail
        all_re: list[float] = []
        all_im: list[float] = []
        for spec in trails:
            kernel = np.asarray(spec["kernel"], dtype=np.complex128)
            t = float(spec["t"])
            color = str(spec["color"])
            label_t = str(spec["label"])
            width = float(spec.get("width", 4.0))
            dash = spec.get("dash")
            marker_symbol = spec.get("marker_symbol", "circle")

            arrows = kernel * np.exp(1j * omegas * t) * domega / (2 * np.pi)
            cum = np.cumsum(arrows)

            line_kw = dict(color=color, width=width)
            if dash is not None:
                line_kw["dash"] = dash

            # 본체 trail
            fig.add_trace(go.Scatter3d(
                x=omegas, y=cum.real, z=cum.imag,
                mode="lines",
                line=line_kw,
                name=label_t,
                hoverinfo="skip",
            ))
            # 끝점 마커
            fig.add_trace(go.Scatter3d(
                x=[float(omegas[-1])],
                y=[float(cum[-1].real)],
                z=[float(cum[-1].imag)],
                mode="markers",
                marker=dict(color=color, size=7, symbol=marker_symbol),
                showlegend=False, hoverinfo="skip",
            ))

            all_re.extend(cum.real.tolist())
            all_im.extend(cum.imag.tolist())

        # Re/Im 축 범위 정사각
        if all_re:
            re_max = max(abs(v) for v in all_re)
            im_max = max(abs(v) for v in all_im)
            lim = max(re_max, im_max, 1.0) * 1.2
            fig.layout.scene.yaxis.range = [-lim, lim]
            fig.layout.scene.zaxis.range = [-lim, lim]

        if label is not None:
            fig.layout.title.text = label

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# WoundSpectrumPlot — X(jω)·e^(-jωα) 3D helix + |X| 회전체 envelope
# ─────────────────────────────────────────────────────────────────────────────
#
# 시간 도메인의 WoundSignalPlot에 대응되는 주파수 도메인 버전. linear phase가
# 곱해진 spectrum이 어떻게 helix-like 모양이 되는지 직접 보여줌. envelope =
# |X(jω)|를 ω축에 회전시킨 회전체.

class WoundSpectrumPlot(PlotPanel):
    """
    ``X(jω)·e^(-jωα)``의 3D 시각화. ω축이 한 축, (Re, Im) 두 축.

    helix 본체 = `X(jω)·e^(-jωα)`. envelope 회전체 = `|X(jω)|`를 ω축에 회전
    (반지름 = |X|). zero crossing(|X|=0)에서 회전체가 ω축에 닿음.

    Examples
    --------
    >>> import numpy as np
    >>> from ftvis import signals, FourierAnalyzer, WoundSpectrumPlot
    >>> sig = signals.rect(width=4)
    >>> an = FourierAnalyzer(sig, t_min=-3, t_max=3, n_samples=4000)
    >>> omegas, X = an.spectrum(omega_min=-6*np.pi, omega_max=6*np.pi)
    >>> p = WoundSpectrumPlot()
    >>> p.show_wound_spectrum(omegas, X, alpha=1.5)
    >>> p.figure  # doctest: +SKIP

    Parameters
    ----------
    show_envelope : bool, default True
        ``|X(jω)|`` 회전체 envelope 표시.
    envelope_alpha : float, default 0.15
        envelope 투명도.
    """

    ENVELOPE_THETA_N = 36
    ENVELOPE_OMEGA_MAX = 200  # 다운샘플 한도

    def __init__(
        self,
        *,
        show_envelope: bool = True,
        envelope_alpha: float = 0.15,
    ) -> None:
        super().__init__()
        self.show_envelope = show_envelope
        self.envelope_alpha = envelope_alpha

    def _build_figure(self) -> go.Figure:
        fig = _new_3d_figure(title="(wound spectrum)", height=520)
        # _new_3d_figure는 x축이 't'로 되어 있음; ω 라벨로 갱신
        fig.update_layout(
            scene=dict(
                xaxis=dict(_3D_AXIS, title="ω"),
                yaxis=dict(_3D_AXIS, title="Re"),
                zaxis=dict(_3D_AXIS, title="Im"),
                aspectmode="manual",
                aspectratio=dict(x=2.5, y=1.0, z=1.0),
                camera=dict(eye=dict(x=2.0, y=1.6, z=1.0)),
            ),
        )
        return fig

    def show_wound_spectrum(
        self,
        omegas: np.ndarray,
        X: np.ndarray,
        alpha: float = 0.0,
        *,
        label: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        omegas : NDArray[float64], shape (n,)
        X : ComplexArray, shape (n,)
            원본 spectrum.
        alpha : float, default 0.0
            linear phase. ``X·e^(-jωα)``로 carrier 추가. 0이면 X 그대로.
        label : str, optional
            제목.
        """
        fig = self.figure
        fig.data = ()

        omegas = np.asarray(omegas, dtype=np.float64)
        X = np.asarray(X, dtype=np.complex128)
        wound = X * np.exp(-1j * omegas * float(alpha))

        # 중심축 (ω, 0, 0)
        fig.add_trace(go.Scatter3d(
            x=[float(omegas[0]), float(omegas[-1])], y=[0, 0], z=[0, 0],
            mode="lines",
            line=dict(color=_FG, width=5),
            hoverinfo="skip", showlegend=False,
        ))

        # Envelope 회전체 (|X|)
        if self.show_envelope:
            n = len(omegas)
            stride = max(1, n // self.ENVELOPE_OMEGA_MAX)
            om_ds = omegas[::stride]
            amp_ds = np.abs(X[::stride])
            theta = np.linspace(0, 2*np.pi, self.ENVELOPE_THETA_N + 1)
            T_grid, Th_grid = np.meshgrid(om_ds, theta, indexing="ij")
            A_grid, _ = np.meshgrid(amp_ds, theta, indexing="ij")
            fig.add_trace(go.Surface(
                x=T_grid,
                y=A_grid * np.cos(Th_grid),
                z=A_grid * np.sin(Th_grid),
                colorscale=[[0, _ORANGE], [1, _ORANGE]],
                showscale=False,
                opacity=self.envelope_alpha,
                hoverinfo="skip",
                lighting=dict(ambient=0.7, diffuse=0.4, specular=0.05),
            ))

        # Carrier helix
        fig.add_trace(go.Scatter3d(
            x=omegas, y=wound.real, z=wound.imag,
            mode="lines",
            line=dict(color=_GREEN, width=4),
            hoverinfo="skip", showlegend=False,
        ))

        # 제목
        if label is None:
            if alpha == 0.0:
                label = (
                    f"X(jω) — ω ∈ [{omegas[0]:.3g}, {omegas[-1]:.3g}], "
                    f"max|X|={float(np.max(np.abs(X))):.3g}"
                )
            else:
                label = (
                    f"X(jω)·e^(-jω·{alpha:g}) — envelope max={float(np.max(np.abs(X))):.3g}"
                )
        fig.layout.title.text = label

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 역변환용 — Inverse winding helix (역변환의 ω-domain winding 회전자)
# ─────────────────────────────────────────────────────────────────────────────

class InverseWindingHelixPlot(PlotPanel):
    """e^(+jωt_fix)의 단위 헬릭스 3D — 역변환 winding 회전자.

    순변환에서 `e^(-jωt)`가 *t축 위의 헬릭스*였듯이, 역변환에서는
    `e^(+jωt_fix)`가 *ω축 위의 헬릭스*다. 각속도가 `t_fix`인 단위 헬릭스.
    fixed_t의 값에 따라 감기는 속도가 결정되며, ω 따라 단위원을 도는 회전자.

    ``show_helix(omegas, t_fix)``로 데이터를 박는다. update()는 Mode A
    인터페이스용 no-op.
    """

    def __init__(self) -> None:
        super().__init__()

    def _build_figure(self) -> go.Figure:
        fig = _new_3d_figure(title="(b') e^(+jωt_fix)", height=400)
        fig.update_layout(
            scene=dict(
                xaxis=dict(_3D_AXIS, title="ω"),
                yaxis=dict(_3D_AXIS, title="Re"),
                zaxis=dict(_3D_AXIS, title="Im"),
                aspectmode="manual",
                aspectratio=dict(x=2.0, y=1.0, z=1.0),
                camera=dict(eye=dict(x=2.2, y=1.5, z=0.9)),
            ),
        )
        # trace 0: 중심축 (ω, 0, 0)
        fig.add_trace(_make_central_axis_trace())
        # trace 1: helix
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines",
            line=dict(color=_BLUE, width=4),
            name="helix",
        ))
        return fig

    def show_helix(self, omegas: np.ndarray, t_fix: float) -> None:
        """ω 범위와 고정 t_fix에서 단위 헬릭스 e^(+jωt_fix)를 박는다."""
        fig = self.figure
        omegas = np.asarray(omegas, dtype=float)
        helix = np.exp(1j * omegas * float(t_fix))
        _set_central_axis_data(fig.data[0], omegas)
        fig.data[1].x = omegas
        fig.data[1].y = helix.real
        fig.data[1].z = helix.imag
        fig.layout.title.text = f"(b') e^(+jω·{t_fix:g})"

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 역변환용 — 3D 누적 적분 trail (ω축 위의 running integral)
# ─────────────────────────────────────────────────────────────────────────────

class InverseIntegral3DPlot(PlotPanel):
    """역변환 누적 적분 (1/2π)·∫ X(jω)·e^(jωt_fix) dω를 3D 누적 trail로.

    순변환 ForwardIntegralPlot의 ω-domain 대응물.
      메인 3D 씬:
        - ω축 위에 누적 끝점이 그리는 trail (ω 진행에 따라 길어짐)
        - 현재 ω에서 (ω_now, 0, 0)에서 누적 끝점으로 향하는 화살표
      옵션 2D inset (복소평면): 위에서 본 trail

    ``show_accumulation(omegas, accumulated, ...)``로 데이터 박음.
    """

    def __init__(self, *, show_complex_plane_inset: bool = True,
                 trail_alpha: float = 0.4) -> None:
        super().__init__()
        self.show_inset = show_complex_plane_inset
        self.trail_alpha = trail_alpha

    def _build_figure(self) -> go.Figure:
        fig = go.Figure()
        if self.show_inset:
            fig.update_layout(
                **_DEFAULT_LAYOUT,
                title=dict(text="(d') ∫ X(jω)·e^(jωt_fix) dω/(2π) — running vector",
                           x=0.02, xanchor="left", font=dict(color=_FG)),
                height=520,
                scene=dict(
                    domain=dict(x=[0.0, 0.7], y=[0.0, 1.0]),
                    xaxis=dict(_3D_AXIS, title="ω"),
                    yaxis=dict(_3D_AXIS, title="Re"),
                    zaxis=dict(_3D_AXIS, title="Im"),
                    aspectmode="manual",
                    aspectratio=dict(x=2.0, y=1.0, z=1.0),
                    camera=dict(eye=dict(x=2.2, y=1.5, z=0.9)),
                ),
                xaxis=dict(_2D_AXIS, title="Re", domain=[0.74, 1.0],
                           anchor="y", scaleanchor="y", scaleratio=1),
                yaxis=dict(_2D_AXIS, title="Im", domain=[0.0, 1.0],
                           anchor="x"),
            )
        else:
            fig.update_layout(
                **_DEFAULT_LAYOUT,
                title=dict(text="(d') ∫ X(jω)·e^(jωt_fix) dω/(2π) — running vector",
                           x=0.02, xanchor="left", font=dict(color=_FG)),
                height=480,
                scene=dict(
                    xaxis=dict(_3D_AXIS, title="ω"),
                    yaxis=dict(_3D_AXIS, title="Re"),
                    zaxis=dict(_3D_AXIS, title="Im"),
                    aspectmode="manual",
                    aspectratio=dict(x=2.0, y=1.0, z=1.0),
                    camera=dict(eye=dict(x=2.2, y=1.5, z=0.9)),
                ),
            )

        # trace 0: 중심축 (ω, 0, 0)
        fig.add_trace(_make_central_axis_trace())
        # trace 1: 3D trail
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines",
            line=dict(color=_GREEN, width=3),
            opacity=self.trail_alpha,
            name="trail",
        ))
        # trace 2: 3D 현재 화살표 (origin → tip)
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines+markers",
            line=dict(color=_ORANGE, width=6),
            marker=dict(color=_ORANGE, size=[3, 7]),
            name="vector",
        ))
        # trace 3, 4 (옵션): 2D inset trail & 화살표
        if self.show_inset:
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="lines",
                line=dict(color=_GREEN, width=2),
                opacity=self.trail_alpha,
                xaxis="x", yaxis="y", name="inset trail",
            ))
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="lines+markers",
                line=dict(color=_ORANGE, width=3),
                marker=dict(color=_ORANGE, size=[5, 9]),
                xaxis="x", yaxis="y", name="inset vector",
            ))
        return fig

    def show_accumulation(self,
                          omegas: np.ndarray,
                          accumulated: np.ndarray,
                          *,
                          progress_index: Optional[int] = None,
                          flatten_noise: bool = True,
                          ) -> None:
        """누적 데이터를 받아 그림.

        Parameters
        ----------
        omegas : NDArray[float64], shape (n,)
            누적 *순서*에 따라 정렬된 ω 배열 (monotonic 권장 — ω축 시각화 일관성).
        accumulated : ComplexArray, shape (n,)
            머리 잇기 누적합. accumulated[-1] ≈ x(t_fix).
        progress_index : int, optional
            None이면 전부. 정수면 처음 k+1개만.
        flatten_noise : bool, default True
            trail의 Re/Im 한 쪽이 ``max|r|`` 대비 1e-9 이하일 때, 즉
            *수학적으로 0이어야 하는데 부동소수점 노이즈만 남은* 경우, 그 축의
            범위를 살아 있는 쪽과 동일하게 강제해 trail이 평면(Im=0 또는 Re=0)에
            누워 보이도록 한다. 실수 신호의 역변환 결과가 *실수 평면* 위로
            누워 보이게 하는 효과. 데이터 자체는 건드리지 않고 *축 범위만* 조정.

            False면 Plotly 자동 축 범위가 사용돼 ε ~ 1e-16 노이즈가 축 전체를
            차지해 평면 메시지가 묻힌다.
        """
        fig = self.figure
        omegas = np.asarray(omegas, dtype=float)
        r = np.asarray(accumulated, dtype=complex)
        n = r.size
        if progress_index is None:
            k = n - 1
        else:
            k = int(np.clip(progress_index, 0, n - 1))

        trail_w = omegas[: k + 1]
        trail_re = r.real[: k + 1]
        trail_im = r.imag[: k + 1]

        current_w = float(omegas[k])
        tip_re = float(r.real[k])
        tip_im = float(r.imag[k])

        _set_central_axis_data(fig.data[0], omegas)
        fig.data[1].x = trail_w
        fig.data[1].y = trail_re
        fig.data[1].z = trail_im
        fig.data[2].x = [current_w, current_w]
        fig.data[2].y = [0.0, tip_re]
        fig.data[2].z = [0.0, tip_im]
        if self.show_inset:
            fig.data[3].x = trail_re
            fig.data[3].y = trail_im
            fig.data[4].x = [0.0, tip_re]
            fig.data[4].y = [0.0, tip_im]

        # ── Re/Im 축 범위 결정 ───────────────────────────────────────────────
        # SpectrumPlot의 flatten_noise와 동일한 패턴. 한 축이 max|r| 대비
        # NOISE_THRESHOLD 이하면 "수학적으로 0인데 노이즈만 남은 축". 두 축을
        # 같은 범위로 묶어 trail이 평면처럼 누워 보이게 한다. 데이터 자체는
        # 그대로 두고 *축 범위만* 조정 — 진짜로 작은 허수성을 보고 싶으면
        # flatten_noise=False로 끌 수 있음.
        rmag = float(np.max(np.abs(r))) if r.size else 0.0
        re_amp = float(np.max(np.abs(r.real))) if r.size else 0.0
        im_amp = float(np.max(np.abs(r.imag))) if r.size else 0.0
        NOISE_THRESHOLD = 1e-9
        if flatten_noise and rmag > 0:
            re_is_noise = re_amp < rmag * NOISE_THRESHOLD
            im_is_noise = im_amp < rmag * NOISE_THRESHOLD
            if re_is_noise or im_is_noise:
                # 살아 있는 쪽 + 작은 fallback (전부 0인 극단 케이스 대비)
                live = max(re_amp, im_amp, rmag * 1e-3)
                lim = live * 1.15  # 15% 패딩
                fig.layout.scene.yaxis.range = [-lim, lim]
                fig.layout.scene.zaxis.range = [-lim, lim]
                if self.show_inset:
                    fig.layout.xaxis.range = [-lim, lim]
                    fig.layout.yaxis.range = [-lim, lim]
            else:
                # 두 축 모두 살아 있음 — 기존 동작: 한 limit으로 정사각
                if self.show_inset:
                    rmax = max(re_amp, im_amp, 1e-3)
                    pad = rmax * 0.15
                    lim = rmax + pad
                    fig.layout.xaxis.range = [-lim, lim]
                    fig.layout.yaxis.range = [-lim, lim]
        else:
            # flatten_noise=False 또는 r이 전부 0
            if self.show_inset:
                rmax = max(re_amp, im_amp, 1e-3)
                pad = rmax * 0.15
                lim = rmax + pad
                fig.layout.xaxis.range = [-lim, lim]
                fig.layout.yaxis.range = [-lim, lim]

        fig.layout.title.text = (
            f"(d') ∫ X(jω')·e^(jω't_fix) dω'/(2π) "
            f"— |·| = {abs(complex(tip_re, tip_im)):.3g}"
        )

    def update(self, wound: WoundSignal, t_index: int) -> None:
        """Mode A 인터페이스용 no-op."""
        pass
