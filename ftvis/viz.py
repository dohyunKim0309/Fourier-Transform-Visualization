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
        # trace 2: envelope 회전체 — |x(t)|를 t축에 대해 회전시킨 surface.
        # 빈 placeholder; update에서 실제 메시 채움.
        fig.add_trace(go.Surface(
            x=[[0, 0], [0, 0]], y=[[0, 0], [0, 0]], z=[[0, 0], [0, 0]],
            colorscale=[[0, _ORANGE], [1, _ORANGE]],
            showscale=False,
            opacity=0.15,
            hoverinfo="skip",
            name="envelope",
            visible=self.show_envelope,
            lighting=dict(ambient=0.7, diffuse=0.4, specular=0.05),
        ))
        # trace 3: cursor
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
        if self.show_envelope:
            # envelope = 원래 신호를 t축에 대해 회전시킨 회전체. 반지름은 |x(t)|.
            # zero crossing에서 회전체가 t축에 정확히 닿는 게 의도된 그림.
            amp = np.abs(wound.signal_values)
            X, Y, Z = self._envelope_mesh(t, amp)
            fig.data[2].x = X
            fig.data[2].y = Y
            fig.data[2].z = Z
            fig.data[2].visible = True
        else:
            fig.data[2].visible = False
        fig.data[3].x = [float(t[i])]
        fig.data[3].y = [float(w.real[i])]
        fig.data[3].z = [float(w.imag[i])]
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
