"""
Day 2 done definition 검증용 스크립트.

단일 톤(cos·sin) 중심으로 winding 원리를 보여주고, 마지막에 무한 적분 가능
신호(감쇠 정현파, causal·two-sided exp)를 둔다.

산출: examples/screenshots/<scenario>__<panel>.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ftvis import (
    FourierAnalyzer,
    ForwardIntegralPlot,
    InverseAccumulationPlot,
    SignalPlot,
    SpectrumPlot,
    WindingHelixPlot,
    WoundSignalPlot,
    signals,
)


def render_scenario(out_dir: Path, scenario: str, sig, omega_focus: float,
                    t_index_frac: float = 0.98,
                    t_min: float = 0.0,
                    t_max: float = 10.0,
                    n_samples: int = 2000) -> None:
    """한 시나리오에 대해 4개 HTML 생성."""
    out_dir.mkdir(parents=True, exist_ok=True)
    an = FourierAnalyzer(sig, t_min=t_min, t_max=t_max, n_samples=n_samples)
    ws = an.wound_at_omega(omega_focus)
    t_index = int(len(ws.t) * t_index_frac)

    panels = {
        "a_signal": SignalPlot(),
        "b_helix": WindingHelixPlot(),
        "c_wound": WoundSignalPlot(show_envelope=True),
        "d_forward_integral": ForwardIntegralPlot(show_complex_plane_inset=True),
    }
    for tag, panel in panels.items():
        panel.update(ws, t_index)
        out_path = out_dir / f"{scenario}__{tag}.html"
        panel.figure.write_html(str(out_path), include_plotlyjs="cdn",
                                full_html=True, default_width="900px")
        print(f"  wrote {out_path.name}")


def main():
    here = Path(__file__).parent
    out = here / "screenshots"

    # ────────────────────────────────────────────────────────────────────
    # Part 0 — 복소 정현파 e^(j2πt). 모든 것의 기초.
    # ────────────────────────────────────────────────────────────────────
    sig_z = signals.complex_exp(omega=2 * np.pi)

    print("[0A] e^(j2πt), wound at ω=+2π — MATCH (perfect straight line, final=10)")
    render_scenario(out, "0A_cexp_match_pos", sig_z, omega_focus=2 * np.pi)

    print("[0B] e^(j2πt), wound at ω=-2π — NEG ω, NO MATCH (only one direction!)")
    render_scenario(out, "0B_cexp_neg_no_match", sig_z, omega_focus=-2 * np.pi)

    print("[0C] e^(j2πt), wound at ω=π — orthogonal")
    render_scenario(out, "0C_cexp_orthogonal", sig_z, omega_focus=np.pi)

    print("[0D] e^(j2πt), wound at ω=2π·0.85 — irrational quasi-mismatch")
    render_scenario(out, "0D_cexp_quasi", sig_z, omega_focus=2 * np.pi * 0.85)

    # ────────────────────────────────────────────────────────────────────
    # Part A — 단일 cos(2π·t). 단일 톤 + winding ω 3가지로 원리 직접 관찰.
    # ────────────────────────────────────────────────────────────────────
    sig_cos = signals.cosine(omega=2 * np.pi)

    print("[A1] cos(2π·t), wound at ω=+2π — MATCH (final → 5.0 on RE)")
    render_scenario(out, "A1_cos_match_pos", sig_cos, omega_focus=2 * np.pi)

    print("[A1n] cos(2π·t), wound at ω=-2π — also matches (cos has both ±ω₀)")
    render_scenario(out, "A1n_cos_match_neg", sig_cos, omega_focus=-2 * np.pi)

    print("[A2] cos(2π·t), wound at ω=π — ORTHOGONAL (final → 0)")
    render_scenario(out, "A2_cos_orthogonal", sig_cos, omega_focus=np.pi)

    print("[A3] cos(2π·t), wound at ω=2π·0.85 — IRRATIONAL MISMATCH (small loop)")
    render_scenario(out, "A3_cos_quasi", sig_cos, omega_focus=2 * np.pi * 0.85)

    # ────────────────────────────────────────────────────────────────────
    # Part B — 단일 sin(2π·t). cos과 같은 ω 3종.
    # 핵심: matching 시 trail이 *허수축 음의 방향* (cos은 실수축 양의 방향).
    # ────────────────────────────────────────────────────────────────────
    sig_sin = signals.sine(omega=2 * np.pi)

    print("[B1] sin(2π·t), wound at ω=+2π — MATCH (final → 0 - 5j)")
    render_scenario(out, "B1_sin_match_pos", sig_sin, omega_focus=2 * np.pi)

    print("[B1n] sin(2π·t), wound at ω=-2π — MATCH with OPPOSITE sign (0 + 5j)")
    render_scenario(out, "B1n_sin_match_neg", sig_sin, omega_focus=-2 * np.pi)

    print("[B2] sin(2π·t), wound at ω=π — ORTHOGONAL (final → 0)")
    render_scenario(out, "B2_sin_orthogonal", sig_sin, omega_focus=np.pi)

    print("[B3] sin(2π·t), wound at ω=2π·0.85 — IRRATIONAL MISMATCH "
          "(small loop on RE axis, mirror of cos)")
    render_scenario(out, "B3_sin_quasi", sig_sin, omega_focus=2 * np.pi * 0.85)

    # ────────────────────────────────────────────────────────────────────
    # Part C — 감쇠 정현파 e^(-t)·cos(2π·t)·u(t). RLC 자연 응답.
    # 무한 적분 가능 → trail이 진짜로 한 점에 수렴.
    # ────────────────────────────────────────────────────────────────────
    sig_damped = signals.decaying_exp(rate=1.0) * signals.cosine(omega=2 * np.pi)

    print("[C1] e^(-t)·cos(2π·t)·u(t), wound at ω=2π — MATCH (resonance, |X|≈0.5)")
    render_scenario(out, "C1_damped_match", sig_damped,
                    omega_focus=2 * np.pi, t_max=15.0, n_samples=3000)

    print("[C2] e^(-t)·cos(2π·t)·u(t), wound at ω=4π — OFF-RESONANCE (small spiral)")
    render_scenario(out, "C2_damped_off", sig_damped,
                    omega_focus=4 * np.pi, t_max=15.0, n_samples=3000)

    # ────────────────────────────────────────────────────────────────────
    # Part D — 비대칭 vs 짝함수 비교 (causal exp vs two-sided exp).
    # X(jω)의 *위상 회전* vs *순수 실수성*의 시각적 대비.
    # ────────────────────────────────────────────────────────────────────
    sig_causal = signals.decaying_exp(rate=1.0)  # e^(-t)·u(t), 비대칭
    print("[D1] e^(-t)·u(t), wound at ω=1 — asymmetric → trail spirals to 0.5-0.5j")
    render_scenario(out, "D1_causal_w1", sig_causal, omega_focus=1.0,
                    t_max=15.0, n_samples=3000)

    sig_two_sided = signals.two_sided_exp(rate=1.0)  # e^(-|t|), 짝함수
    print("[D2] e^(-|t|), wound at ω=1 — even → trail stays on RE axis, ends at 1.0")
    render_scenario(out, "D2_two_sided_w1", sig_two_sided, omega_focus=1.0,
                    t_min=-10.0, t_max=10.0, n_samples=4000)

    # ────────────────────────────────────────────────────────────────────
    # Part E — Time shift = phase rotation (linear phase property).
    # cos(2π·(t-α))을 ω=2π로 winding하면 trail이 e^(-jωα) 만큼 회전.
    # α=1/4면 정확히 sin(2πt)의 trail (Part B1과 동일).
    # ────────────────────────────────────────────────────────────────────
    omega0 = 2 * np.pi
    # E2: α = 1/4 → trail이 0 - 5j (sin matching과 같음)
    sig_shift_q = signals.cosine(omega=omega0, phase=-omega0 * 0.25)
    print("[E2] cos(2π·(t-1/4)) @ ω=2π — equals sin(2πt) trail (0-5j)")
    render_scenario(out, "E2_shift_quarter", sig_shift_q, omega_focus=omega0)
    # E3: α = 1/8 → 45° 회전 (3.54 - 3.54j)
    sig_shift_e = signals.cosine(omega=omega0, phase=-omega0 * 0.125)
    print("[E3] cos(2π·(t-1/8)) @ ω=2π — 45° rotation (3.54 - 3.54j)")
    render_scenario(out, "E3_shift_eighth", sig_shift_e, omega_focus=omega0)

    # ────────────────────────────────────────────────────────────────────
    # Part F — Spectrum X(jω), 3D 시각화
    # ────────────────────────────────────────────────────────────────────
    out.mkdir(parents=True, exist_ok=True)

    def render_spectrum(scenario, sig_, t_lim, n_t, omega_lim, n_omega, view='re_im'):
        an = FourierAnalyzer(sig_, t_min=-t_lim if t_lim < 0 else 0.0,
                             t_max=t_lim, n_samples=n_t)
        omegas, X = an.spectrum(-omega_lim, omega_lim, n_omega)
        p = SpectrumPlot(view=view, show_inset=True)
        p.show_spectrum(omegas, X)
        out_path = out / f"{scenario}__e_spectrum.html"
        p.figure.write_html(str(out_path), include_plotlyjs="cdn",
                            full_html=True, default_width="900px")
        print(f"  wrote {out_path.name}")

    print("[F1] e^(-t)·u(t) spectrum — asymmetric, X(jω)=1/(1+jω)")
    render_spectrum("F1_causal", signals.decaying_exp(rate=1.0),
                    t_lim=20.0, n_t=4000, omega_lim=10.0, n_omega=400)

    print("[F2] e^(-|t|) spectrum — even, X(jω)=2/(1+ω²) (purely real)")
    an_two = FourierAnalyzer(signals.two_sided_exp(rate=1.0),
                             t_min=-15.0, t_max=15.0, n_samples=4000)
    omegas_two, X_two = an_two.spectrum(-10.0, 10.0, 400)
    p = SpectrumPlot(view='re_im', show_inset=True)
    p.show_spectrum(omegas_two, X_two)
    out_path = out / "F2_two_sided__e_spectrum.html"
    p.figure.write_html(str(out_path), include_plotlyjs="cdn",
                        full_html=True, default_width="900px")
    print(f"  wrote {out_path.name}")

    print("[F3] gaussian spectrum — self-dual, both gaussian")
    render_spectrum("F3_gaussian", signals.gaussian(sigma=1.0),
                    t_lim=8.0, n_t=2000, omega_lim=5.0, n_omega=300)

    print("[F4] damped cos spectrum — peaks near ±2π (mag/phase view)")
    sig_dc = signals.decaying_exp(rate=1.0) * signals.cosine(omega=2*np.pi)
    an_dc = FourierAnalyzer(sig_dc, t_min=0.0, t_max=20.0, n_samples=4000)
    omegas_dc, X_dc = an_dc.spectrum(-15.0, 15.0, 600)
    p = SpectrumPlot(view='mag_phase', show_inset=True)
    p.show_spectrum(omegas_dc, X_dc)
    out_path = out / "F4_damped_cos__e_spectrum.html"
    p.figure.write_html(str(out_path), include_plotlyjs="cdn",
                        full_html=True, default_width="900px")
    print(f"  wrote {out_path.name}")

    # ────────────────────────────────────────────────────────────────────
    # Part G — Inverse Accumulation
    # ────────────────────────────────────────────────────────────────────

    def render_inverse(scenario, sig_, t_lim, n_t, t_fix, omega_lim, n_omega,
                       order='monotonic', target=None, progress_index=None):
        an = FourierAnalyzer(sig_, t_min=-t_lim if t_lim < 0 else 0.0,
                             t_max=t_lim, n_samples=n_t)
        omegas, arrows, accum = an.inverse_accumulation(
            t_fix=t_fix, omega_min=-omega_lim, omega_max=omega_lim,
            n_omega=n_omega, order=order,
        )
        p = InverseAccumulationPlot()
        p.show_accumulation(omegas, arrows, accum,
                            target=complex(target),
                            progress_index=progress_index)
        suffix = f"__f_inverse"
        if progress_index is not None:
            suffix += f"_k{progress_index:04d}"
        out_path = out / f"{scenario}{suffix}.html"
        p.figure.write_html(str(out_path), include_plotlyjs="cdn",
                            full_html=True, default_width="900px")
        print(f"  wrote {out_path.name}")

    # G1: gaussian @ t=1, monotonic
    print("[G1] gaussian inverse @ t=1, monotonic — recovers x(1)=e^(-1/2)≈0.6065")
    render_inverse("G1_gaussian_mono", signals.gaussian(sigma=1.0),
                   t_lim=8.0, n_t=4001, t_fix=1.0,
                   omega_lim=15.0, n_omega=400,
                   order='monotonic', target=np.exp(-0.5))

    # G2: 같은 신호, paired_by_abs
    print("[G2] gaussian inverse @ t=1, paired_by_abs — Im cancels at every step")
    render_inverse("G2_gaussian_paired", signals.gaussian(sigma=1.0),
                   t_lim=8.0, n_t=4001, t_fix=1.0,
                   omega_lim=15.0, n_omega=400,
                   order='paired_by_abs', target=np.exp(-0.5))

    # G3: causal exp, 진행 단계별
    print("[G3] causal exp inverse @ t=0.5 — progress snapshots 25/50/100%")
    sig_c = signals.decaying_exp(rate=1.0)
    an_c = FourierAnalyzer(sig_c, t_min=0.0, t_max=25.0, n_samples=5000)
    omegas_c, arrows_c, accum_c = an_c.inverse_accumulation(
        t_fix=0.5, omega_min=-30.0, omega_max=30.0, n_omega=600,
    )
    target_c = complex(np.exp(-0.5))
    n = arrows_c.size
    for frac in [0.25, 0.5, 1.0]:
        k = int(frac * (n - 1))
        p = InverseAccumulationPlot()
        p.show_accumulation(omegas_c, arrows_c, accum_c, target=target_c,
                            progress_index=k)
        out_path = out / f"G3_causal_progress__f_inverse_k{k:04d}.html"
        p.figure.write_html(str(out_path), include_plotlyjs="cdn",
                            full_html=True, default_width="900px")
        print(f"  wrote {out_path.name}")

    print("\nAll screenshots in:", out)


if __name__ == "__main__":
    main()
