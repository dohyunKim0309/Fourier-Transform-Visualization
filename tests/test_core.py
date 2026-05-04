"""
ftvis.core 단위 테스트.

각 테스트는 DESIGN.md §1.2 학습 목표(L1~L4) 또는 §6 Day 1/1.5 done definition에
대응한다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ftvis.core import (
    ComplexSignal,
    FourierAnalyzer,
    Signal,
    WoundSignal,
    signals,
    _running_integral,
)


TWO_PI = 2.0 * math.pi


# ─────────────────────────────────────────────────────────────────────────────
# Signal: 기본 평가와 합성
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalBasics:

    def test_from_lambda_returns_real_array(self):
        sig = Signal.from_lambda(lambda t: np.cos(t))
        out = sig.sample(np.linspace(0, 1, 10))
        assert out.dtype == np.float64
        assert out.shape == (10,)

    def test_scalar_lambda_broadcasts(self):
        sig = Signal.from_lambda(lambda t: 3.0)
        out = sig.sample(np.linspace(0, 1, 5))
        assert out.shape == (5,)
        assert np.allclose(out, 3.0)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            Signal(42)  # type: ignore[arg-type]

    def test_repr_contains_name(self):
        sig = signals.cosine(omega=2 * math.pi)
        r = repr(sig)
        assert "Signal" in r
        assert "cos" in r


class TestSignalComposition:
    """합성 연산자가 sample 결과에서 선형성·곱을 보존하는지."""

    def test_addition_real(self):
        a = signals.cosine(omega=TWO_PI * 2.0)
        b = signals.cosine(omega=TWO_PI * 3.0)
        c = a + b
        t = np.linspace(0, 1, 100)
        expected = np.cos(TWO_PI * 2.0 * t) + np.cos(TWO_PI * 3.0 * t)
        np.testing.assert_allclose(c.sample(t), expected, atol=1e-12)

    def test_subtraction(self):
        a = signals.cosine(omega=TWO_PI * 2.0)
        b = signals.cosine(omega=TWO_PI * 3.0)
        t = np.linspace(0, 1, 100)
        np.testing.assert_allclose(
            (a - b).sample(t),
            np.cos(TWO_PI * 2.0 * t) - np.cos(TWO_PI * 3.0 * t),
            atol=1e-12,
        )

    def test_scalar_multiplication_left_and_right(self):
        a = signals.cosine(omega=TWO_PI * 2.0)
        t = np.linspace(0, 1, 50)
        np.testing.assert_allclose((0.5 * a).sample(t),
                                   0.5 * np.cos(TWO_PI * 2.0 * t), atol=1e-12)
        np.testing.assert_allclose((a * 0.5).sample(t),
                                   0.5 * np.cos(TWO_PI * 2.0 * t), atol=1e-12)

    def test_signal_times_signal(self):
        env = signals.gaussian(sigma=0.5, center=0.5)
        car = signals.cosine(omega=TWO_PI * 5.0)
        prod = env * car
        t = np.linspace(0, 1, 200)
        expected = np.exp(-((t - 0.5) ** 2) / (2 * 0.25)) * np.cos(TWO_PI * 5.0 * t)
        np.testing.assert_allclose(prod.sample(t), expected, atol=1e-12)

    def test_negation_preserves_real(self):
        a = signals.cosine(omega=TWO_PI * 2.0)
        neg = -a
        assert isinstance(neg, Signal)
        assert not isinstance(neg, ComplexSignal)
        t = np.linspace(0, 1, 50)
        np.testing.assert_allclose(neg.sample(t), -np.cos(TWO_PI * 2.0 * t), atol=1e-12)

    def test_radd_with_scalar(self):
        a = signals.cosine(omega=TWO_PI * 2.0)
        t = np.linspace(0, 1, 50)
        np.testing.assert_allclose((1.0 + a).sample(t),
                                   1.0 + np.cos(TWO_PI * 2.0 * t), atol=1e-12)

    def test_invalid_operand_raises(self):
        a = signals.cosine(omega=TWO_PI * 2.0)
        with pytest.raises(TypeError):
            _ = a + "not a signal"  # type: ignore[operator]


# ─────────────────────────────────────────────────────────────────────────────
# 새 위계: ComplexSignal, 승격 규칙
# ─────────────────────────────────────────────────────────────────────────────

class TestClassHierarchy:

    def test_complex_signal_is_signal(self):
        z = signals.complex_exp(omega=TWO_PI * 1.0)
        assert isinstance(z, ComplexSignal)
        assert isinstance(z, Signal)

    def test_signal_is_not_complex_signal(self):
        s = signals.cosine(omega=TWO_PI * 1.0)
        assert isinstance(s, Signal)
        assert not isinstance(s, ComplexSignal)

    def test_complex_signal_returns_complex_array(self):
        z = signals.complex_exp(omega=TWO_PI * 1.0)
        out = z.sample(np.linspace(0, 1, 10))
        assert out.dtype == np.complex128

    def test_complex_signal_unit_magnitude(self):
        z = signals.complex_exp(omega=TWO_PI * 3.0)
        out = z.sample(np.linspace(0, 2, 100))
        np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-12)


class TestPromotionRules:
    """DESIGN §4.1.2 타입 승격 표 검증."""

    def test_real_plus_real_is_real(self):
        a = signals.cosine(omega=TWO_PI)
        b = signals.sine(omega=TWO_PI)
        c = a + b
        assert isinstance(c, Signal)
        assert not isinstance(c, ComplexSignal)
        t = np.linspace(0, 1, 50)
        assert c.sample(t).dtype == np.float64

    def test_real_plus_complex_is_complex(self):
        a = signals.cosine(omega=TWO_PI)
        z = signals.complex_exp(omega=TWO_PI)
        c = a + z
        assert isinstance(c, ComplexSignal)
        t = np.linspace(0, 1, 50)
        assert c.sample(t).dtype == np.complex128

    def test_complex_plus_real_is_complex(self):
        a = signals.cosine(omega=TWO_PI)
        z = signals.complex_exp(omega=TWO_PI)
        c = z + a
        assert isinstance(c, ComplexSignal)

    def test_real_times_complex_is_complex(self):
        a = signals.gaussian(sigma=0.5)
        z = signals.complex_exp(omega=TWO_PI * 2.0)
        c = a * z
        assert isinstance(c, ComplexSignal)

    def test_real_scalar_times_real_signal_is_real(self):
        a = signals.cosine(omega=TWO_PI)
        c = 2.0 * a
        assert isinstance(c, Signal)
        assert not isinstance(c, ComplexSignal)

    def test_complex_scalar_times_real_signal_is_complex(self):
        a = signals.cosine(omega=TWO_PI)
        c = 1j * a
        assert isinstance(c, ComplexSignal)
        t = np.linspace(0, 1, 50)
        np.testing.assert_allclose(c.sample(t),
                                   1j * np.cos(TWO_PI * t), atol=1e-12)

    def test_negate_complex_is_complex(self):
        z = signals.complex_exp(omega=TWO_PI)
        neg = -z
        assert isinstance(neg, ComplexSignal)


# ─────────────────────────────────────────────────────────────────────────────
# 표준 신호 팩토리
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalFactories:

    def test_cosine_known_value(self):
        sig = signals.cosine(omega=TWO_PI, amp=2.0, phase=0.0)
        assert sig.sample(np.array([0.0]))[0] == pytest.approx(2.0)

    def test_sine_known_value(self):
        sig = signals.sine(omega=TWO_PI * 0.25)
        assert sig.sample(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-12)

    def test_complex_exp_known_value(self):
        z = signals.complex_exp(omega=math.pi / 2)
        # exp(j·π/2·1) = exp(j·π/2) = j
        out = z.sample(np.array([1.0]))[0]
        assert out.real == pytest.approx(0.0, abs=1e-12)
        assert out.imag == pytest.approx(1.0, abs=1e-12)

    def test_gaussian_peak_at_center(self):
        sig = signals.gaussian(sigma=0.5, center=1.0, amp=3.0)
        t = np.linspace(-2, 4, 1000)
        idx = int(np.argmax(sig.sample(t)))
        assert t[idx] == pytest.approx(1.0, abs=0.01)

    def test_rect_zero_outside(self):
        sig = signals.rect(width=2.0, center=0.0, amp=1.0)
        assert sig.sample(np.array([0.0]))[0] == 1.0
        assert sig.sample(np.array([2.0]))[0] == 0.0

    def test_step_at_t0(self):
        sig = signals.step(t0=1.0, amp=2.0)
        vals = sig.sample(np.array([0.99, 1.0, 1.01]))
        np.testing.assert_array_equal(vals, [0.0, 2.0, 2.0])

    def test_decaying_exp_causal(self):
        sig = signals.decaying_exp(rate=1.0, amp=1.0, t0=0.0)
        vals = sig.sample(np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(vals, [0.0, 1.0, math.exp(-1.0)], atol=1e-12)

    def test_invalid_gaussian_sigma_raises(self):
        with pytest.raises(ValueError):
            signals.gaussian(sigma=0)
        with pytest.raises(ValueError):
            signals.gaussian(sigma=-1)

    def test_invalid_rect_width_raises(self):
        with pytest.raises(ValueError):
            signals.rect(width=0)


# ─────────────────────────────────────────────────────────────────────────────
# _running_integral
# ─────────────────────────────────────────────────────────────────────────────

class TestRunningIntegral:

    def test_constant_function_linear_growth(self):
        t = np.linspace(0, 1, 101)
        y = np.ones_like(t, dtype=np.complex128)
        out = _running_integral(t, y)
        np.testing.assert_allclose(out.real, t, atol=1e-12)
        np.testing.assert_allclose(out.imag, 0.0, atol=1e-12)

    def test_linear_function_quadratic_growth(self):
        t = np.linspace(0, 2, 1001)
        y = t.astype(np.complex128)
        out = _running_integral(t, y)
        np.testing.assert_allclose(out.real, t**2 / 2, atol=1e-6)

    def test_first_value_is_zero(self):
        t = np.linspace(0, 1, 50)
        y = np.exp(t).astype(np.complex128)
        out = _running_integral(t, y)
        assert out[0] == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            _running_integral(np.array([0.0, 1.0]),
                              np.array([1.0+0j, 2.0+0j, 3.0+0j]))

    def test_complex_integration(self):
        t = np.linspace(0, math.pi, 1001)
        y = np.exp(1j * t)
        out = _running_integral(t, y)
        expected = -1j * (np.exp(1j * t) - 1.0)
        np.testing.assert_allclose(out, expected, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# FourierAnalyzer 구성
# ─────────────────────────────────────────────────────────────────────────────

class TestFourierAnalyzerConstruction:

    def test_t_array_shape_and_endpoints(self):
        sig = signals.cosine(omega=TWO_PI)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=4.0, n_samples=500)
        t = an.t
        assert t.shape == (500,)
        assert t[0] == 0.0
        assert t[-1] == pytest.approx(4.0)

    def test_t_is_cached(self):
        sig = signals.cosine(omega=TWO_PI)
        an = FourierAnalyzer(sig)
        assert an.t is an.t

    def test_invalid_signal_type_raises(self):
        with pytest.raises(TypeError):
            FourierAnalyzer(lambda t: t)  # type: ignore[arg-type]

    def test_invalid_t_range_raises(self):
        sig = signals.cosine(omega=TWO_PI)
        with pytest.raises(ValueError):
            FourierAnalyzer(sig, t_min=1.0, t_max=1.0)
        with pytest.raises(ValueError):
            FourierAnalyzer(sig, t_min=2.0, t_max=1.0)

    def test_too_few_samples_raises(self):
        sig = signals.cosine(omega=TWO_PI)
        with pytest.raises(ValueError):
            FourierAnalyzer(sig, n_samples=1)


# ─────────────────────────────────────────────────────────────────────────────
# *** Day 1 done definition: 학습 목표 L2~L4의 수치 검증 ***
# ─────────────────────────────────────────────────────────────────────────────

class TestWindingMatchesFrequency:
    """L2: ω가 신호의 한 성분 ω₀와 일치할 때, 누적 벡터가 한 방향으로 자라남."""

    def test_cosine_wound_at_matching_omega_grows_monotonically(self):
        # cos(ω₀t) winding e^(-jω₀t): 적분이 (T/2) + 빠르게 cancel.
        omega0 = TWO_PI * 2.0  # 2 Hz
        sig = signals.cosine(omega=omega0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=1.0, n_samples=2001)
        ws = an.wound_at_omega(omega0)

        assert np.all(np.diff(ws.running_integral.real) >= -1e-6)
        T = an.t_max - an.t_min
        assert ws.final_integral.real == pytest.approx(T / 2, rel=1e-3)
        assert abs(ws.final_integral.imag) < 1e-3

    def test_sine_wound_at_matching_omega_imag_grows(self):
        omega0 = TWO_PI * 2.0
        sig = signals.sine(omega=omega0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=1.0, n_samples=2001)
        ws = an.wound_at_omega(omega0)
        T = an.t_max - an.t_min
        assert ws.final_integral.imag == pytest.approx(-T / 2, rel=1e-3)
        assert abs(ws.final_integral.real) < 1e-3

    def test_complex_exp_perfect_cancellation_match(self):
        omega0 = TWO_PI * 3.0
        sig = signals.complex_exp(omega=omega0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=2.0, n_samples=1001)
        ws = an.wound_at_omega(omega0)
        T = an.t_max - an.t_min
        assert ws.final_integral.real == pytest.approx(T, abs=1e-10)
        assert abs(ws.final_integral.imag) < 1e-10


class TestWindingCancelsAtMismatch:
    """L3: ω ≠ ω₀이면 충분히 긴 구간에서 누적 벡터가 작다."""

    def test_complex_exp_orthogonal_omegas_cancel(self):
        omega0 = TWO_PI * 2.0
        sig = signals.complex_exp(omega=omega0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=1.0, n_samples=4001)
        for k_hz in [1.0, 3.0, 5.0]:
            ws = an.wound_at_omega(TWO_PI * k_hz)
            assert abs(ws.final_integral) < 1e-3, \
                f"orthogonal omega={TWO_PI * k_hz} should cancel, got {ws.final_integral}"

    def test_cosine_far_from_signal_omega_is_small(self):
        sig = signals.cosine(omega=TWO_PI * 2.0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=10.0, n_samples=10001)
        ws = an.wound_at_omega(TWO_PI * 7.0)
        # 매칭은 ≈ T/2 = 5.0; mismatch는 그 1/10 미만이어야.
        assert abs(ws.final_integral) < 0.5


class TestLinearity:
    """L4 보조: winding의 선형성."""

    def test_linearity_of_winding(self):
        a = signals.cosine(omega=TWO_PI * 2.0)
        b = signals.cosine(omega=TWO_PI * 3.0)
        combined = a + b
        an_a = FourierAnalyzer(a, t_min=0.0, t_max=1.0, n_samples=2001)
        an_b = FourierAnalyzer(b, t_min=0.0, t_max=1.0, n_samples=2001)
        an_c = FourierAnalyzer(combined, t_min=0.0, t_max=1.0, n_samples=2001)

        for omega_hz in [1.5, 2.0, 2.5, 3.0]:
            omega = TWO_PI * omega_hz
            x_combined = an_c.wound_at_omega(omega).final_integral
            x_sum = (an_a.wound_at_omega(omega).final_integral
                     + an_b.wound_at_omega(omega).final_integral)
            assert x_combined == pytest.approx(x_sum, abs=1e-10), \
                f"linearity failed at omega={omega}"

    def test_two_peaks_in_combined_signal(self):
        sig = signals.cosine(omega=TWO_PI * 2.0) + signals.cosine(omega=TWO_PI * 3.0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=2.0, n_samples=4001)
        peak_2 = abs(an.wound_at_omega(TWO_PI * 2.0).final_integral)
        peak_3 = abs(an.wound_at_omega(TWO_PI * 3.0).final_integral)
        valley = abs(an.wound_at_omega(TWO_PI * 2.5).final_integral)
        assert peak_2 == pytest.approx(1.0, rel=1e-3)
        assert peak_3 == pytest.approx(1.0, rel=1e-3)
        assert valley < 0.3 * peak_2


# ─────────────────────────────────────────────────────────────────────────────
# WoundSignal: 새 위계 + 데이터 컨테이너 역할
# ─────────────────────────────────────────────────────────────────────────────

class TestWoundSignalStructure:

    def test_is_complex_signal_and_signal(self):
        sig = signals.cosine(omega=TWO_PI * 2.0)
        an = FourierAnalyzer(sig)
        ws = an.wound_at_omega(TWO_PI * 2.0)
        assert isinstance(ws, WoundSignal)
        assert isinstance(ws, ComplexSignal)
        assert isinstance(ws, Signal)

    def test_all_fields_present_and_shaped(self):
        sig = signals.cosine(omega=TWO_PI * 2.0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=1.0, n_samples=500)
        ws = an.wound_at_omega(TWO_PI * 2.0)
        assert ws.t.shape == (500,)
        assert ws.signal_values.shape == (500,)
        assert ws.wound_values.shape == (500,)
        assert ws.running_integral.shape == (500,)
        assert ws.signal_values.dtype == np.complex128
        assert ws.wound_values.dtype == np.complex128
        assert ws.running_integral.dtype == np.complex128
        assert ws.omega == pytest.approx(TWO_PI * 2.0)
        assert ws.base is sig

    def test_wound_equals_signal_times_carrier(self):
        sig = signals.gaussian(sigma=0.3, center=0.5)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=1.0, n_samples=200)
        ws = an.wound_at_omega(TWO_PI * 2.0)
        carrier = np.exp(-1j * TWO_PI * 2.0 * ws.t)
        np.testing.assert_allclose(ws.wound_values, ws.signal_values * carrier, atol=1e-12)

    def test_running_integral_starts_at_zero(self):
        sig = signals.cosine(omega=TWO_PI)
        an = FourierAnalyzer(sig)
        ws = an.wound_at_omega(TWO_PI)
        assert ws.running_integral[0] == 0.0

    def test_final_integral_property(self):
        sig = signals.cosine(omega=TWO_PI)
        an = FourierAnalyzer(sig)
        ws = an.wound_at_omega(TWO_PI)
        assert ws.final_integral == complex(ws.running_integral[-1])


class TestWoundSignalIsCallable:
    """WoundSignal은 ComplexSignal이므로 임의의 t 배열에 다시 sample 가능."""

    def test_sample_at_new_t_matches_definition(self):
        # x(t) = cos(2π·2t), wound at ω=2π·1 → wound(t) = cos(2π·2t)·e^(-j·2π·1·t)
        sig = signals.cosine(omega=TWO_PI * 2.0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=1.0, n_samples=100)
        ws = an.wound_at_omega(TWO_PI * 1.0)

        # 분석에 사용된 t와는 다른 t 배열에서 평가.
        t_new = np.linspace(0, 0.5, 47)
        out = ws.sample(t_new)
        expected = np.cos(TWO_PI * 2.0 * t_new) * np.exp(-1j * TWO_PI * 1.0 * t_new)
        np.testing.assert_allclose(out, expected, atol=1e-12)
        assert out.dtype == np.complex128

    def test_sample_at_original_t_matches_wound_values(self):
        sig = signals.gaussian(sigma=0.3) * signals.cosine(omega=TWO_PI * 2.0)
        an = FourierAnalyzer(sig, t_min=-1.0, t_max=1.0, n_samples=200)
        ws = an.wound_at_omega(TWO_PI * 2.0)
        # 원래 t에서 다시 샘플링 → wound_values와 일치
        np.testing.assert_allclose(ws.sample(ws.t), ws.wound_values, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 알려진 푸리에 변환
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownFourierTransforms:

    def test_gaussian_fourier_transform(self):
        # x(t) = exp(-t²/(2σ²)). 우리 정의 X(jω) = ∫ x(t) e^(-jωt) dt
        # = σ·√(2π)·exp(-σ²ω²/2)
        sigma = 0.5
        sig = signals.gaussian(sigma=sigma, center=0.0)
        an = FourierAnalyzer(sig, t_min=-5*sigma, t_max=5*sigma, n_samples=4001)

        for omega_hz in [0.0, 0.5, 1.0, 2.0]:
            omega = TWO_PI * omega_hz
            X_numeric = an.wound_at_omega(omega).final_integral
            X_analytic = sigma * math.sqrt(TWO_PI) * math.exp(
                -sigma**2 * omega**2 / 2.0
            )
            assert X_numeric.real == pytest.approx(X_analytic, rel=1e-3, abs=1e-4)
            assert abs(X_numeric.imag) < 1e-6

    def test_rect_sinc_transform(self):
        # rect_T(t) (폭 T, 중심 0, 높이 1)의 FT는
        # X(jω) = T · sinc(ωT/(2π)) (정규화된 sinc).
        T = 1.0
        sig = signals.rect(width=T, center=0.0, amp=1.0)
        an = FourierAnalyzer(sig, t_min=-2.0, t_max=2.0, n_samples=8001)

        for omega_hz in [0.0, 0.5, 1.0, 1.5, 2.0]:
            omega = TWO_PI * omega_hz
            X_numeric = an.wound_at_omega(omega).final_integral
            X_analytic = T * np.sinc(omega * T / (2.0 * math.pi))
            assert X_numeric.real == pytest.approx(X_analytic, abs=2e-3)
            assert abs(X_numeric.imag) < 2e-3

    def test_causal_exp_transform(self):
        # x(t) = e^(-at)·u(t),  X(jω) = 1/(a + jω)
        a = 1.0
        sig = signals.decaying_exp(rate=a, amp=1.0, t0=0.0)
        # 충분히 긴 적분 구간 (e^(-1·15) ≈ 3e-7)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=15.0, n_samples=8001)

        for omega in [0.0, 0.5, 1.0, 2.0, 5.0]:
            X_numeric = an.wound_at_omega(omega).final_integral
            X_analytic = 1.0 / (a + 1j * omega)
            assert X_numeric.real == pytest.approx(X_analytic.real, abs=1e-3)
            assert X_numeric.imag == pytest.approx(X_analytic.imag, abs=1e-3)

    def test_two_sided_exp_transform(self):
        # x(t) = e^(-a|t|),  X(jω) = 2a / (a² + ω²)
        # 짝함수 → X(jω)는 순수 실수.
        a = 1.0
        sig = signals.two_sided_exp(rate=a, amp=1.0, center=0.0)
        an = FourierAnalyzer(sig, t_min=-15.0, t_max=15.0, n_samples=8001)

        for omega in [0.0, 0.5, 1.0, 2.0, 5.0]:
            X_numeric = an.wound_at_omega(omega).final_integral
            X_analytic = 2.0 * a / (a**2 + omega**2)
            assert X_numeric.real == pytest.approx(X_analytic, abs=1e-3)
            assert abs(X_numeric.imag) < 1e-6

    def test_spectrum_returns_correct_shape_and_values(self):
        # gaussian의 spectrum이 해석값과 일치 (Mode B의 가장 기본 검증)
        sig = signals.gaussian(sigma=1.0, center=0.0)
        an = FourierAnalyzer(sig, t_min=-8.0, t_max=8.0, n_samples=4001)
        omegas, X = an.spectrum(omega_min=-3.0, omega_max=3.0, n_omega=21)
        assert omegas.shape == (21,)
        assert X.shape == (21,)
        assert X.dtype == np.complex128
        expected = math.sqrt(TWO_PI) * np.exp(-omegas**2 / 2)
        np.testing.assert_allclose(X.real, expected, atol=1e-4)
        np.testing.assert_allclose(X.imag, 0.0, atol=1e-6)

    def test_spectrum_invalid_args_raise(self):
        sig = signals.gaussian(sigma=1.0)
        an = FourierAnalyzer(sig, t_min=-5, t_max=5, n_samples=200)
        with pytest.raises(ValueError):
            an.spectrum(0, 1, n_omega=1)
        with pytest.raises(ValueError):
            an.spectrum(2, 1, n_omega=10)

    def test_inverse_accumulation_recovers_signal(self):
        # gaussian의 spectrum으로부터 역변환이 원본 x(t)를 복원하는지.
        sig = signals.gaussian(sigma=1.0, center=0.0)
        an = FourierAnalyzer(sig, t_min=-8.0, t_max=8.0, n_samples=4001)
        for t_fix in [0.0, 0.5, 1.0, 2.0]:
            _, _, accum = an.inverse_accumulation(
                t_fix=t_fix, omega_min=-30.0, omega_max=30.0, n_omega=2001,
            )
            expected_x = math.exp(-t_fix**2 / 2)
            assert accum[-1].real == pytest.approx(expected_x, abs=1e-3)
            assert abs(accum[-1].imag) < 1e-6

    def test_inverse_accumulation_arrows_match_spectrum(self):
        # arrows[k] = (1/(2π)) · X(ω_k) · e^(jω_k t_fix) · dω
        # accumulated = cumsum(arrows). 두 결과의 일관성.
        sig = signals.two_sided_exp(rate=1.0)
        an = FourierAnalyzer(sig, t_min=-15.0, t_max=15.0, n_samples=8001)
        omegas, X = an.spectrum(-20.0, 20.0, 1001)
        d_omega = float(omegas[1] - omegas[0])
        t_fix = 0.5
        expected_arrows = (1/(2*math.pi)) * X * np.exp(1j*omegas*t_fix) * d_omega
        omegas_out, arrows, accum = an.inverse_accumulation(
            t_fix=t_fix, omega_min=-20.0, omega_max=20.0, n_omega=1001,
            order='monotonic',
        )
        np.testing.assert_allclose(omegas_out, omegas)
        np.testing.assert_allclose(arrows, expected_arrows, atol=1e-12)
        np.testing.assert_allclose(accum, np.cumsum(expected_arrows), atol=1e-12)

    def test_inverse_accumulation_order_paired_by_abs(self):
        # paired_by_abs는 |ω|가 단조증가하도록 정렬.
        sig = signals.gaussian(sigma=1.0)
        an = FourierAnalyzer(sig, t_min=-8.0, t_max=8.0, n_samples=2001)
        omegas, _, accum = an.inverse_accumulation(
            t_fix=1.0, omega_min=-10.0, omega_max=10.0, n_omega=21,
            order='paired_by_abs',
        )
        # |ω|가 단조증가
        abs_omegas = np.abs(omegas)
        assert np.all(np.diff(abs_omegas) >= -1e-12)
        # 최종 결과는 monotonic과 같아야 (cumsum이지만 합은 commutative)
        _, _, accum_mono = an.inverse_accumulation(
            t_fix=1.0, omega_min=-10.0, omega_max=10.0, n_omega=21,
            order='monotonic',
        )
        assert accum[-1] == pytest.approx(accum_mono[-1], abs=1e-12)

    def test_inverse_accumulation_invalid_order_raises(self):
        sig = signals.gaussian(sigma=1.0)
        an = FourierAnalyzer(sig, t_min=-5, t_max=5, n_samples=200)
        with pytest.raises(ValueError):
            an.inverse_accumulation(
                t_fix=0.0, omega_min=-5, omega_max=5, n_omega=10,
                order='bogus',
            )

    def test_damped_cosine_transform(self):
        # x(t) = e^(-at) · cos(ω₀ t) · u(t)
        # X(jω) = (a + jω) / ((a + jω)² + ω₀²)
        # — 비대칭 + 진동. ω = ±ω₀ 근처에 완만한 피크.
        a, omega0 = 1.0, 2 * math.pi
        sig = signals.decaying_exp(rate=a) * signals.cosine(omega=omega0)
        an = FourierAnalyzer(sig, t_min=0.0, t_max=15.0, n_samples=4001)

        for omega in [0.0, math.pi, 2 * math.pi, 3 * math.pi, 4 * math.pi]:
            X_numeric = an.wound_at_omega(omega).final_integral
            s = a + 1j * omega
            X_analytic = s / (s * s + omega0 * omega0)
            assert X_numeric.real == pytest.approx(X_analytic.real, abs=1e-3)
            assert X_numeric.imag == pytest.approx(X_analytic.imag, abs=1e-3)

        # 매칭 ω=ω₀에서 |X|가 다른 모든 테스트 ω보다 명확히 큼 (resonance peak)
        peak = abs(an.wound_at_omega(omega0).final_integral)
        for omega in [0.0, math.pi, 3 * math.pi, 4 * math.pi]:
            other = abs(an.wound_at_omega(omega).final_integral)
            assert peak > 2 * other, f"matching peak should dominate at ω={omega}"
