# ftvis — Fourier Transform Visualizer

> 연속 시간 신호의 푸리에 변환을 3차원 복소공간에서 인터랙티브하게 탐험하는 교육용 도구.

3Blue1Brown의 "winding machine" 시각화는 푸리에 변환을 직관적으로 보여주는 강력한 도구이지만, 본질적으로 2D 평면 시각화이다. 그래서 다음과 같은 핵심 개념이 그림으로 직접 보이지 않는다.

- **푸리에 변환 결과가 왜 복소수인가** — center of mass의 (x, y) 좌표를 복소수의 (Re, Im)로 해석한다는 추가 설명이 필요함.
- **음의 주파수의 의미** — 회전 방향이 반대인 헬릭스라는 것이 평면에서는 표현이 어렵다.
- **직교성(orthogonality)** — "회전이 멈춘 성분만 적분에서 살아남는다"가 평면 시각화에서는 "원점 부근으로 모인다"라는 간접적 신호로만 드러난다.

`ftvis`는 **시간축(또는 주파수축)을 한 축, 복소평면(Re, Im)을 다른 두 축**으로 두는 3D 시각화를 메인으로 하여, 위 개념들이 그림에서 직접 읽히도록 한다. 자세한 설계 의도는 [DESIGN.md](./DESIGN.md)를 참고.

## 설치

Python 3.10 이상.

```bash
git clone https://github.com/dkim7800/Signals_Systems.git
cd Signals_Systems
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[notebook]"
```

`-e .[notebook]`은 ftvis 자체를 editable로 설치하면서 jupyter/ipywidgets까지 같이 설치한다. 개발/테스트 의존성까지 원하면 `pip install -e ".[notebook,dev]"`.

## 5줄 데모

```python
import numpy as np
from ftvis import signals, FourierAnalyzer, ForwardIntegralPlot

sig = signals.cosine(omega=2*np.pi)
analyzer = FourierAnalyzer(sig, t_min=0.0, t_max=10.0, n_samples=2000)
ws = analyzer.wound_at_omega(2*np.pi)         # ω를 신호 주파수와 매칭
plot = ForwardIntegralPlot(); plot.update(ws, len(ws.t)-1); plot.figure.show()
```

ω가 신호 성분과 일치하면(`cos(2π·t)`를 ω = 2π로 winding), 누적 벡터가 한 방향으로 일관되게 자라며 끝점이 (5, 0) 부근에 도달한다. ω를 다른 값으로 바꾸면 누적 벡터가 작은 원을 그리며 제자리를 맴돈다 — 이게 직교성의 시각적 정체.

## 네 가지 패널

`ftvis.viz`의 핵심 4종 (Mode A, v0.1):

| 클래스 | 무엇을 그리는가 |
|---|---|
| `SignalPlot` | 시간 도메인 곡선 `x(t)` (2D) |
| `WindingHelixPlot` | 단위 헬릭스 `e^(-jωt)` (3D) |
| `WoundSignalPlot` | 변조된 헬릭스 `x(t)·e^(-jωt)` + envelope 회전체 (3D) |
| `ForwardIntegralPlot` | 누적 적분 `∫₀ᵗ x(τ)·e^(-jωτ) dτ` 의 3D 누적 벡터 + 2D 인셋 |

Mode B (v0.2)도 본격 구현되어 있음:

| 클래스 | 무엇을 그리는가 |
|---|---|
| `SpectrumPlot` | `X(jω)`를 ω축 + 복소평면 3D로 시각화 |
| `InverseAccumulationPlot` | 역변환 누적 — 화살표 머리잇기로 `x(t_fix)` 복원 |

모든 플롯은 같은 인터페이스를 따른다.

```python
plot = SomePlot(...)
plot.update(wound_signal, t_index)   # 데이터 갱신
plot.figure.show()                    # 또는 go.FigureWidget(plot.figure)
```

## 표준 신호 팩토리

`ftvis.signals`에서 즉시 쓸 수 있는 신호:

```python
from ftvis import signals
import numpy as np

signals.cosine(omega=2*np.pi)                          # cos(ω·t)
signals.sine(omega=2*np.pi, phase=np.pi/4)             # sin(ω·t + φ)
signals.complex_exp(omega=2*np.pi)                     # e^(jω·t)
signals.gaussian(sigma=1.0, center=0.0)                # exp(-t²/(2σ²))
signals.rect(width=1.0, center=0.0)                    # 사각 펄스
signals.step(t0=0.0)                                   # u(t-t0)
signals.decaying_exp(rate=1.0, t0=0.0)                 # e^(-rate·t)·u(t)
signals.two_sided_exp(rate=1.0, center=0.0)            # e^(-rate·|t|)

# 합성 (타입 자동 승격: Signal + ComplexSignal → ComplexSignal 등)
sig = signals.cosine(omega=2*np.pi) + signals.cosine(omega=3*np.pi)
sig = signals.gaussian(sigma=1.0, center=5.0) * signals.cosine(omega=3*np.pi)
```

모든 주파수는 angular frequency `ω` (rad/s)로 통일. Hz가 익숙하면 `ω = 2π·f`로 직접 변환 — API에 freq/omega 두 갈래를 두지 않는다 ([DESIGN §3.1](./DESIGN.md)).

## 데모 노트북

- [`examples/01_intro.ipynb`](./examples/01_intro.ipynb) — 복소 정현파에서 출발해 cos/sin → 감쇠 정현파 → 짝/홀 대칭 → time shift까지 점진적으로 쌓아 올리는 Mode A 투어. ω 매칭/직교/quasi-주기 미스매치를 모두 한 노트북 안에서 비교.
- [`examples/02_inverse.ipynb`](./examples/02_inverse.ipynb) — *다양한 신호*의 역변환 메커니즘. 가우시안·causal exp·two-sided exp 등을 SpectrumPlot과 InverseAccumulationPlot으로 분석. 화살표 머리잇기 누적, paired_by_abs 정렬에 의한 켤레 cancel. 여러 `t_fix`에서 `x(t_fix)` 복원 정확도 10⁻¹⁴ 수준.
- [`examples/03_rect_sinc.ipynb`](./examples/03_rect_sinc.ipynb) — `rect_W(t)` 한 신호의 *순방향+역방향* 양방향 분석. (Phase 1) sin(ωW) 항이 단위원 호의 chord에서 나옴 — (Phase 2) 1/ω 감쇠가 호 위 단위벡터의 시간 밀도에서 나옴 — (Phase 3) 두 항이 합쳐져 sinc — (Phase 4) sinc를 다시 rect로 역변환 — (Phase 5) linear phase = time shift property를 두 적분의 *동일한 화살표 시퀀스*로 시각 증명.

## 아키텍처 요약

3-레이어 구조. 각 레이어는 아래 레이어만 안다.

```
Layer 3 — app/    FourierExplorer (ipywidgets, 후속 작업)
                       ↑ uses
Layer 2 — viz/    *Plot 클래스들 (Plotly)
                       ↑ uses
Layer 1 — core/   Signal · ComplexSignal · WoundSignal · FourierAnalyzer
                  (numpy만 의존, 렌더러 무지)
```

## 테스트

```bash
pip install -e ".[dev]"
pytest
```

66개 테스트 (수치 정확도 검증 포함) 통과.

## 비목표 (v0.1)

명시적으로 빼두는 것들 (scope creep 방지).

- 이산 시간 신호 처리 (DFT/FFT, 윈도우 함수)
- 실시간 오디오 입력/스펙트로그램
- 2D 푸리에 변환 (이미지)
- 라플라스/Z 변환
- 수치 정확도 벤치마크 (이건 교육용이지 신호 처리 라이브러리가 아니다)

자세한 내용은 [DESIGN.md §7 Out of Scope](./DESIGN.md) 참조.

## 라이선스

MIT.
