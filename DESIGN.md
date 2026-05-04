# Fourier Transform Visualizer — Design Document

> 연속 시간 신호의 푸리에 변환을 3차원 복소공간에서 인터랙티브하게 탐험하는 교육용 도구.
> v0.1, 작성일 2026-04-30. 마지막 개정 2026-05-03. 목표 릴리스: 다음 주 월요일까지(약 4일).

---

## 1. Motivation & Goals

### 1.1 왜 만드는가

3Blue1Brown의 "winding machine" 시각화는 푸리에 변환을 직관적으로 보여주는 강력한 도구이지만, 본질적으로 2D 평면 시각화이다. 그래서 다음과 같은 핵심 개념이 그림으로 직접 보이지 않는다:

- **푸리에 변환 결과가 왜 복소수인가** — center of mass의 (x, y) 좌표를 복소수의 (Re, Im)로 해석한다는 추가 설명이 필요함.
- **음의 주파수의 의미** — 회전 방향이 반대인 헬릭스라는 것이 평면에서는 표현이 어렵다.
- **직교성(orthogonality)** — "회전이 멈춘 성분만 적분에서 살아남는다"는 것이 평면 시각화에서는 "원점 부근으로 모인다"라는 간접적 신호로만 드러난다.

이 도구는 **시간축(또는 주파수축)을 한 축, 복소평면(Re, Im)을 다른 두 축**으로 두는 3D 시각화를 메인으로 하여, 위 개념들이 그림에서 직접 읽히도록 한다.

### 1.2 핵심 학습 목표 (Learning Objectives)

도구를 한 시간 가지고 논 학습자가 다음을 *시각적으로* 설명할 수 있어야 한다.

- **L1**: 푸리에 변환은 신호 `x(t)`를 `e^(-jωt)`로 "감은(wound)" 헬릭스를 적분한 것이다.
- **L2**: ω가 신호의 한 성분 `ω₀`와 일치하면, `x(t)·e^(-jωt)`의 그 성분은 회전이 멈추고, 적분에 살아남는다.
- **L3**: ω ≠ ω₀인 다른 성분들은 여전히 회전 중이므로, 충분히 긴 구간에서 적분하면 벡터합이 0에 가까워진다 (직교성).
- **L4**: 따라서 `X(jω)`는 "ω에 대해 정지한 성분의 누적 벡터합"이며, 자연스럽게 복소수 값을 가진다.
- **L5** *(v0.2)*: 푸리에 역변환은 모든 ω에 대한 `X(jω)·e^(jωt)`를 누적합한 것이며, 양의 ω 성분과 그 켤레인 음의 ω 성분이 짝을 지어 실수 신호 `x(t)`를 복원한다.

### 1.3 비목표 (Non-Goals)

이 도구는 다음을 *하지 않는다*. 명시적으로 빼두는 것이 scope creep을 막는다.

- 이산 시간 신호 처리 (DFT, FFT, 윈도우 함수 등) — 이건 별도의 도구 영역이다.
- 실시간 오디오 입력/스펙트로그램.
- 2D 푸리에 변환 (이미지 처리).
- 라플라스/Z 변환 등 일반화.
- 수치 정확도 벤치마크 — 우리는 교육용이지 신호 처리 라이브러리가 아니다.

---

## 2. Conceptual Model — 무엇을 그리는가

이 절은 문서의 심장이다. 모든 API는 여기서 정의한 여섯 가지 시각적 요소를 표현하는 수단이다.

### 2.1 여섯 가지 시각적 요소

#### (a) Signal Curve `x(t)`

시간축 위에 그려진 실수값(또는 복소값) 곡선. 사용자가 정의하는 입력. 2D 또는 3D(복소 신호일 경우 Re/Im 분리)로 표시.

#### (b) Winding Helix `e^(-jωt)`

3D 공간에서 시간 축을 따라 복소평면을 회전하는 단위 헬릭스. 각속도는 `-ω`. ω가 클수록 빽빽하게 감긴다. ω가 음수면 반대 방향으로 감긴다.

#### (c) Wound Signal `x(t)·e^(-jωt)`

(a)의 envelope 안에 (b)의 carrier가 들어간 3D 곡선. `x(t)`가 헬릭스의 반지름을 변조한다. 푸리에 적분의 피적분 함수 그 자체.

#### (d) Forward Running Integral — *Mode A의 메인 디시*

Wound signal `(c)`를 시간에 대해 누적 적분한 결과를 **3D 누적 벡터**로 본다.

좌표계는 (c)와 동일한 시간 축 + 복소평면 3D 공간. 각 시점 t에서 원점에서 출발해 누적 적분값으로 향하는 벡터를 그리고, 그 벡터의 끝점이 그리는 궤적도 함께 표시한다.

학습 메시지:

- ω가 신호의 한 성분 ω₀와 일치하면 → 정지 성분이 살아남아 누적 벡터가 한 방향으로 일관되게 자라남.
- ω ≠ ω₀이면 → 회전 성분만 남아 누적 벡터가 작은 원을 그리며 제자리를 맴돔. 한 주기를 돌 때마다 원점으로 돌아옴.

이 그림이 **"적분이 곧 내적이고, 직교성이 곧 cancel-out"**이라는 메시지를 가장 정직하게 전달한다. (별도의 정규화나 center of mass 트릭 없이 raw 누적합 그대로 보여줌.)

#### (e) Spectrum `X(jω)` — *v0.2 (Mode B에서 활성화)*

ω를 ω_min에서 ω_max까지 훑으면서, 각 ω에 대한 `(d)`의 최종값(즉 Mode A의 적분 구간 [t_min, t_max] 끝에서의 누적 벡터값)을 점으로 찍어 그린 곡선. 양/음의 ω 모두 포함.

표시 옵션은 magnitude/phase 또는 Re/Im 두 가지 (Open Question §8).

#### (f) Inverse Running Integral — *v0.2 (Mode B에서 활성화)*

푸리에 역변환을 ω에 대한 누적합으로 시각화한 단일 패널.

특정 시점 `t = t_fix`를 슬라이더로 고정한 뒤, 사전에 계산해둔 스펙트럼 `X(jω)`를 ω가 작은 값(보통 음의 큰 값)부터 점차 큰 값까지 훑으면서, 각 ω에 대한 복소 벡터 `(1/2π) · X(jω)·e^(jωt_fix) · dω`를 2D 복소평면 위에 *벡터 머리에 머리를 잇는 방식*으로 누적한다.

- 각 ω의 기여는 작은 화살표 한 개.
- 화살표들이 첫 번째 끝에 두 번째가 이어붙고, 그 끝에 세 번째가 이어붙는 식으로 누적.
- ω = ω_max에서 도달한 누적 끝점이 곧 `x(t_fix)`. 실수 신호이면 결국 실수축 위의 점.
- 양의 ω 성분과 그 켤레인 음의 ω 성분은 서로 짝을 이뤄 허수부가 cancel out, 실수부만 남음. 이게 시각적으로 직접 보임.

학습 메시지: **푸리에 역변환도 결국 회전 벡터의 누적합이며, 실수성(realness)은 켤레쌍이 만들어내는 자연스러운 결과**.

### 2.2 두 가지 동작 모드

**Mode A — Time-sweep (v0.1 구현 대상)**
- ω 고정. 시간 t를 t_min에서 t_max까지 증가시킨다.
- (a)~(d)가 모두 t의 함수로 애니메이션된다.
- 학습 메시지: "이 ω에서 이 신호를 감았을 때, 적분이 어떻게 누적되는가."

**Mode B — Frequency-sweep (v0.2)**
- 적분 구간 [t_min, t_max] 고정. ω를 ω_min에서 ω_max까지 (양/음 모두) 훑는다.
- (e) 스펙트럼: 매 ω에 대한 최종 누적값을 한 점으로 찍어 X(jω) 곡선을 만든다.
- (f) 역변환 누적: 슬라이더로 고정한 t_fix에 대해, ω를 늘려가며 `X(jω)·e^(jωt_fix)` 화살표를 2D 복소평면에 머리잇기로 누적해 x(t_fix)에 수렴하는 과정을 본다.
- 학습 메시지 두 가지: "어떤 ω에서 누적 벡터가 크게 살아남는가" + "그 X(jω)들을 다시 합치면 원래 신호가 어떻게 복원되는가".

v0.1에서는 Mode A에만 집중하되, API와 UI 레이아웃은 Mode B((e), (f))를 끼워넣을 자리를 명시적으로 남겨둔다 (placeholder 패널).

### 2.3 사용자가 직접 만지는 것

Mode A의 풀 인터랙티브 정의 (v0.1에서는 후속 작업으로 미룸 — §6 참조):

- **재생/일시정지/스크럽**: 시간 t의 자동 진행 또는 슬라이더 직접 조작
- **ω 슬라이더**: winding angular frequency. 신호의 성분 주파수와 일치시켜보는 게 메인 액티비티
- **신호 파라미터**: 표준 신호의 진폭/주파수/위상 등 슬라이더
- **3D 시점 회전**: 마우스 드래그로 카메라 각도 변경 (Plotly 기본 지원)
- **연동된 패널**: 위 시각 요소들이 같은 t·ω 상태를 공유하며 동시에 갱신

Mode B 진입 시 추가:

- **t_fix 슬라이더**: (f)에서 역변환을 보고 싶은 시점.
- **ω 범위 슬라이더**: ω_min, ω_max. 양/음 모두 다룸.
- **(f) 누적 진행 슬라이더**: ω를 어디까지 누적해서 보여줄지.

---

## 3. Mathematical Conventions

### 3.1 푸리에 변환 정의

**기본 표기는 angular frequency ω(rad/s) 기반**으로 한다. 공학·신호처리 표준이고, 시각화에서도 winding 헬릭스의 각속도가 곧 ω라 직관과 코드가 일치한다.

```
순방향:   X(jω) = ∫ x(t)·e^(-jωt) dt
역방향:   x(t)  = (1/2π) ∫ X(jω)·e^(jωt) dω
```

함수 인자, 슬라이더, 라벨, 클래스 속성 모두 `omega`로 통일. Hz가 필요하면 사용자가 직접 `omega = 2π·f`로 변환해 입력 — API에 freq/omega 두 갈래를 두지 않는다.

### 3.2 적분 근사

연속 시간이라고 했지만 컴퓨터에서 적분은 결국 수치 근사다.

- **유한 구간** [t_min, t_max]에서 균등 샘플링된 N개 점에 대해 사다리꼴 합산.
- 사용자가 t_min, t_max, N을 슬라이더 또는 파라미터로 조절 가능.
- 비감쇠 신호(cos, sin)에 대해 "구간 길이가 늘어나면 ω₀에서의 응답이 점점 뾰족해진다"를 보여줄 수 있는 게 의도된 교육 효과다.

Mode B의 역변환 적분도 동일하게 사다리꼴 누적합으로 근사한다. ω 도메인에서 균등 샘플링.

### 3.3 비감쇠 신호의 한계

`cos(ω₀t)`처럼 ∫|x(t)|dt = ∞인 신호는 엄밀하게는 푸리에 변환이 디랙 델타 함수다. 우리 도구는 그런 분포 이론적 결과를 보여주지 않고, **유한 구간에서의 sinc-like 근사**를 보여준다. 이건 버그가 아니라 의도된 교육적 단순화다. 문서에서 이 점을 사용자에게 명시한다.

---

## 4. API Layers — 3-Layer Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Layer 3: app/                                             │
│   FourierExplorer (인터랙티브 컴포저)                     │
│   ipywidgets 슬라이더 + 패널들 연동                        │
│   v0.1: 후속 작업, v0.2 본격화                             │
└──────────────────────────────────────────────────────────┘
                       ↑ uses
┌──────────────────────────────────────────────────────────┐
│ Layer 2: viz/                                             │
│   SignalPlot · WindingHelixPlot · WoundSignalPlot ·       │
│   ForwardIntegralPlot                                     │
│   SpectrumPlot · InverseAccumulationPlot (v0.2)           │
│   추상 인터페이스 + Plotly 백엔드 구현                     │
└──────────────────────────────────────────────────────────┘
                       ↑ uses
┌──────────────────────────────────────────────────────────┐
│ Layer 1: core/                                            │
│   Signal · ComplexSignal · WoundSignal · FourierAnalyzer  │
│   순수 수학. 렌더러 의존성 0.                              │
└──────────────────────────────────────────────────────────┘
```

각 레이어는 아래 레이어만 알고, 위 레이어는 모른다. 이 분리 덕분에 다음이 가능하다.

- Layer 1만으로도 단위 테스트 가능 (수치 정확도 검증).
- 나중에 Plotly가 아닌 다른 백엔드(Three.js, Manim 등)로 교체 시 Layer 2만 수정.
- Layer 3는 "어떤 슬라이더가 어떤 플롯을 갱신하는가"만 다루면 됨.

### 4.1 Layer 1: core/

순수 수치 계산. numpy 외 의존성 없음.

#### 4.1.1 Signal 클래스 계층

```
Signal (실수 신호, 기본·추상-아님)
 └── ComplexSignal (복소수 신호)
      └── WoundSignal (x(t)·e^(-jωt) — 평가 가능 + winding 메타 + 분석 결과)
```

설계 의도:

- **`Signal`**은 실수 시간 함수의 표현. `signals.cosine(omega=...)`, `gaussian(...)`, `rect(...)` 등이 모두 `Signal` 인스턴스. `.sample(t)`는 실수 배열 반환.
- **`ComplexSignal`**은 `Signal`의 자식이지만 평가 결과가 복소수. `signals.complex_exp(omega=...)`가 대표 예. `.sample(t)`는 복소 배열 반환. 모든 `Signal` 자리에 `ComplexSignal`이 들어갈 수 있다 (LSP 만족).
- **`WoundSignal`**은 `ComplexSignal`의 자식. `x(t)·e^(-jωt)`는 그 자체가 평가 가능한 복소 함수이므로 `ComplexSignal`임이 자연스럽다. 추가로 winding 메타데이터 (`omega`)와 분석 결과 (사전 계산된 `t`, `wound_values`, `running_integral`, `final_integral`) 필드를 들고 있어, viz 레이어가 다시 계산하지 않고 바로 그릴 수 있다.

#### 4.1.2 합성 시 타입 승격 규칙

| 좌측 | 연산 | 우측 | 결과 타입 |
|------|------|------|-----------|
| Signal | `+ -` `*` | Signal | Signal |
| Signal | `+ -` `*` | ComplexSignal | ComplexSignal |
| ComplexSignal | `+ -` `*` | (any Signal) | ComplexSignal |
| Signal | `+ -` `*` | 실수 스칼라 | Signal |
| Signal | `+ -` `*` | 복소 스칼라 | ComplexSignal |

`-x`(단항)는 입력 타입을 보존. WoundSignal은 합성 결과로는 만들어지지 않는다 (오직 `FourierAnalyzer.wound_at_omega`로만 생성).

#### 4.1.3 주요 인터페이스

- **`signals` 모듈**: `cosine(omega, ...)`, `sine(omega, ...)`, `complex_exp(omega, ...)`, `gaussian(...)`, `rect(...)`, `step(...)`, `decaying_exp(...)`. `complex_exp`만 ComplexSignal 반환, 나머지는 Signal 반환.
- **`FourierAnalyzer`**: Signal과 적분 파라미터(t_min, t_max, n_samples)를 받아 분석을 제공. 핵심 메서드는 `wound_at_omega(ω)`, 한 winding angular frequency에 대해 `WoundSignal`을 반환. `wound_at_freq` 같은 Hz 보조는 두지 않는다 (§3.1).
- **`FourierAnalyzer.spectrum(omega_min, omega_max, n_omega)`** (v0.2): ω 범위의 X(jω) 배열 반환.
- **`FourierAnalyzer.inverse_accumulation(t_fix, omega_min, omega_max, n_omega)`** (v0.2): (f)의 데이터 — ω별 역변환 화살표 시퀀스와 누적 벡터.

### 4.2 Layer 2: viz/

Plotly 기반. 각 플롯 클래스는 단일 책임:

- **`SignalPlot`**: (a) 시간 도메인 곡선
- **`WindingHelixPlot`**: (b) 단위 헬릭스
- **`WoundSignalPlot`**: (c) 변조된 헬릭스
- **`ForwardIntegralPlot`**: (d) Wound signal의 3D 누적 적분 벡터 — *Mode A의 메인 디시*
- **`SpectrumPlot`**: (e) v0.1에선 placeholder, v0.2 활성화
- **`InverseAccumulationPlot`**: (f) v0.1에선 placeholder, v0.2 활성화

각 플롯은 동일한 인터페이스를 따른다: `update(wound, t_index)` 호출 시 자기 패널을 갱신.

### 4.3 Layer 3: app/

- **`FourierExplorer`**: ipywidgets로 슬라이더 묶음(t, ω, 신호 파라미터)을 만들고, 위 플롯들을 같은 상태에 묶음. v0.1에서는 후속 작업.

```python
from ftvis import signals, FourierAnalyzer

sig = signals.cosine(omega=2*np.pi) + signals.cosine(omega=3*np.pi)
analyzer = FourierAnalyzer(sig, t_min=0.0, t_max=10.0, n_samples=2000)
ws = analyzer.wound_at_omega(2*np.pi)  # 첫 성분으로 감기
# viz 레이어가 ws를 받아 4개 패널을 그림 (Day 2에서 구현)
```

---

## 5. Interaction Model — 사용자 시나리오

### 5.1 시나리오 1: "winding speed가 신호 주파수와 일치할 때"

1. 사용자가 `signals.cosine(omega=2*np.pi)`를 만든다.
2. `FourierExplorer(sig).show()`를 호출.
3. ω 슬라이더를 0부터 천천히 올린다.
4. ω = 2π 부근에서 (d) 누적 벡터가 일관된 방향으로 자라는 것을 본다.
5. ω를 2π에서 살짝 벗어나면, 누적 벡터가 작은 원을 그리며 제자리를 맴도는 것을 본다.

### 5.2 시나리오 2: "신호를 합성하면"

1. `signals.cosine(omega=2*np.pi) + signals.cosine(omega=3*np.pi)`로 합성 신호를 만든다.
2. ω 슬라이더를 훑으며 ω=2π와 ω=3π 두 곳에서 누적 벡터가 살아남는 것을 본다.
3. 그 사이 주파수에서는 두 성분 모두 회전 중이라 합이 작은 것을 확인.

### 5.3 시나리오 3: "감쇠 신호"

1. `signals.gaussian(sigma=1.0, center=5.0) * signals.cosine(omega=3*np.pi)`을 만든다.
2. 누적 벡터가 처음에는 자라다가, gaussian envelope가 0에 가까워지면 자라지 않는 것을 본다.
3. ω를 훑으며 ω=3π 근처에서 매끄러운 가우시안 모양 응답을 확인 (Mode B에서 더 명확히, v0.2).

### 5.4 시나리오 4: "역변환의 시각화" *(v0.2)*

1. 위 시나리오의 분석기로 `FourierAnalyzer.spectrum(...)`을 호출해 X(jω)를 얻는다.
2. (e) 패널에서 X(jω)의 형태를 본다.
3. (f) 패널에서 t_fix 슬라이더를 1.0으로 두고, ω 누적 슬라이더를 -∞ 쪽에서 시작.
4. ω를 0 부근으로 가져갈 때 화살표가 작게 누적되다가, ω가 신호의 성분 주파수에 가까워지면 큰 화살표들이 더해짐.
5. ω = +∞ 근처까지 가면 누적 끝점이 실수축 위의 한 점에 도달 — 이게 곧 `x(1.0)`.
6. 켤레 쌍이 어떻게 허수부를 cancel out하는지 t_fix를 바꿔가며 확인.

---

## 6. Roadmap (4 Days)

각 단계에서 **데모 가능한 산출물**이 있어야 한다는 원칙으로 잘랐다.

### Day 1 — core/ 골격 *(완료)*

- `Signal`, `signals` 팩토리 (cosine, sine, complex_exp, gaussian, rect, step, decaying_exp)
- `FourierAnalyzer.wound_at_omega(ω)`가 `WoundSignal`(wound + running_integral 포함)을 반환
- 단위 테스트 45개 통과

**Done definition** *(달성)*: `python -m pytest`가 통과하고, `cos(ω₀t)`에 대해 ω = ω₀일 때 누적 벡터 방향이 일정함을 자동 확인.

### Day 1.5 — 클래스 계층 리팩터 *(현재 진행)*

- `Signal` → `ComplexSignal` → `WoundSignal` 위계 도입
- 합성 시 타입 승격
- ω 기본화 (`wound_at_freq` 제거)
- 기존 테스트 유지 + 위계 관련 테스트 추가

### Day 2 — viz/ 정적 플롯

- Plotly로 (a)(b)(c)(d) 4개 플롯 클래스. (d)는 *3D 누적 벡터*로 그림 (center of mass 없음).
- 정적 스크린샷 4장 — README 후보 이미지
- (e) SpectrumPlot, (f) InverseAccumulationPlot은 placeholder 박스만

**Done definition**: 노트북에서 5줄로 4개 플롯을 띄울 수 있다. 각 플롯이 시각적으로 그럴듯하다.

### Day 3 — Mode A 인터랙티브 (옵션) 또는 데모 노트북 강화

원래 인터랙티브 앱(`FourierExplorer`)은 *후속 작업*으로 미뤘다. Day 3는 두 가지 옵션 중 택1:

- (옵션 A) ipywidgets 슬라이더 한두 개만 붙인 가벼운 인터랙티브 (ω 슬라이더 → (d) 갱신).
- (옵션 B) 인터랙티브를 빼고, 시나리오 1~3을 각각 정적 슬라이드 시퀀스로 보여주는 데모 노트북.

옵션 B가 더 안전. 옵션 A는 시간이 남으면.

### Day 4 — 정리, 문서, 깃허브

- README.md (정적 이미지 4~5장 포함)
- `requirements.txt`, `pyproject.toml`
- Open Questions(아래 §8)와 Mode B((e),(f)) 작업을 GitHub Issues로 정리
- 깃허브 푸시

**Done definition**: 처음 보는 사람이 README만 읽고 `pip install -e .` → 노트북 실행이 된다.

### 위험 요소와 완화책

- **3D 플롯의 성능**: Plotly의 3D 씬은 점이 많으면 느려진다. 누적 벡터 궤적은 N=500 정도로 시작하고, 필요하면 다운샘플.
- **하루 일정이 밀릴 경우**: Day 3을 옵션 B로 잡고 Day 4의 문서/깃허브 작업을 살린다.

---

## 7. Out of Scope (v0.1)

명시적으로 v0.1에서 빼는 것들:

- **Mode B ((e) 스펙트럼 + (f) 역변환 누적)** — v0.2 최우선 후보. v0.1에서는 viz 레이어에 placeholder만.
- **인터랙티브 앱 (`FourierExplorer`)** — 후속 작업. v0.1에서는 정적 데모 노트북만.
- **SymPy 심볼릭 입력 / closed-form FT 표시**
- **DFT/FFT 연결, 이산 신호**
- **마우스 드래그로 신호 직접 그리기**
- **여러 신호를 패널 위에 겹쳐 비교하기**
- **저장/내보내기 기능** (스크린샷, GIF 자동 생성 등)
- **Hz 단위 헬퍼 메서드 (`wound_at_freq` 등)** — 의도적으로 두지 않음 (§3.1)

---

## 8. Open Questions (결정 보류)

설계 단계에서 잘라낸 결정들. v0.1 구현 중 또는 v0.2 진입 시 다시 다룬다.

1. **(d) 누적 벡터의 정규화**: 현재 안은 "정규화 없이 raw 누적합". 하지만 구간 길이 T에 따라 크기가 달라진다. 이걸 (1/T)로 나눠 평균 위치로 보여주는 옵션을 토글로 노출할지?
2. **위상 표시**: 누적 벡터의 phase를 별도로 보여주는 미니 다이얼을 (d) 옆에 둘지?
3. **Re/Im vs magnitude/phase**: (e) 스펙트럼 표시를 Re-Im 좌표축으로 그릴지, 극좌표(반지름·각도)로 그릴지. 둘 다 토글?
4. **(f)의 ω 누적 순서**: ω_min → ω_max로 단조롭게 늘릴지, 아니면 |ω| 작은 것부터 큰 것 순으로 누적해 켤레쌍이 나란히 추가되게 할지? 후자가 "허수부가 cancel out 되는 모습"을 더 명료하게 보여줌.
5. **렌더러 백엔드 추상화 깊이**: Layer 2 인터페이스를 정말 백엔드 중립으로 설계할지, 아니면 Plotly에 결합된 채로 두고 추후 리팩터로 미룰지? (ADR 후보)
6. **합성 신호의 차수 제한**: `cos + cos + cos + ...`를 100개 더해도 동작은 하지만 시각화는 깨진다. UI에서 합성항 수에 제한을 둘지?
7. **WoundSignal의 `sample(t_new)` 의미**: WoundSignal은 사전 샘플링된 분석 결과를 들고 있다. 이걸 또 다른 t 배열로 다시 평가할 때(상속받은 `.sample()`), 닫힌 형태인 `x(t)·e^(-jωt)`를 다시 계산할지(자연스러운 정의), 아니면 원본 샘플에서 보간할지? 전자가 일관적.

---

## 9. Glossary

- **Signal `x(t)`**: 시간 도메인 입력 함수. 기본은 실수, 자식인 ComplexSignal은 복소.
- **Winding**: 신호를 `e^(-jωt)`와 곱해 복소평면을 따라 회전시키는 연산.
- **Wound signal**: `x(t)·e^(-jωt)`. 푸리에 적분의 피적분 함수. 코드상 `WoundSignal` 클래스 (ComplexSignal의 자식).
- **Forward running integral**: 시점 t에서의 부분 적분 `∫₀ᵗ x(τ)·e^(-jωτ) dτ`를 복소 벡터로 본 것. (d).
- **Inverse running accumulation**: 고정된 t에서, ω를 누적해가며 `(1/2π)·X(jω)·e^(jωt)·dω`를 머리잇기로 더한 것. (f).
- **Mode A / Mode B**: 시간 t를 진행시키는 모드(ω 고정) / 주파수 ω를 훑는 모드(t 고정 또는 적분 구간 고정).
- **omega (ω)**: 각주파수 (rad/s). 이 도구의 기본 단위.
