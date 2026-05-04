"""
ftvis.app — Layer 3: Interactive composer.

ipywidgets 슬라이더로 신호 파라미터·ω·t를 묶고, viz 레이어의 5개 플롯을
같은 상태에 동기화시킨다. Jupyter Notebook에서 한 줄로 띄우는 것이 목표.

설계 원칙
---------
- FourierExplorer는 "위젯 + 플롯 묶음을 들고, 슬라이더 변경 콜백을 푸리에 분석으로
  라우팅하는 컨트롤러"이다. 자체 수치 계산 로직은 없다 — core를 호출할 뿐.
- 모드 A에 대해서만 v0.1 구현.
- 슬라이더 우선순위: t > ω > 신호 파라미터 > 카메라 회전.
  일정이 밀리면 아래 항목부터 잘라낸다.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipywidgets as widgets

from ftvis.core import FourierAnalyzer, Signal


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExplorerConfig:
    """FourierExplorer의 외부 노출 파라미터.

    여기 모인 필드는 모두 슬라이더 가능 여부와 별개로, '시작값' 또는
    '슬라이더 범위'를 결정한다.
    """

    # 적분 구간 (sec)
    t_min: float = 0.0
    t_max: float = 10.0
    n_samples: int = 2000

    # winding angular frequency 슬라이더 (rad/s). 코어와 통일된 단위.
    omega_min: float = 0.0
    omega_max: float = 4 * 3.141592653589793  # 4π
    omega_step: float = 3.141592653589793 / 16  # π/16
    omega_init: float = 0.0

    # 시간 커서 슬라이더
    t_cursor_step: float = 0.01

    # 자동 재생
    autoplay: bool = False
    autoplay_fps: int = 30


# ─────────────────────────────────────────────────────────────────────────────
# FourierExplorer — 메인 진입점
# ─────────────────────────────────────────────────────────────────────────────

class FourierExplorer:
    """
    한 신호에 대한 Mode A 인터랙티브 탐험기.

    Examples
    --------
    >>> import numpy as np
    >>> from ftvis import FourierExplorer, signals
    >>> sig = signals.cosine(omega=2*np.pi) + signals.cosine(omega=3*np.pi)
    >>> explorer = FourierExplorer(sig)
    >>> explorer.show()  # Jupyter에 슬라이더 + 4개 플롯 패널이 뜬다

    구성된 후에도 외부에서 상태를 바꿀 수 있다:

    >>> explorer.set_omega(2*np.pi)  # ω 슬라이더 동기화 (rad/s)
    >>> explorer.set_t_index(120)    # 시간 커서 동기화
    """

    signal: Signal
    config: ExplorerConfig
    analyzer: FourierAnalyzer

    def __init__(
        self,
        signal: Signal,
        *,
        config: ExplorerConfig | None = None,
    ) -> None:
        """
        signal : 분석할 신호.
        config : ExplorerConfig. None이면 기본값 사용.
        """
        ...

    # ── 표시 ──────────────────────────────────────────────────────────────
    def show(self) -> widgets.Widget:
        """Jupyter cell에서 호출. 슬라이더 + 플롯이 묶인 상위 위젯을 반환."""
        ...

    # ── 외부 제어 (자동화·테스트용) ─────────────────────────────────────────
    def set_omega(self, omega: float) -> None:
        """현재 winding angular frequency를 설정 (rad/s). 슬라이더와 모든 플롯이 갱신됨."""
        ...

    def set_t_index(self, t_index: int) -> None:
        """현재 시간 커서 위치를 인덱스로 설정."""
        ...

    def set_t(self, t: float) -> None:
        """현재 시간 커서를 실제 t 값으로 설정 (가장 가까운 인덱스로 스냅)."""
        ...

    def play(self) -> None:
        """t 커서를 자동 진행 시작."""
        ...

    def pause(self) -> None:
        """자동 진행 중지."""
        ...

    # ── 내부 콜백 (subclass 또는 이벤트 디버깅용) ────────────────────────────
    def _on_omega_change(self, change: dict) -> None: ...
    def _on_t_change(self, change: dict) -> None: ...
    def _refresh_all_plots(self) -> None:
        """모든 플롯에 현재 (omega, t_index) 상태를 반영."""
        ...
