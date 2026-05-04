"""
ftvis — Fourier Transform Visualizer.

탑레벨에서 가장 자주 쓰일 이름들만 재노출. 자세한 시그니처는 각 서브모듈의
.pyi를 참고.
"""

from ftvis.core import (
    ComplexSignal,
    FourierAnalyzer,
    Signal,
    WoundSignal,
    signals,
)

__version__ = "0.1.0.dev0"

# core는 numpy만 의존하므로 항상 import. viz는 plotly가 필요한데, 사용자가 viz를
# 쓰지 않을 수도 있으므로 lazy하게 처리한다. 접근 시점에 plotly가 없으면
# 친절한 ImportError를 던진다.

_VIZ_NAMES = (
    "ForwardIntegralPlot",
    "InverseAccumulationPlot",
    "SignalPlot",
    "SpectrumPlot",
    "WindingHelixPlot",
    "WoundSignalPlot",
)


def __getattr__(name):
    if name in _VIZ_NAMES:
        try:
            from ftvis import viz as _viz
        except ImportError as e:
            raise ImportError(
                f"ftvis.{name}을(를) 사용하려면 plotly가 필요합니다.\n"
                f"  PyCharm 터미널 또는 venv 활성 후:  pip install plotly\n"
                f"원본 에러: {e}"
            ) from e
        return getattr(_viz, name)
    raise AttributeError(f"module 'ftvis' has no attribute {name!r}")


__all__ = [
    "Signal",
    "ComplexSignal",
    "WoundSignal",
    "FourierAnalyzer",
    "signals",
    "__version__",
    *_VIZ_NAMES,
]
