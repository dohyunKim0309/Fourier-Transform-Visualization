"""
ftvis — Fourier Transform Visualizer.

탑레벨에서 가장 자주 쓰일 이름들만 재노출. 자세한 시그니처는 각 서브모듈의
.pyi를 참고.
"""

from ftvis.core import (
    Signal as Signal,
    WoundSignal as WoundSignal,
    FourierAnalyzer as FourierAnalyzer,
    signals as signals,
)
from ftvis.viz import (
    SignalPlot as SignalPlot,
    WindingHelixPlot as WindingHelixPlot,
    WoundSignalPlot as WoundSignalPlot,
    RunningIntegralPlot as RunningIntegralPlot,
    SpectrumPlot as SpectrumPlot,
)
from ftvis.app import (
    FourierExplorer as FourierExplorer,
    ExplorerConfig as ExplorerConfig,
)

__version__: str
