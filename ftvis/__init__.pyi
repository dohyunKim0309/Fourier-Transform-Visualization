"""
ftvis — Fourier Transform Visualizer.

탑레벨에서 가장 자주 쓰일 이름들만 재노출. 자세한 시그니처는 각 서브모듈의
.pyi를 참고.
"""

from ftvis.core import (
    ComplexSignal as ComplexSignal,
    FourierAnalyzer as FourierAnalyzer,
    Signal as Signal,
    WoundSignal as WoundSignal,
    signals as signals,
)
from ftvis.viz import (
    ForwardIntegralPlot as ForwardIntegralPlot,
    InverseAccumulationPlot as InverseAccumulationPlot,
    SignalPlot as SignalPlot,
    SpectrumPlot as SpectrumPlot,
    WindingHelixPlot as WindingHelixPlot,
    WoundSignalPlot as WoundSignalPlot,
)
from ftvis.app import (
    ExplorerConfig as ExplorerConfig,
    FourierExplorer as FourierExplorer,
)

__version__: str
