"""
协同探测V4 - 算法模块
支持简化版和专业版算法切换
"""
from .estimators import SimpleEstimator, IMMUKFEstimator
from .scan_calculators import ArctanCalculator, CovarianceEllipseCalc
from .fusion import WeightedAverageFusion, T2TFusion
from .predictors import KinematicPredictor, ParticleFilterPredictor
from .search_optimizers import UniformSearchOptimizer, InfoEntropyOptimizer
from .probability_map import SimpleProbabilityMap, BayesianProbabilityMap

__all__ = [
    'SimpleEstimator', 'IMMUKFEstimator',
    'ArctanCalculator', 'CovarianceEllipseCalc',
    'WeightedAverageFusion', 'T2TFusion',
    'KinematicPredictor', 'ParticleFilterPredictor',
    'UniformSearchOptimizer', 'InfoEntropyOptimizer',
    'SimpleProbabilityMap', 'BayesianProbabilityMap'
]
