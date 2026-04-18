"""
协同探测算法配置 - 算法开关控制
默认使用简化版算法，可切换为专业版

用法：
    from .algorithm_config import ALGO_CONFIG
    
    if ALGO_CONFIG.use_kalman_filter:
        estimator.update_kalman(...)
    else:
        estimator.update_simple(...)
"""
from dataclasses import dataclass


@dataclass
class AlgorithmConfig:
    """算法配置类"""
    
    # === 算法1：目标状态估计 ===
    use_kalman_filter: bool = False  # True=卡尔曼滤波, False=直接位置
    kalman_process_noise: float = 0.1  # 过程噪声
    kalman_measurement_noise: float = 2.5  # 测量噪声(km)
    
    # === 算法2：任务分配 ===
    use_hungarian_algorithm: bool = False  # True=匈牙利算法, False=贪婪分配
    
    # === 算法3：航迹融合 ===
    use_federated_filter: bool = False  # True=联邦滤波, False=加权平均
    awacs_weight: float = 0.3  # 预警机权重
    radar_weight: float = 0.7  # 雷达权重
    fusion_threshold_km: float = 30.0  # 关联阈值
    
    # === 算法4：跟踪质量评估 ===
    use_advanced_quality: bool = False  # True=多因子评估, False=简单评估
    min_quality_for_launch: float = 0.6  # 发射所需最低质量
    
    # === 算法5：目标丢失搜索 ===
    use_prediction_search: bool = False  # True=预测搜索, False=简单超时
    target_lost_timeout: float = 20.0  # 丢失超时(秒)
    prediction_horizon: float = 30.0  # 预测时间窗(秒)


# 全局配置实例（启用专业算法）
ALGO_CONFIG = AlgorithmConfig(
    use_kalman_filter=True,           # 卡尔曼滤波状态估计
    use_hungarian_algorithm=True,     # 匈牙利算法任务分配
    use_federated_filter=True,        # 联邦滤波航迹融合
    use_advanced_quality=True,        # 多因子质量评估
    use_prediction_search=True        # 预测搜索
)


def enable_advanced_algorithms():
    """启用所有专业算法"""
    global ALGO_CONFIG
    ALGO_CONFIG = AlgorithmConfig(
        use_kalman_filter=True,
        use_hungarian_algorithm=True,
        use_federated_filter=True,
        use_advanced_quality=True,
        use_prediction_search=True
    )


def disable_advanced_algorithms():
    """禁用所有专业算法（使用默认简化版）"""
    global ALGO_CONFIG
    ALGO_CONFIG = AlgorithmConfig()
