"""
战术系统配置文件
"""

# ==================== 系统配置 ====================
SYSTEM_CONFIG = {
    'version': '1.0.0',
    'debug_mode': False,
    'log_level': 'INFO',
}

# ==================== 默认意图配置 ====================
DEFAULT_INTENT = 'conservative_clear'  # 'aggressive_clear' / 'conservative_clear' / 'defensive'

# ==================== 控制距离配置 ====================
# 可以在这里调整控制距离，单位：km
CUSTOM_CONTROL_RANGES = {
    'NLT': 120.0,
    'MELD': 100.0,
    'MTR': 80.0,
    'LR': 78.0,
    'TR': 75.0,
    'DOR': 70.0,
    'DR': 65.0,
    'MTR_PRIME': 55.0,
    'LR_PRIME': 53.0,
    'TR_PRIME': 50.0,
    'MAR': 40.0,
}

# ==================== 威胁值配置 ====================
# 威胁值计算权重
THREAT_WEIGHTS = {
    'distance': 0.3,
    'angle': 0.3,
    'altitude': 0.2,
    'speed': 0.2,
}

# 威胁值阈值
CUSTOM_THREAT_THRESHOLDS = {
    'HIGH': 0.8,
    'MEDIUM': 0.5,
    'LOW': 0.3,
}

# 撤退条件
RETREAT_CONDITIONS = {
    'total_threat_threshold': 0.8,
    'single_threat_threshold': 0.8,
    'single_threat_count': 2,  # 需要≥2个单项威胁值超过阈值
}

# 规避条件
EVASION_CONDITIONS = {
    'single_threat_threshold': 0.8,
    'single_threat_count_min': 1,
    'single_threat_count_max': 2,
}

# ==================== 战术选择配置 ====================
# 拖曳射击
DRAG_SHOOT_CONFIG = {
    'high_threat_threshold': 0.8,
    'low_threat_threshold': 0.5,
    'drag_offset': 2000.0,  # 拖曳机前出距离 (m)
    'shooter_offset': -2000.0,  # 射击机后退距离 (m)
}

# 钳形攻势
PINCER_ATTACK_CONFIG = {
    'angle_threat_threshold': 0.8,
    'pincer_angle': 45.0,  # 钳形角度 (度)
}

# 上下夹击
HIGH_LOW_ATTACK_CONFIG = {
    'altitude_threat_threshold': 0.8,
    'altitude_separation': 1000.0,  # 高度间隔 (m)
}

# 前后攻击
SEQUENTIAL_ATTACK_CONFIG = {
    'threat_sum_threshold': 1.2,
    'threat_ratio_threshold': 1.5,
    'front_offset': 3000.0,  # 前机前出距离 (m)
    'rear_offset': -3000.0,  # 后机后退距离 (m)
}

# 并排射击
SIDE_BY_SIDE_CONFIG = {
    'threat_sum_threshold': 0.8,
    'threat_ratio_threshold': 1.5,
    'lateral_separation': 1000.0,  # 横向间隔 (m)
}

# ==================== 机动参数配置 ====================
# LR节点微偏置
CRANK_CONFIG = {
    'adjustment_range': (-15.0, 15.0),  # 调整角度范围 (度)
    'radar_effective_angle': 30.0,  # 雷达有效照射角度 (度)
}

# DR节点时间窗口
DR_CONFIG = {
    'time_window': 20.0,  # DR节点决策时间窗口 (秒)
}

# 第二轮进攻
SECOND_ATTACK_CONFIG = {
    'time_window': 30.0,  # 第二轮进攻时间窗口 (秒)
    'enabled': True,  # 是否启用第二轮进攻
}

# ==================== 物理参数配置 ====================
PHYSICS_CONFIG = {
    'max_speed': 600.0,  # 最大速度 (m/s)
    'max_altitude': 15000.0,  # 最大高度 (m)
    'max_turn_rate': 20.0,  # 最大转弯率 (度/秒)
    'min_altitude': 1000.0,  # 最小高度 (m)
}

# ==================== 敌方意图识别配置 ====================
INTENT_RECOGNITION_CONFIG = {
    'attack_aspect_angle': 45.0,  # 进攻判断角度阈值 (度)
    'attack_closing_speed': 100.0,  # 进攻判断接近速度阈值 (m/s)
    'disengage_aspect_angle': 135.0,  # 逃逸判断角度阈值 (度)
    'disengage_speed_ratio': 0.9,  # 逃逸判断速度比例
}

# ==================== 目标分配配置 ====================
TARGET_ASSIGNMENT_CONFIG = {
    'method': 'fixed',  # 'fixed' / 'dynamic'
    'lead_to_lead': True,  # 长机打敌长机
    'wingman_to_wingman': True,  # 僚机打敌僚机
}

# ==================== 协同决策配置 ====================
COORDINATION_CONFIG = {
    'enabled': True,  # 是否启用协同决策
    'retreat_separation': 1000.0,  # 撤退时的横向间隔 (m)
}

# ==================== 特殊情况处理配置 ====================
SPECIAL_CASES_CONFIG = {
    'tactical_turn_timeout': 30.0,  # 战术回转超时时间 (秒)
    'missile_count_max': 4,  # 最大导弹数量
    'enable_special_case_handling': True,  # 是否启用特殊情况处理
}

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    'log_decisions': True,  # 是否记录决策
    'log_threats': True,  # 是否记录威胁值
    'log_maneuvers': True,  # 是否记录机动
    'log_file': None,  # 日志文件路径，None表示不写入文件
}

# ==================== 可视化配置 ====================
VISUALIZATION_CONFIG = {
    'enabled': False,  # 是否启用可视化
    'update_interval': 1.0,  # 可视化更新间隔 (秒)
    'show_threats': True,  # 是否显示威胁值
    'show_trajectories': True,  # 是否显示轨迹
}


def get_config(config_name: str = None):
    """
    获取配置
    
    Args:
        config_name: 配置名称，None表示获取所有配置
    
    Returns:
        配置字典
    """
    all_configs = {
        'system': SYSTEM_CONFIG,
        'intent': DEFAULT_INTENT,
        'control_ranges': CUSTOM_CONTROL_RANGES,
        'threat_weights': THREAT_WEIGHTS,
        'threat_thresholds': CUSTOM_THREAT_THRESHOLDS,
        'retreat_conditions': RETREAT_CONDITIONS,
        'evasion_conditions': EVASION_CONDITIONS,
        'drag_shoot': DRAG_SHOOT_CONFIG,
        'pincer_attack': PINCER_ATTACK_CONFIG,
        'high_low_attack': HIGH_LOW_ATTACK_CONFIG,
        'sequential_attack': SEQUENTIAL_ATTACK_CONFIG,
        'side_by_side': SIDE_BY_SIDE_CONFIG,
        'crank': CRANK_CONFIG,
        'dr': DR_CONFIG,
        'second_attack': SECOND_ATTACK_CONFIG,
        'physics': PHYSICS_CONFIG,
        'intent_recognition': INTENT_RECOGNITION_CONFIG,
        'target_assignment': TARGET_ASSIGNMENT_CONFIG,
        'coordination': COORDINATION_CONFIG,
        'special_cases': SPECIAL_CASES_CONFIG,
        'logging': LOGGING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
    }
    
    if config_name is None:
        return all_configs
    
    return all_configs.get(config_name, {})


def update_config(config_name: str, config_dict: dict):
    """
    更新配置
    
    Args:
        config_name: 配置名称
        config_dict: 新的配置字典
    """
    global SYSTEM_CONFIG, DEFAULT_INTENT, CUSTOM_CONTROL_RANGES
    global THREAT_WEIGHTS, CUSTOM_THREAT_THRESHOLDS
    global RETREAT_CONDITIONS, EVASION_CONDITIONS
    global DRAG_SHOOT_CONFIG, PINCER_ATTACK_CONFIG
    global HIGH_LOW_ATTACK_CONFIG, SEQUENTIAL_ATTACK_CONFIG
    global SIDE_BY_SIDE_CONFIG, CRANK_CONFIG, DR_CONFIG
    global SECOND_ATTACK_CONFIG, PHYSICS_CONFIG
    global INTENT_RECOGNITION_CONFIG, TARGET_ASSIGNMENT_CONFIG
    global COORDINATION_CONFIG, SPECIAL_CASES_CONFIG
    global LOGGING_CONFIG, VISUALIZATION_CONFIG
    
    config_map = {
        'system': SYSTEM_CONFIG,
        'intent': DEFAULT_INTENT,
        'control_ranges': CUSTOM_CONTROL_RANGES,
        'threat_weights': THREAT_WEIGHTS,
        'threat_thresholds': CUSTOM_THREAT_THRESHOLDS,
        'retreat_conditions': RETREAT_CONDITIONS,
        'evasion_conditions': EVASION_CONDITIONS,
        'drag_shoot': DRAG_SHOOT_CONFIG,
        'pincer_attack': PINCER_ATTACK_CONFIG,
        'high_low_attack': HIGH_LOW_ATTACK_CONFIG,
        'sequential_attack': SEQUENTIAL_ATTACK_CONFIG,
        'side_by_side': SIDE_BY_SIDE_CONFIG,
        'crank': CRANK_CONFIG,
        'dr': DR_CONFIG,
        'second_attack': SECOND_ATTACK_CONFIG,
        'physics': PHYSICS_CONFIG,
        'intent_recognition': INTENT_RECOGNITION_CONFIG,
        'target_assignment': TARGET_ASSIGNMENT_CONFIG,
        'coordination': COORDINATION_CONFIG,
        'special_cases': SPECIAL_CASES_CONFIG,
        'logging': LOGGING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
    }
    
    if config_name in config_map:
        if isinstance(config_map[config_name], dict):
            config_map[config_name].update(config_dict)
        else:
            # 对于非字典类型的配置（如DEFAULT_INTENT），直接赋值
            globals()[config_name.upper()] = config_dict
