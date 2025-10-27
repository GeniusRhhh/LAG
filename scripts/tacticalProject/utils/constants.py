"""
战术系统常量定义
"""

# ==================== 控制距离定义 (单位: km) ====================
CONTROL_RANGES = {
    'NLT': 120.0,      # No Later Than - 不迟于此距离
    'MELD': 100.0,     # Merge Entry Launch Decision - 合并进入发射决策
    'MTR': 80.0,       # Missile Target Range - 导弹目标距离
    'LR': 78.0,        # Launch Range - 发射距离
    'TR': 75.0,        # Turn Range - 转向距离
    'DOR': 70.0,       # Decision on Re-attack - 重新攻击决策
    'DR': 65.0,        # Disengagement Range - 脱离距离
    'MTR_PRIME': 55.0, # 第二轮MTR
    'LR_PRIME': 53.0,  # 第二轮LR
    'TR_PRIME': 50.0,  # 第二轮TR
    'MAR': 40.0,       # Minimum Abort Range - 最小中止距离
}

# 控制距离顺序（第一轮）
FIRST_PHASE_RANGES = ['NLT', 'MELD', 'MTR', 'LR', 'TR', 'DOR', 'DR']

# 控制距离顺序（第二轮）
SECOND_PHASE_RANGES = ['MTR_PRIME', 'LR_PRIME', 'TR_PRIME', 'MAR']

# ==================== 威胁值阈值 ====================
THREAT_THRESHOLDS = {
    'HIGH': 0.8,       # 高威胁阈值
    'MEDIUM': 0.5,     # 中威胁阈值
    'LOW': 0.3,        # 低威胁阈值
}

# 撤退条件：总威胁值阈值
RETREAT_TOTAL_THREAT = 0.8

# 撤退条件：单项威胁值阈值（需要≥2个超过此值）
RETREAT_SINGLE_THREAT = 0.8

# 规避条件：单项威胁值阈值（需要1-2个超过此值）
EVASION_SINGLE_THREAT = 0.8

# ==================== 战术选择阈值 ====================
# 拖曳射击：某机总威胁>0.8，另一机≤0.5
DRAG_SHOOT_HIGH_THREAT = 0.8
DRAG_SHOOT_LOW_THREAT = 0.5

# 钳形攻势：某机角度威胁>0.8
PINCER_ANGLE_THREAT = 0.8

# 上下夹击：某机高度威胁>0.8
HIGH_LOW_ALTITUDE_THREAT = 0.8

# 前后攻击：敌方威胁和>1.2 且 max/min<1.5
SEQUENTIAL_THREAT_SUM = 1.2
SEQUENTIAL_THREAT_RATIO = 1.5

# 并排射击：敌方威胁和<0.8 且 max/min<1.5
SIDE_BY_SIDE_THREAT_SUM = 0.8
SIDE_BY_SIDE_THREAT_RATIO = 1.5

# ==================== 战术编号 ====================
TACTICS = {
    'DRAG_SHOOT': 1,        # 拖曳射击
    'PINCER_ATTACK': 2,     # 钳形攻势
    'HIGH_LOW_ATTACK': 3,   # 上下夹击
    'SEQUENTIAL_ATTACK': 4, # 前后攻击
    'SIDE_BY_SIDE': 5,      # 并排射击
    'TACTICAL_EVASION': 6,  # 战术规避
    'TACTICAL_TURN': 7,     # 战术回转
}

# ==================== 我方意图类型 ====================
OUR_INTENT = {
    'AGGRESSIVE_CLEAR': 'aggressive_clear',     # 激进肃清
    'CONSERVATIVE_CLEAR': 'conservative_clear', # 保守肃清
    'DEFENSIVE': 'defensive',                   # 防御意图
}

# ==================== 敌方意图类型 ====================
ENEMY_INTENT = {
    'ATTACK': 'attack',         # 进攻
    'NEUTRAL': 'neutral',       # 中立
    'DISENGAGE': 'disengage',   # 逃逸
}

# ==================== 决策结果类型 ====================
DECISION_TYPE = {
    'CONTINUE': 'continue',     # 继续执行核心任务
    'EVASION': 'evasion',       # 规避
    'RETREAT': 'retreat',       # 撤退
}

# ==================== 机动参数 ====================
# LR节点微偏置角度范围
CRANK_ADJUSTMENT_RANGE = (-15.0, 15.0)  # 度

# 雷达照射有效角度范围
RADAR_EFFECTIVE_ANGLE = 30.0  # 度

# DR节点时间窗口
DR_TIME_WINDOW = 20.0  # 秒

# 第二轮进攻时间窗口
SECOND_ATTACK_TIME_WINDOW = 30.0  # 秒

# ==================== 机动类型 ====================
MANEUVER_TYPES = {
    'MAINTAIN_HEADING': 'maintain_heading',     # 保持航向
    'CRANK': 'crank',                           # Crank机动
    'TACTICAL_CRANK': 'tactical_crank',         # 战术Crank
    'BEAM': 'beam',                             # Beam机动
    'SHORT_SKATE': 'short_skate',               # Short Skate
    'NOTCH_BACK': 'notch_back',                 # Notch Back
    'TURN_180': 'turn_180',                     # 180度回转
}

# ==================== 飞机角色 ====================
AIRCRAFT_ROLE = {
    'LEAD': 'lead',         # 长机
    'WINGMAN': 'wingman',   # 僚机
}

# ==================== 战术角色 ====================
TACTICAL_ROLE = {
    'DRAG': 'drag',         # 拖曳机
    'SHOOTER': 'shooter',   # 射击机
    'FRONT': 'front',       # 前机
    'REAR': 'rear',         # 后机
    'HIGH': 'high',         # 高机
    'LOW': 'low',           # 低机
    'LEFT': 'left',         # 左机
    'RIGHT': 'right',       # 右机
}

# ==================== 态势评估权重 ====================
# 不同阶段的权重配置
SITUATION_WEIGHTS = {
    'EARLY': {  # 初期（NLT/MELD）
        'angle': 0.3,
        'distance': 0.3,
        'altitude': 0.2,
        'speed': 0.2,
    },
    'MID': {  # 中期（MTR）
        'angle': 0.25,
        'distance': 0.25,
        'altitude': 0.3,
        'speed': 0.2,
    },
    'LATE': {  # 后期（LR/TR/DOR/DR/MAR）
        'angle': 0.25,
        'distance': 0.25,
        'altitude': 0.2,
        'speed': 0.3,
    },
}

# ==================== 物理常量 ====================
MAX_SPEED = 600.0  # 最大速度 m/s
MAX_ALTITUDE = 15000.0  # 最大高度 m
MAX_TURN_RATE = 20.0  # 最大转弯率 度/秒

# ==================== 敌方意图识别阈值 ====================
# 进攻判断
ATTACK_ASPECT_ANGLE = 45.0  # 度，敌机航向指向我机
ATTACK_CLOSING_SPEED = 100.0  # m/s，敌机接近速度

# 逃逸判断
DISENGAGE_ASPECT_ANGLE = 135.0  # 度，敌机航向背离我机
DISENGAGE_SPEED_RATIO = 0.9  # 敌机速度 > 最大速度 * 0.9

# ==================== 目标分配 ====================
# 简化版：固定分配
TARGET_ASSIGNMENT = {
    'LEAD_TO_LEAD': True,  # 长机打敌长机
    'WINGMAN_TO_WINGMAN': True,  # 僚机打敌僚机
}
