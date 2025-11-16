"""
战术阶段和常用类型定义
独立文件，避免循环导入
"""
from enum import Enum


class TacticalPhase(Enum):
    """战术阶段枚举 - 范围阶段"""
    NLT_MELD = "NLT_MELD"        # 120-100km: 搜索目标&编队
    MELD_MTR = "MELD_MTR"        # 100-80km: 雷达融合&调整编队
    MTR_LR = "MTR_LR"            # 80-78km: 进入发射区
    LR_TR = "LR_TR"              # 78-75km: 发射&中制导
    TR_DOR = "TR_DOR"            # 75-70km: 中制导结束&规避
    DOR_DR = "DOR_DR"            # 70-65km: 规避机动
    DR_MAR = "DR_MAR"            # 65-40km: 脱离/重新进攻
    BEYOND_MAR = "BEYOND_MAR"    # <40km: WVR或完全脱离


def get_target_with_fallback(agent_id: str, env) -> str:
    """
    获取目标飞机ID（带fallback机制）
    优先使用固定目标，如果目标已被击毁则寻找其他目标
    
    Args:
        agent_id: 当前飞机ID
        env: 环境对象
        
    Returns:
        目标飞机ID，如果没有目标返回None
    """
    # 确定敌方队伍
    enemy_team = 'B' if agent_id.startswith('A') else 'A'
    
    # 固定目标映射
    fixed_targets = {
        'A0100': 'B0100',  # 我方长机 vs 敌方长机
        'A0200': 'B0200',  # 我方僚机 vs 敌方僚机
        'B0100': 'A0100',  # 敌方长机 vs 我方长机
        'B0200': 'A0200',  # 敌方僚机 vs 我方僚机
    }
    
    # 尝试获取固定目标
    fixed_target = fixed_targets.get(agent_id)
    if fixed_target and fixed_target in env.agents and env.agents[fixed_target].is_alive:
        return fixed_target
    
    # Fallback: 寻找任何存活的敌方飞机
    for target_id in env.agents.keys():
        if target_id.startswith(enemy_team) and env.agents[target_id].is_alive:
            return target_id
    
    return None
