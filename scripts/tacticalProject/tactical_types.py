"""
战术阶段和常用类型定义
独立文件，避免循环导入
"""
from enum import Enum


class TacticalPhase(Enum):
    """战术阶段枚举 - 范围阶段"""
    BEYOND_NLT = "BEYOND_NLT"    # >120km: 意图识别/抵近
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
    统一委托给目标分配模块，避免不同文件各自维护一套目标逻辑。
    
    Args:
        agent_id: 当前飞机ID
        env: 环境对象
        
    Returns:
        目标飞机ID，如果没有目标返回None
    """
    from core.target_assignment import get_target_with_fallback as _shared_get_target_with_fallback

    return _shared_get_target_with_fallback(agent_id, env)
