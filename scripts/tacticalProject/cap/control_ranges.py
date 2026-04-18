"""
战术控制距离配置 - V5优化版
基于协同探测V5方案定义的时间线节点

注意：
- 200km（雷达探测范围）定义在 cap_radar.py
- 400km（预警机探测范围）定义在 awacs_source.py
- 本模块仅定义战术动作触发节点
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ControlRanges:
    """战术控制距离(km) - 不可变配置
    
    V5流程：
    编队前出(400~200km) → 协同探测(≤200km) → 动态交战 → 紧急规避
    
    战术节点（基于雷达探测信息触发）：
    NLT(180) → MELD(150) → MTR(120) → LR(100) → TR(80) → DOR(60) → DR(50) → MAR(35)
    
    二次进攻(DR到MAR之间)：
    MTR'(45) → LR'(42) → TR'(38)
    """
    # === 交战前阶段 ===
    NLT: float = 180.0     # No Later Than - 意图识别
    MELD: float = 150.0    # 目标跟踪距离 - 雷达融合
    
    # === 第一轮交战 ===
    MTR: float = 120.0     # 最小跟踪距离 - 协同跟踪
    LR: float = 100.0      # 发射距离 - 导弹发射
    TR: float = 80.0       # 转换距离 - 接力制导
    DOR: float = 60.0      # 期望脱离距离 - 防御机动
    DR: float = 50.0       # 决断距离 - 决断点
    
    # === 第二轮交战（DR到MAR之间）===
    MTR_PRIME: float = 45.0   # 第二轮跟踪距离
    LR_PRIME: float = 42.0    # 第二轮发射距离
    TR_PRIME: float = 38.0    # 第二轮转换距离
    
    # === 紧急阈值 ===
    MAR: float = 35.0      # 最小脱离距离 - 紧急规避
    
    # === 动态压缩参数 ===
    COMPRESSION_BASE: float = 180.0  # 压缩基准距离（NLT理论值）
    
    def get_all(self) -> Dict[str, float]:
        """获取所有控制距离"""
        return {k: getattr(self, k) for k in 
                ['NLT', 'MELD', 'MTR', 'LR', 'TR', 'DOR', 'DR', 'MAR',
                 'MTR_PRIME', 'LR_PRIME', 'TR_PRIME']}
    
    def get_compressed_ranges(self, actual_distance: float) -> Dict[str, float]:
        """获取动态压缩后的节点距离
        
        V5动态压缩算法：
        - D_real ≥ 180km: Ratio=1（无压缩）
        - 35km ≤ D_real < 180km: Ratio=D_real/180（比例压缩）
        - D_real < 35km: 返回None（紧急规避）
        
        Args:
            actual_distance: 我方基准机与敌方最近机的实际距离(km)
            
        Returns:
            压缩后的节点距离字典，若需紧急规避则返回空字典
        """
        if actual_distance < self.MAR:
            return {}  # 紧急规避，不计算节点
        
        ratio = min(1.0, actual_distance / self.COMPRESSION_BASE)
        
        return {
            'NLT': self.NLT * ratio,
            'MELD': self.MELD * ratio,
            'MTR': self.MTR * ratio,
            'LR': self.LR * ratio,
            'TR': self.TR * ratio,
            'DOR': self.DOR * ratio,
            'DR': self.DR * ratio,
            'MAR': self.MAR * ratio,
            'MTR_PRIME': self.MTR_PRIME * ratio,
            'LR_PRIME': self.LR_PRIME * ratio,
            'TR_PRIME': self.TR_PRIME * ratio,
        }
    
    def get_current_node(self, distance: float, ranges_dict: Dict[str, float] = None) -> str:
        """根据距离获取当前所处节点
        
        Args:
            distance: 当前距离
            ranges_dict: (可选) 动态压缩后的节点距离字典。若提供则使用动态值，否则使用理论值。
        """
        # 获取判定阈值 (优先使用动态值)
        def get_val(key):
            return ranges_dict[key] if ranges_dict and key in ranges_dict else getattr(self, key)
            
        if distance > get_val('NLT'):
            return "BEYOND_NLT"
        elif distance > get_val('MELD'):
            return "NLT_MELD"
        elif distance > get_val('MTR'):
            return "MELD_MTR"
        elif distance > get_val('LR'):
            return "MTR_LR"
        elif distance > get_val('TR'):
            return "LR_TR"
        elif distance > get_val('DOR'):
            return "TR_DOR"
        elif distance > get_val('DR'):
            return "DOR_DR"
        elif distance > get_val('MAR'):
            return "DR_MAR"
        else:
            return "BELOW_MAR"
    
    def is_second_attack_zone(self, distance: float) -> bool:
        """判断是否在二次进攻区域(DR到MAR之间)"""
        return self.MAR < distance <= self.DR


# 默认配置实例
DEFAULT_RANGES = ControlRanges()

