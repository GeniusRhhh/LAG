"""
控制距离管理器
"""
from typing import Dict, Optional, List
from utils.constants import CONTROL_RANGES, FIRST_PHASE_RANGES, SECOND_PHASE_RANGES
from utils.geometry import calculate_distance


class ControlRangeManager:
    """控制距离管理器"""
    
    def __init__(self):
        """初始化控制距离管理器"""
        self.current_phase = 'first'  # 'first' or 'second'
        self.passed_ranges = {
            'lead': set(),
            'wingman': set()
        }
        self.last_distances = {
            'lead': {},
            'wingman': {}
        }
    
    def check_range_trigger(self, aircraft_id: str, my_pos, target_pos) -> Optional[str]:
        """
        检查是否触发控制距离节点
        
        Args:
            aircraft_id: 飞机ID ('lead' or 'wingman')
            my_pos: 我机位置
            target_pos: 目标位置
        
        Returns:
            触发的控制距离节点名称，如果未触发则返回None
        """
        # 计算当前距离
        current_distance = calculate_distance(my_pos, target_pos)
        
        # 获取当前阶段的控制距离列表
        if self.current_phase == 'first':
            ranges = FIRST_PHASE_RANGES
        else:
            ranges = SECOND_PHASE_RANGES
        
        # 检查每个控制距离
        for range_name in ranges:
            # 跳过已经通过的节点
            if range_name in self.passed_ranges[aircraft_id]:
                continue
            
            range_value = CONTROL_RANGES[range_name]
            
            # 检查是否穿越控制距离阈值
            if self._check_crossing(aircraft_id, range_name, current_distance, range_value):
                # 标记为已通过
                self.passed_ranges[aircraft_id].add(range_name)
                
                # 根据飞机角色添加后缀
                if aircraft_id == 'lead':
                    node_name = range_name + '1'
                else:
                    node_name = range_name + '2'
                
                # 如果是第二阶段，添加'标记
                if self.current_phase == 'second' and range_name in ['MTR', 'LR', 'TR']:
                    node_name = node_name.replace('1', "1'").replace('2', "2'")
                
                return node_name
        
        return None
    
    def _check_crossing(self, aircraft_id: str, range_name: str, 
                       current_distance: float, range_value: float) -> bool:
        """
        检查是否穿越控制距离阈值
        
        Args:
            aircraft_id: 飞机ID
            range_name: 控制距离名称
            current_distance: 当前距离 (km)
            range_value: 控制距离阈值 (km)
        
        Returns:
            True表示穿越阈值
        """
        # 获取上一次的距离
        last_distance = self.last_distances[aircraft_id].get(range_name, float('inf'))
        
        # 更新距离记录
        self.last_distances[aircraft_id][range_name] = current_distance
        
        # 检查是否从大于阈值变为小于阈值（穿越）
        if last_distance > range_value and current_distance <= range_value:
            return True
        
        return False
    
    def enter_second_phase(self):
        """进入第二轮进攻阶段"""
        self.current_phase = 'second'
        # 清空第二阶段的已通过节点
        for aircraft_id in ['lead', 'wingman']:
            # 只保留第一阶段的节点
            self.passed_ranges[aircraft_id] = {
                r for r in self.passed_ranges[aircraft_id] 
                if r in FIRST_PHASE_RANGES
            }
    
    def reset(self):
        """重置控制距离管理器"""
        self.current_phase = 'first'
        self.passed_ranges = {
            'lead': set(),
            'wingman': set()
        }
        self.last_distances = {
            'lead': {},
            'wingman': {}
        }
    
    def get_next_range(self, aircraft_id: str) -> Optional[str]:
        """
        获取下一个控制距离节点
        
        Args:
            aircraft_id: 飞机ID
        
        Returns:
            下一个控制距离节点名称
        """
        if self.current_phase == 'first':
            ranges = FIRST_PHASE_RANGES
        else:
            ranges = SECOND_PHASE_RANGES
        
        for range_name in ranges:
            if range_name not in self.passed_ranges[aircraft_id]:
                return range_name
        
        return None
    
    def has_passed(self, aircraft_id: str, range_name: str) -> bool:
        """
        检查是否已经通过某个控制距离节点
        
        Args:
            aircraft_id: 飞机ID
            range_name: 控制距离名称
        
        Returns:
            True表示已通过
        """
        return range_name in self.passed_ranges[aircraft_id]
