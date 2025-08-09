#!/usr/bin/env python3
"""
战术工厂 - 用于创建和管理不同类型的战术任务
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

from base_tactical_task import BaseTacticalTask, TacticalType


class TacticalFactory:
    """
    战术工厂类
    
    负责：
    1. 创建不同类型的战术任务
    2. 管理战术任务的生命周期
    3. 提供战术切换功能
    4. 验证战术配置
    """
    
    def __init__(self):
        """初始化战术工厂"""
        self._registered_tactics = {}
        self._active_task = None
        self._task_history = []
        
        # 注册可用的战术类型
        self._register_default_tactics()
        
        logging.info("战术工厂初始化完成")
    
    def _register_default_tactics(self):
        """注册默认的战术类型"""
        try:
            # 注册拖曳射击战术
            from drag_shoot_tactical_task import DragShootTacticalTask
            self._registered_tactics[TacticalType.DRAG_SHOOT] = DragShootTacticalTask
            logging.info("拖曳射击战术已注册")

        except ImportError as e:
            logging.warning(f"拖曳射击战术注册失败: {e}")

        try:
            # 注册钳形夹击战术
            from pincer_attack_tactical_task import PincerAttackTacticalTask
            self._registered_tactics[TacticalType.PINCER_ATTACK] = PincerAttackTacticalTask
            logging.info("钳形夹击战术已注册")

        except ImportError as e:
            logging.warning(f"钳形夹击战术注册失败: {e}")
    
    def register_tactical_task(self, tactical_type: TacticalType, task_class: type):
        """
        注册新的战术任务类型
        
        Args:
            tactical_type: 战术类型
            task_class: 战术任务类
        """
        if not issubclass(task_class, BaseTacticalTask):
            raise ValueError(f"战术任务类必须继承自BaseTacticalTask: {task_class}")
        
        self._registered_tactics[tactical_type] = task_class
        logging.info(f"战术类型 {tactical_type.value} 已注册")
    
    def create_tactical_task(self, tactical_type: TacticalType, config: Dict[str, Any]) -> BaseTacticalTask:
        """
        创建战术任务实例
        
        Args:
            tactical_type: 战术类型
            config: 配置参数
            
        Returns:
            战术任务实例
        """
        if tactical_type not in self._registered_tactics:
            available_types = list(self._registered_tactics.keys())
            raise ValueError(f"未注册的战术类型: {tactical_type}. 可用类型: {available_types}")
        
        task_class = self._registered_tactics[tactical_type]
        
        try:
            # 验证配置
            validated_config = self._validate_config(tactical_type, config)
            
            # 创建任务实例
            task_instance = task_class(validated_config)
            
            # 记录创建历史
            self._task_history.append({
                'tactical_type': tactical_type,
                'config': validated_config,
                'instance': task_instance
            })
            
            logging.info(f"战术任务创建成功: {tactical_type.value}")
            return task_instance

        except Exception as e:
            logging.error(f"战术任务创建失败 {tactical_type.value}: {e}")
            raise
    
    def _validate_config(self, tactical_type: TacticalType, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证战术配置
        
        Args:
            tactical_type: 战术类型
            config: 原始配置
            
        Returns:
            验证后的配置
        """
        validated_config = config.copy()
        
        # 通用配置验证
        if 'scenario' not in validated_config:
            validated_config['scenario'] = 'default_2v2'
        
        # 特定战术的配置验证
        if tactical_type == TacticalType.DRAG_SHOOT:
            validated_config = self._validate_drag_shoot_config(validated_config)
        elif tactical_type == TacticalType.PINCER_ATTACK:
            validated_config = self._validate_pincer_attack_config(validated_config)
        
        return validated_config
    
    def _validate_drag_shoot_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证拖曳射击配置"""
        # 设置拖曳射击特有的默认值
        defaults = {
            'friendly_missile_cooldown': 2.0,
            'enemy_missile_cooldown': 10.0,
            'wingman_delay': {
                'TR_DOR_delay': 4000,
                'DOR_DR_delay': 8000,
            }
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _validate_pincer_attack_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证钳形夹击配置"""
        # 设置钳形夹击特有的默认值
        defaults = {
            'pincer_config': {
                'crank_angle': 45.0,
                'max_off_boresight': 60.0,
                'formation_spacing': 8000,
                'leader_priority': True,
                'wingman_delay': 5.0,
            }
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def switch_tactical_task(self, new_tactical_type: TacticalType, config: Dict[str, Any]) -> BaseTacticalTask:
        """
        切换战术任务
        
        Args:
            new_tactical_type: 新的战术类型
            config: 新的配置
            
        Returns:
            新的战术任务实例
        """
        # 保存当前任务状态
        if self._active_task:
            logging.info(f"🔄 从 {self._active_task.tactical_type.value} 切换到 {new_tactical_type.value}")
        
        # 创建新任务
        new_task = self.create_tactical_task(new_tactical_type, config)
        
        # 更新活跃任务
        self._active_task = new_task
        
        return new_task
    
    def get_available_tactics(self) -> list:
        """获取可用的战术类型列表"""
        return list(self._registered_tactics.keys())
    
    def get_tactical_info(self, tactical_type: TacticalType) -> Dict[str, Any]:
        """
        获取战术信息
        
        Args:
            tactical_type: 战术类型
            
        Returns:
            战术信息字典
        """
        if tactical_type not in self._registered_tactics:
            return {}
        
        task_class = self._registered_tactics[tactical_type]
        
        info = {
            'name': tactical_type.value,
            'class': task_class.__name__,
            'description': task_class.__doc__ or "无描述",
        }
        
        # 添加特定战术的信息
        if tactical_type == TacticalType.DRAG_SHOOT:
            info.update({
                'phases': ['NLT_MELD', 'MELD_MTR', 'MTR_TR', 'TR_DOR', 'DOR_DR'],
                'key_features': ['拖曳射击', '僚机滞后', 'Short Skate脱离'],
                'suitable_scenarios': ['2v2对抗', 'BVR作战']
            })
        elif tactical_type == TacticalType.PINCER_ATTACK:
            info.update({
                'phases': ['钳形展开', '钳形收拢', '分层攻击', '脱离机动', '返航'],
                'key_features': ['双机钳形', '分层攻击', '内侧脱离'],
                'suitable_scenarios': ['2v2对抗', '包夹战术']
            })
        
        return info
    
    def get_task_history(self) -> list:
        """获取任务创建历史"""
        return self._task_history.copy()
    
    def cleanup(self):
        """清理资源"""
        self._active_task = None
        self._task_history.clear()
        logging.info("🧹 战术工厂资源已清理")


# 全局战术工厂实例
_tactical_factory = None

def get_tactical_factory() -> TacticalFactory:
    """获取全局战术工厂实例"""
    global _tactical_factory
    if _tactical_factory is None:
        _tactical_factory = TacticalFactory()
    return _tactical_factory

def create_tactical_task(tactical_type: TacticalType, config: Dict[str, Any]) -> BaseTacticalTask:
    """便捷函数：创建战术任务"""
    factory = get_tactical_factory()
    return factory.create_tactical_task(tactical_type, config)

def get_available_tactics() -> list:
    """便捷函数：获取可用战术类型"""
    factory = get_tactical_factory()
    return factory.get_available_tactics()

def get_tactical_info(tactical_type: TacticalType) -> Dict[str, Any]:
    """便捷函数：获取战术信息"""
    factory = get_tactical_factory()
    return factory.get_tactical_info(tactical_type)
