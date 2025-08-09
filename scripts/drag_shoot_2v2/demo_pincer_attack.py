#!/usr/bin/env python3
"""
钳形夹击战术演示脚本 - 简化版本，展示新架构的功能
"""

import os
import sys
import logging
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


def setup_simple_logging():
    """设置简单日志（避免Unicode问题）"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'pincer_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        ]
    )


def demo_tactical_framework():
    """演示战术框架"""
    print("=" * 60)
    print("钳形夹击战术框架演示")
    print("=" * 60)
    
    try:
        # 1. 演示战术工厂
        print("\n1. 战术工厂演示:")
        from tactical_factory import get_tactical_factory, TacticalType
        
        factory = get_tactical_factory()
        available_tactics = factory.get_available_tactics()
        print(f"   可用战术: {[t.value for t in available_tactics]}")
        
        # 2. 演示钳形夹击任务创建
        print("\n2. 钳形夹击任务创建:")
        config = {
            'scenario': 'demo_pincer',
            'pincer_config': {
                'crank_angle': 45.0,
                'max_off_boresight': 60.0,
                'formation_spacing': 8000,
            }
        }
        
        pincer_task = factory.create_tactical_task(TacticalType.PINCER_ATTACK, config)
        print(f"   任务类型: {pincer_task.get_tactical_type().value}")
        print(f"   Crank角度: {pincer_task.pincer_config['crank_angle']}度")
        print(f"   编队间距: {pincer_task.pincer_config['formation_spacing']/1000}km")
        
        # 3. 演示拖曳射击任务创建（重构版本）
        print("\n3. 拖曳射击任务创建（重构版本）:")
        if os.path.exists("drag_shoot_tactical_task_refactored.py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "drag_shoot_refactored", 
                "drag_shoot_tactical_task_refactored.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            DragShootTacticalTask = module.DragShootTacticalTask
            
            drag_config = {
                'scenario': 'demo_drag_shoot',
                'wingman_delay': {
                    'TR_DOR_delay': 4000,
                    'DOR_DR_delay': 8000,
                }
            }
            
            drag_task = DragShootTacticalTask(drag_config)
            print(f"   任务类型: {drag_task.get_tactical_type().value}")
            print(f"   僚机延迟: TR-DOR={drag_task.drag_shoot_config['wingman_delay']['TR_DOR_delay']/1000}km")
        else:
            print("   重构文件不存在，跳过演示")
        
        # 4. 演示战术信息
        print("\n4. 战术信息对比:")
        for tactical_type in available_tactics:
            info = factory.get_tactical_info(tactical_type)
            print(f"\n   {tactical_type.value}:")
            print(f"     阶段: {info.get('phases', [])}")
            print(f"     特性: {info.get('key_features', [])}")
            print(f"     场景: {info.get('suitable_scenarios', [])}")
        
        # 5. 演示阶段管理
        print("\n5. 阶段管理演示:")
        from base_tactical_task import TacticalPhase
        
        phases = [phase.value for phase in TacticalPhase]
        print(f"   通用阶段: {phases}")
        print(f"   距离配置: {pincer_task.tactical_distances}")
        
        print("\n=" * 60)
        print("演示完成！架构设计成功。")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"演示失败: {e}")
        logging.error(f"演示失败: {e}")
        return False


def demo_pincer_phases():
    """演示钳形夹击阶段"""
    print("\n钳形夹击战术阶段详解:")
    print("-" * 40)
    
    phases = [
        ("NTL-MELD", "钳形展开", "双机向两侧执行Crank机动，形成钳形包夹态势"),
        ("MELD-MTR", "钳形收拢", "双机回正指向敌机，准备发射"),
        ("MTR-TR", "分层攻击", "长机优先发射，僚机保持滞后"),
        ("TR-DOR", "脱离机动", "双机执行内侧Short Skate机动"),
        ("DOR-DR", "返航阶段", "双机返航脱离")
    ]
    
    for i, (phase, name, description) in enumerate(phases, 1):
        print(f"{i}. {phase} - {name}")
        print(f"   {description}")
        print()


def demo_architecture_benefits():
    """演示架构优势"""
    print("\n新架构优势:")
    print("-" * 40)
    
    benefits = [
        "模块化设计: 每种战术独立封装，便于维护",
        "组件复用: 最大化复用现有基础设施",
        "策略模式: 支持动态切换不同战术类型",
        "扩展性强: 为未来添加更多战术类型预留接口",
        "统一接口: 所有战术使用相同的基础框架",
        "配置灵活: 每种战术可以有独立的参数配置"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")


def main():
    """主函数"""
    setup_simple_logging()
    
    try:
        # 演示战术框架
        success = demo_tactical_framework()
        
        if success:
            # 演示钳形夹击阶段
            demo_pincer_phases()
            
            # 演示架构优势
            demo_architecture_benefits()
            
            print("\n下一步建议:")
            print("1. 运行完整的钳形夹击仿真: python run_pincer_attack_simulation.py")
            print("2. 对比拖曳射击和钳形夹击的性能差异")
            print("3. 根据需要调整战术参数")
            print("4. 扩展更多战术类型")
        
    except Exception as e:
        print(f"演示异常: {e}")
        logging.error(f"演示异常: {e}")


if __name__ == "__main__":
    main()
