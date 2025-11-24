"""
战术决策系统测试
"""
import numpy as np
from core import ThreatEvaluator, TacticSelector, TacticalDecisionManager
from intent import OurIntentSystem, EnemyIntentRecognizer
from situation import SituationEvaluator


def test_threat_calculator():
    """测试威胁值计算"""
    print("=" * 50)
    print("测试威胁值计算")
    print("=" * 50)
    
    calc = ThreatEvaluator()
    
    # 我机状态
    my_state = {
        'position': np.array([0.0, 0.0, 5000.0]),
        'velocity': np.array([200.0, 0.0, 0.0]),
        'heading': 0.0
    }
    
    # 敌机状态（距离80km，正对我机）
    enemy_state = {
        'position': np.array([80000.0, 0.0, 5000.0]),
        'velocity': np.array([-200.0, 0.0, 0.0]),
        'heading': 180.0
    }
    
    threat = calc.calculate_threat(my_state, enemy_state)
    
    print(f"距离威胁值: {threat['distance']:.3f}")
    print(f"角度威胁值: {threat['angle']:.3f}")
    print(f"高度威胁值: {threat['altitude']:.3f}")
    print(f"速度威胁值: {threat['speed']:.3f}")
    print(f"总威胁值: {threat['total']:.3f}")
    print()


def test_enemy_intent():
    """测试敌方意图识别"""
    print("=" * 50)
    print("测试敌方意图识别")
    print("=" * 50)
    
    recognizer = EnemyIntentRecognizer()
    
    # 我机状态
    my_state = {
        'position': np.array([0.0, 0.0, 5000.0]),
        'velocity': np.array([200.0, 0.0, 0.0]),
        'heading': 0.0
    }
    
    # 测试1：敌机进攻（正对我机）
    enemy_attack = {
        'position': np.array([80000.0, 0.0, 5000.0]),
        'velocity': np.array([-200.0, 0.0, 0.0]),
        'heading': 180.0
    }
    intent1 = recognizer.recognize(my_state, enemy_attack)
    print(f"敌机进攻意图: {intent1}")
    
    # 测试2：敌机逃逸（背离我机）
    enemy_disengage = {
        'position': np.array([80000.0, 0.0, 5000.0]),
        'velocity': np.array([200.0, 0.0, 0.0]),
        'heading': 0.0
    }
    intent2 = recognizer.recognize(my_state, enemy_disengage)
    print(f"敌机逃逸意图: {intent2}")
    print()


def test_tactic_selector():
    """测试战术选择"""
    print("=" * 50)
    print("测试战术选择")
    print("=" * 50)
    
    selector = TacticSelector()
    
    # 测试1：拖曳射击（长机威胁大，僚机威胁小）
    blue_threats1 = {
        'lead': {'total': 0.85, 'angle': 0.6, 'altitude': 0.3},
        'wingman': {'total': 0.4, 'angle': 0.3, 'altitude': 0.2}
    }
    red_threats1 = {
        'enemy1_to_lead': {'total': 0.7},
        'enemy2_to_lead': {'total': 0.5}
    }
    tactic1, roles1 = selector.select_tactic(blue_threats1, red_threats1)
    print(f"测试1 - 拖曳射击: 战术={tactic1}, 角色={roles1}")
    
    # 测试2：钳形攻势（角度威胁大）
    blue_threats2 = {
        'lead': {'total': 0.6, 'angle': 0.85, 'altitude': 0.3},
        'wingman': {'total': 0.5, 'angle': 0.7, 'altitude': 0.2}
    }
    red_threats2 = {
        'enemy1_to_lead': {'total': 0.6},
        'enemy2_to_lead': {'total': 0.5}
    }
    tactic2, roles2 = selector.select_tactic(blue_threats2, red_threats2)
    print(f"测试2 - 钳形攻势: 战术={tactic2}, 角色={roles2}")
    
    # 测试3：并排射击（威胁小）
    blue_threats3 = {
        'lead': {'total': 0.3, 'angle': 0.3, 'altitude': 0.2},
        'wingman': {'total': 0.3, 'angle': 0.3, 'altitude': 0.2}
    }
    red_threats3 = {
        'enemy1_to_lead': {'total': 0.3},
        'enemy2_to_lead': {'total': 0.3}
    }
    tactic3, roles3 = selector.select_tactic(blue_threats3, red_threats3)
    print(f"测试3 - 并排射击: 战术={tactic3}, 角色={roles3}")
    print()


def test_decision_manager():
    """测试顶层决策管理器"""
    print("=" * 50)
    print("测试顶层决策管理器")
    print("=" * 50)
    
    manager = TacticalDecisionManager('conservative_clear')
    
    # 我方编队
    blue_formation = {
        'lead': {
            'position': np.array([0.0, 0.0, 5000.0]),
            'velocity': np.array([200.0, 0.0, 0.0]),
            'heading': 0.0
        },
        'wingman': {
            'position': np.array([0.0, 1000.0, 5000.0]),
            'velocity': np.array([200.0, 0.0, 0.0]),
            'heading': 0.0
        }
    }
    
    # 敌方编队（距离100km，触发MELD节点）
    red_formation = {
        'enemy1': {
            'position': np.array([100000.0, 0.0, 5000.0]),
            'velocity': np.array([-200.0, 0.0, 0.0]),
            'heading': 180.0
        },
        'enemy2': {
            'position': np.array([100000.0, 1000.0, 5000.0]),
            'velocity': np.array([-200.0, 0.0, 0.0]),
            'heading': 180.0
        }
    }
    
    # 执行决策
    decisions = manager.update(blue_formation, red_formation, 0.0)
    
    print(f"当前阶段: {decisions['phase']}")
    print(f"当前战术: {decisions['tactic']}")
    if decisions['lead']:
        print(f"长机决策: {decisions['lead'].get('action', 'N/A')}")
    if decisions['wingman']:
        print(f"僚机决策: {decisions['wingman'].get('action', 'N/A')}")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("战术决策系统测试")
    print("=" * 50 + "\n")
    
    test_threat_calculator()
    test_enemy_intent()
    test_tactic_selector()
    test_decision_manager()
    
    print("=" * 50)
    print("所有测试完成")
    print("=" * 50)


if __name__ == '__main__':
    run_all_tests()
