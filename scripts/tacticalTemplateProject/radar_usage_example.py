#!/usr/bin/env python3
"""
雷达系统功能使用示例
展示如何使用文档3.7节要求的RWR和导弹发射集成检查功能
"""

import logging
from radar_manager import (
    get_unified_radar_manager,
    get_rwr_threat_level,
    get_rwr_threat_sources,
    get_rwr_max_threat_bearing,
    check_missile_launch_conditions
)

logging.basicConfig(level=logging.INFO)


def example_rwr_usage(env, agent_id: str):
    """
    示例1: RWR系统使用（文档3.7.1节）
    
    RWR威胁等级定义（文档表3.2）：
    0 = 无威胁
    1 = SEARCH（被搜索）- 我方5%激活ECM，敌方10%
    2 = TRACK（被跟踪）- 我方30%激活ECM，敌方40%
    3 = LOCK（被锁定）- 我方80%激活ECM，敌方90%
    4 = MISSILE_LAUNCH（导弹发射）- 100%激活ECM
    5 = MISSILE_GUIDANCE（导弹制导）- 100%激活ECM
    """
    
    # 获取当前威胁等级
    threat_level = get_rwr_threat_level(agent_id)
    
    print(f"\n{'='*60}")
    print(f"RWR系统状态 - {agent_id}")
    print(f"{'='*60}")
    print(f"威胁等级: {threat_level}")
    
    if threat_level == 0:
        print("状态: 安全，无雷达威胁")
    elif threat_level == 1:
        print("状态: 被敌方雷达搜索")
    elif threat_level == 2:
        print("状态: ⚠️ 被敌方雷达跟踪")
    elif threat_level == 3:
        print("状态: 🚨 被敌方雷达锁定！")
    elif threat_level >= 4:
        print("状态: 🚨🚨 导弹威胁！")
    
    # 获取威胁源详情
    threat_sources = get_rwr_threat_sources(agent_id)
    if threat_sources:
        print(f"\n威胁源数量: {len(threat_sources)}")
        for threat in threat_sources:
            print(f"  - {threat['source']}: 等级{threat['level']}, 方位{threat['bearing']:.1f}°")
        
        # 获取最高威胁方位
        max_threat_bearing = get_rwr_max_threat_bearing(agent_id)
        print(f"\n最高威胁方位角 θ_threat: {max_threat_bearing:.1f}°")
        
        # 战术决策示例
        if threat_level >= 3:
            print(f"\n🎯 战术建议: 执行规避机动，朝向{(max_threat_bearing + 90) % 360:.0f}°进行Beam机动")
    else:
        print("威胁源: 无")


def example_missile_launch_check(env, shooter_id: str, target_id: str):
    """
    示例2: 导弹发射条件检查（文档3.7.2节）
    
    完整的6项检查：
    1. 目标在雷达跟踪列表中
    2. 雷达处于TRACK或LOCK模式
    3. 跟踪质量达到最低要求（Q_track ≥ 0.3）
    4. 目标在最大跟踪距离内
    5. 目标不在多普勒盲区
    6. 探测概率达到基本要求（P_d ≥ 0.2）
    """
    
    print(f"\n{'='*60}")
    print(f"导弹发射条件检查 - {shooter_id} → {target_id}")
    print(f"{'='*60}")
    
    # 执行完整的6项检查
    result = check_missile_launch_conditions(env, shooter_id, target_id)
    
    print(f"发射许可: {'✅ 允许' if result['can_launch'] else '❌ 禁止'}")
    print(f"原因: {result['reason']}")
    
    # 显示各项条件详情
    print(f"\n条件检查详情:")
    conditions = result['conditions']
    print(f"  1. 目标在跟踪列表: {'✅' if conditions.get('in_track_list') else '❌'}")
    print(f"  2. 雷达模式正确: {'✅' if conditions.get('radar_mode_ok') else '❌'}")
    print(f"  3. 跟踪质量合格: {'✅' if conditions.get('track_quality_ok') else '❌'}")
    print(f"  4. 目标在范围内: {'✅' if conditions.get('in_range') else '❌'}")
    print(f"  5. 不在多普勒盲区: {'✅' if conditions.get('not_in_notch') else '❌'}")
    print(f"  6. 探测概率足够: {'✅' if conditions.get('detection_prob_ok') else '❌'}")
    
    return result['can_launch']


def example_tactical_decision(env, agent_id: str):
    """
    示例3: 基于RWR的战术决策
    """
    
    threat_level = get_rwr_threat_level(agent_id)
    
    # 根据威胁等级调整战术
    if threat_level >= 3:
        # 被锁定：执行防御机动
        max_threat_bearing = get_rwr_max_threat_bearing(agent_id)
        print(f"\n🚨 {agent_id} 被锁定！执行Beam机动朝向{(max_threat_bearing + 90) % 360:.0f}°")
        
        # 检查是否可以反击
        threat_sources = get_rwr_threat_sources(agent_id)
        for threat in threat_sources:
            if threat['level'] >= 3:
                can_launch = example_missile_launch_check(env, agent_id, threat['source'])
                if can_launch:
                    print(f"✅ 可以对{threat['source']}进行反击")
                else:
                    print(f"❌ 暂时无法对{threat['source']}反击，继续规避")
    
    elif threat_level == 2:
        # 被跟踪：保持警惕
        print(f"\n⚠️ {agent_id} 被跟踪，保持机动")
    
    elif threat_level == 1:
        # 被搜索：正常作战
        print(f"\n✅ {agent_id} 被搜索，继续执行任务")
    
    else:
        # 安全：主动进攻
        print(f"\n✅ {agent_id} 安全，可以主动进攻")


def main():
    """
    主函数：展示完整的雷达功能使用流程
    """
    print("="*60)
    print("雷达系统功能使用示例（文档第3章）")
    print("="*60)
    
    # 注意：这只是示例代码，实际使用时需要传入真实的env对象
    # 在战术任务类中，可以这样调用：
    #
    # # 在战术任务的step()方法中
    # for agent_id in ["A0100", "A0200"]:
    #     # 1. 检查RWR威胁
    #     threat_level = get_rwr_threat_level(agent_id)
    #     
    #     # 2. 根据威胁等级调整战术
    #     if threat_level >= 3:
    #         # 执行防御机动
    #         pass
    #     
    #     # 3. 检查导弹发射条件
    #     for target_id in ["B0100", "B0200"]:
    #         result = check_missile_launch_conditions(env, agent_id, target_id)
    #         if result['can_launch']:
    #             # 发射导弹
    #             pass
    
    print("\n使用方法:")
    print("1. 在战术任务类中导入: from radar_manager import get_rwr_threat_level, check_missile_launch_conditions")
    print("2. 在step()方法中调用RWR接口获取威胁等级")
    print("3. 在导弹发射前调用check_missile_launch_conditions()验证条件")
    print("4. 根据RWR威胁等级调整战术决策（规避、反击等）")


if __name__ == "__main__":
    main()
