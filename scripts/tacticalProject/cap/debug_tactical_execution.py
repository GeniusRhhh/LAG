"""
CAP战术模板执行诊断脚本
用于诊断为什么战术模板代码存在但未执行

使用方法：
1. 在cap_task.py中导入此模块
2. 在关键位置调用诊断函数
3. 运行仿真，观察输出

或者直接运行此脚本进行静态检查
"""
import os
import sys
import logging

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class TacticalExecutionDiagnostics:
    """战术执行诊断工具"""
    
    def __init__(self):
        self.step_count = 0
        self.last_log_step = {}
        self.log_interval = 30  # 每30步打印一次
    
    def should_log(self, key: str) -> bool:
        """检查是否应该打印日志（避免刷屏）"""
        if key not in self.last_log_step:
            self.last_log_step[key] = self.step_count
            return True
        if self.step_count - self.last_log_step[key] >= self.log_interval:
            self.last_log_step[key] = self.step_count
            return True
        return False
    
    def diagnose_cap_state(self, task, agent_id: str):
        """诊断CAP状态机状态"""
        if not self.should_log(f"cap_state_{agent_id}"):
            return
        
        state = task.cap_state_machine.state
        log.info(f"=" * 60)
        log.info(f"[诊断1] CAP状态机 - {agent_id}")
        log.info(f"  当前状态: {state}")
        log.info(f"  step: {self.step_count}")
        
        # 检查状态转换条件
        if hasattr(task, 'picture'):
            threats = task.picture.get_all_threats()
            log.info(f"  威胁数量: {len(threats)}")
            if threats:
                # 计算最近威胁距离
                import numpy as np
                try:
                    x, y = task._get_battlefield_pos(task.env, agent_id)
                    nearest = task.picture.get_nearest_threat(np.array([x, y, 8.0]))
                    if nearest:
                        distance = np.sqrt((nearest.x - x)**2 + (nearest.y - y)**2)
                        log.info(f"  最近威胁: {nearest.track_id}")
                        log.info(f"  威胁距离: {distance:.1f}km")
                        log.info(f"  ENGAGE阈值: 200km")
                        log.info(f"  应该ENGAGE: {distance < 200}")
                except Exception as e:
                    log.error(f"  计算威胁距离失败: {e}")
        log.info(f"=" * 60)
    
    def diagnose_threats(self, task):
        """诊断威胁目标"""
        if not self.should_log("threats"):
            return
        
        log.info(f"=" * 60)
        log.info(f"[诊断2] 威胁目标")
        log.info(f"  step: {self.step_count}")
        
        # 检查AWACS数据
        if hasattr(task, 'awacs'):
            awacs_tracks = task.awacs.get_tracks()
            log.info(f"  AWACS航迹数: {len(awacs_tracks)}")
            for track_id, track in list(awacs_tracks.items())[:3]:  # 只显示前3个
                log.info(f"    {track_id}: pos={track.position[:2]}")
        
        # 检查雷达数据
        if hasattr(task, 'picture'):
            if hasattr(task.picture, '_radar_tracks'):
                log.info(f"  雷达航迹数: {len(task.picture._radar_tracks)}")
            
            # 检查融合航迹
            threats = task.picture.get_all_threats()
            log.info(f"  融合航迹数: {len(threats)}")
            for threat in threats[:3]:  # 只显示前3个
                log.info(f"    {threat.track_id}: pos=({threat.x:.1f}, {threat.y:.1f})")
        
        log.info(f"=" * 60)
    
    def diagnose_tactic_assignment(self, task, agent_id: str):
        """诊断战术分配"""
        if not self.should_log(f"tactic_{agent_id}"):
            return
        
        log.info(f"=" * 60)
        log.info(f"[诊断3] 战术分配 - {agent_id}")
        log.info(f"  step: {self.step_count}")
        
        # 检查战术分配字典
        has_dict = hasattr(task, '_tactic_assignments_by_agent')
        log.info(f"  has_assignments_dict: {has_dict}")
        
        if has_dict:
            assignments = task._tactic_assignments_by_agent
            log.info(f"  assignments数量: {len(assignments)}")
            
            if agent_id in assignments:
                assign = assignments[agent_id]
                log.info(f"  {agent_id}的分配:")
                log.info(f"    战术: {assign.tactic}")
                log.info(f"    目标: {assign.target_id}")
                log.info(f"    射手: {assign.shooter_id}")
                log.info(f"    支援: {assign.support_id}")
            else:
                log.info(f"  {agent_id}无战术分配")
                log.info(f"  已分配的agent: {list(assignments.keys())}")
        
        # 检查最后一次分配
        if hasattr(task, '_last_tactic_assignment'):
            last = task._last_tactic_assignment
            if last:
                log.info(f"  最后分配:")
                log.info(f"    战术: {last.tactic}")
                log.info(f"    目标: {last.target_id}")
        
        log.info(f"=" * 60)
    
    def diagnose_engage_action(self, task, agent_id: str, assign, maneuver_action):
        """诊断ENGAGE动作执行"""
        if not self.should_log(f"engage_{agent_id}"):
            return
        
        log.info(f"=" * 60)
        log.info(f"[诊断4] ENGAGE动作 - {agent_id}")
        log.info(f"  step: {self.step_count}")
        log.info(f"  assign: {assign}")
        
        if assign:
            log.info(f"  战术类型: {assign.tactic}")
            log.info(f"  目标ID: {assign.target_id}")
            log.info(f"  maneuver_action: {maneuver_action}")
            
            # 检查战术方法是否存在
            from .tactic_selector import TacticType
            method_map = {
                TacticType.T_DS: '_execute_drag_shoot',
                TacticType.T_PA: '_execute_pincer_attack',
                TacticType.T_HL: '_execute_high_low_attack',
                TacticType.T_SBS: '_execute_side_by_side',
                TacticType.T_FB: '_execute_front_back',
            }
            
            if assign.tactic in method_map:
                method_name = method_map[assign.tactic]
                has_method = hasattr(task, method_name)
                log.info(f"  战术方法: {method_name}")
                log.info(f"  方法存在: {has_method}")
                
                if maneuver_action is None:
                    log.warning(f"  ⚠️ 战术方法返回None!")
        else:
            log.info(f"  无战术分配，使用默认动作")
        
        log.info(f"=" * 60)
    
    def print_summary(self, task):
        """打印诊断摘要"""
        if not self.should_log("summary"):
            return
        
        log.info(f"\n" + "=" * 80)
        log.info(f"[诊断摘要] step={self.step_count}")
        log.info(f"=" * 80)
        
        # CAP状态
        log.info(f"1. CAP状态: {task.cap_state_machine.state}")
        
        # 威胁数量
        if hasattr(task, 'picture'):
            threats = task.picture.get_all_threats()
            log.info(f"2. 威胁数量: {len(threats)}")
        
        # 战术分配
        if hasattr(task, '_tactic_assignments_by_agent'):
            assignments = task._tactic_assignments_by_agent
            log.info(f"3. 战术分配数: {len(assignments)}")
            for aid, assign in assignments.items():
                log.info(f"   {aid}: {assign.tactic.value} -> {assign.target_id}")
        
        # 战术选择器状态
        if hasattr(task, 'tactic_selector'):
            current_tactic = task.tactic_selector.get_current_tactic()
            log.info(f"4. 当前战术: {current_tactic}")
        
        log.info(f"=" * 80 + "\n")


# 全局诊断实例
_diagnostics = TacticalExecutionDiagnostics()


def diagnose_cap_state(task, agent_id: str):
    """诊断CAP状态机"""
    _diagnostics.step_count = task.step_count
    _diagnostics.diagnose_cap_state(task, agent_id)


def diagnose_threats(task):
    """诊断威胁目标"""
    _diagnostics.step_count = task.step_count
    _diagnostics.diagnose_threats(task)


def diagnose_tactic_assignment(task, agent_id: str):
    """诊断战术分配"""
    _diagnostics.step_count = task.step_count
    _diagnostics.diagnose_tactic_assignment(task, agent_id)


def diagnose_engage_action(task, agent_id: str, assign, maneuver_action):
    """诊断ENGAGE动作"""
    _diagnostics.step_count = task.step_count
    _diagnostics.diagnose_engage_action(task, agent_id, assign, maneuver_action)


def print_summary(task):
    """打印诊断摘要"""
    _diagnostics.step_count = task.step_count
    _diagnostics.print_summary(task)


# ============================================================================
# 静态检查：验证代码完整性
# ============================================================================

def static_check():
    """静态检查：验证战术模板代码是否存在"""
    log.info("=" * 80)
    log.info("CAP战术模板静态检查")
    log.info("=" * 80)
    
    cap_task_path = os.path.join(os.path.dirname(__file__), 'cap_task.py')
    
    if not os.path.exists(cap_task_path):
        log.error(f"❌ cap_task.py不存在: {cap_task_path}")
        return False
    
    with open(cap_task_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键方法
    methods_to_check = [
        ('get_action', '动作分发入口'),
        ('_get_engage_action', '交战动作处理'),
        ('_execute_drag_shoot', '拖曳射击'),
        ('_execute_pincer_attack', '钳形攻势'),
        ('_execute_front_back', '前后攻击'),
        ('_execute_high_low_attack', '高低攻击'),
        ('_execute_side_by_side', '并排射击'),
        ('_update_tactic_selection', '战术选择更新'),
    ]
    
    log.info("\n检查关键方法:")
    all_exist = True
    for method_name, description in methods_to_check:
        exists = f'def {method_name}(' in content
        status = "✅" if exists else "❌"
        log.info(f"  {status} {method_name:30s} - {description}")
        if not exists:
            all_exist = False
    
    # 检查关键变量
    log.info("\n检查关键变量:")
    variables_to_check = [
        ('_tactic_assignments_by_agent', '战术分配字典'),
        ('tactic_selector', '战术选择器'),
        ('cap_state_machine', 'CAP状态机'),
        ('picture', '态势图'),
    ]
    
    for var_name, description in variables_to_check:
        exists = var_name in content
        status = "✅" if exists else "❌"
        log.info(f"  {status} {var_name:30s} - {description}")
        if not exists:
            all_exist = False
    
    # 检查调用链
    log.info("\n检查调用链:")
    call_chains = [
        ('if assign.tactic == TacticType.T_DS:', '拖曳射击调用'),
        ('maneuver_action = self._execute_drag_shoot', '拖曳射击执行'),
        ('if assign.tactic == TacticType.T_PA:', '钳形攻势调用'),
        ('maneuver_action = self._execute_pincer_attack', '钳形攻势执行'),
    ]
    
    for pattern, description in call_chains:
        exists = pattern in content
        status = "✅" if exists else "❌"
        log.info(f"  {status} {description}")
        if not exists:
            all_exist = False
    
    log.info("\n" + "=" * 80)
    if all_exist:
        log.info("✅ 静态检查通过：所有关键代码都存在")
        log.info("问题可能是运行时状态问题，需要动态诊断")
    else:
        log.error("❌ 静态检查失败：部分关键代码缺失")
    log.info("=" * 80)
    
    return all_exist


if __name__ == '__main__':
    # 运行静态检查
    static_check()
    
    log.info("\n使用方法:")
    log.info("1. 在cap_task.py中导入此模块:")
    log.info("   from .debug_tactical_execution import diagnose_cap_state, diagnose_threats, ...")
    log.info("\n2. 在get_action()中添加:")
    log.info("   diagnose_cap_state(self, agent_id)")
    log.info("\n3. 在_get_engage_action()中添加:")
    log.info("   diagnose_tactic_assignment(self, agent_id)")
    log.info("   diagnose_engage_action(self, agent_id, assign, maneuver_action)")
    log.info("\n4. 在step()中添加:")
    log.info("   diagnose_threats(self)")
    log.info("   print_summary(self)")
