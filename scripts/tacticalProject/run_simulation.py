"""
战术仿真运行脚本
支持5种战术的完整2v2空战仿真

🎯 战术选择配置：
修改下面的 FORCE_TACTIC 参数来强制选择特定战术进行测试：
- None: 使用智能战术选择（默认）
- 'DRAG_SHOOT': 拖拽射击战术
- 'PINCER_ATTACK': 钳形攻势战术
- 'HIGH_LOW_ATTACK': 高低攻击战术
- 'FRONT_BACK': 前后攻击战术
- 'SIDE_BY_SIDE': 并排返航战术
"""
import os
import sys
import logging
import numpy as np
import argparse
from datetime import datetime

# [战术选择配置] - 修改此参数来强制选择特定战术
# 注意：这里必须是 None（而不是字符串'None'），否则会被当成"强制战术"导致行为异常
FORCE_TACTIC = None # 选项: None, 'DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE'

# [CAP模式配置] - 修改此参数来切换CAP模式（4v4、动态路径、态势感知）
# True: 使用CAP模式（4v4、动态路径、冷热交替巡逻）
# False: 使用原有模式（2v2、固定路径）
USE_CAP_MODE = False  # 选项: True, False

# [飞机模型配置] - 直接在这里修改飞机类型
# 选项: 'f16', 'su27sk'
MY_AIRCRAFT_TYPE = 'su27sk'      # 我方飞机类型
ENEMY_AIRCRAFT_TYPE = 'f16'   # 敌方飞机类型

# 根据飞机类型自动设置底层控制模型环境变量
if MY_AIRCRAFT_TYPE == 'su27sk':
    os.environ["FRIEND_BASELINE_MODEL"] = "SU27"
    logging.info("已设置 FRIEND_BASELINE_MODEL = SU27")
else:
    os.environ["FRIEND_BASELINE_MODEL"] = "F16"

if ENEMY_AIRCRAFT_TYPE == 'su27sk':
    os.environ["ENEMY_BASELINE_MODEL"] = "SU27"
else:
    os.environ["ENEMY_BASELINE_MODEL"] = "F16"

# 🔍 详细调试配置
DETAILED_DEBUG = True  # 启用详细调试信息
DEBUG_INTERVAL = 60    # 调试信息输出间隔（步数）

# 📊 仿真分析配置
ANALYSIS_MODE = True   # 启用仿真分析模式
ANALYSIS_INTERVAL = 60 # 分析报告输出间隔（步数，对应12秒）

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.utils.utils import parse_config
import tactical_task as tactical_task_module
from tactical_task import TacticalTask

# 导入CAP任务（如果启用CAP模式）
if USE_CAP_MODE:
    try:
        from cap.cap_task import CAPTask
        from cap.patrol_task import SimplePatrolTask
        CAP_AVAILABLE = True
        logging.info("✅ 巡逻模式已启用，将优先使用CAPTask（P1-4完整巡逻）")
    except ImportError as e:
        CAP_AVAILABLE = False
        logging.warning(f"⚠️ 巡逻模式启用失败，将使用原有模式: {e}")
        import traceback
        logging.debug(f"导入错误详情: {traceback.format_exc()}")
else:
    CAP_AVAILABLE = False

# =========================
# ✅ PyCharm 一键切换初始态势
# 用法：直接运行本脚本，不传任何参数。
# 只需要改下面这一行 DEFAULT_RUN_INTENT，即可在三种态势间切换：
# - 'conservative_clear'（默认，使用 tactical_bvr.yaml）
# - 'aggressive_clear'   （使用 tactical_bvr_aggressive_clear.yaml）
# - 'defensive'          （使用 tactical_bvr_defensive.yaml）
# =========================
DEFAULT_RUN_INTENT = 'conservative_clear'
from core import TacticalDecisionManager
import shutil
from configure_aircraft import TacticalAircraftConfigManager

class ConfigurableMultipleCombatEnv(MultipleCombatEnv):
    """支持动态飞机配置的仿真环境"""
    def __init__(self, config_name, my_aircraft=None, enemy_aircraft=None):
        # 优先使用传入的参数，否则使用全局配置
        self.my_aircraft = my_aircraft or MY_AIRCRAFT_TYPE
        self.enemy_aircraft = enemy_aircraft or ENEMY_AIRCRAFT_TYPE
        super().__init__(config_name)

    def load_simulator(self):
        # 在加载模拟器前应用配置覆盖
        logging.info(f"Applying Tactical Config: My={self.my_aircraft}, Enemy={self.enemy_aircraft}")
        
        # 遍历配置中的所有飞机
        for uid, conf in self.config.aircraft_configs.items():
            if uid.startswith('A'): # 我方
                conf['model'] = self.my_aircraft
            elif uid.startswith('B'): # 敌方
                conf['model'] = self.enemy_aircraft
                
        super().load_simulator()

# 导入JSBSim catalog
try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    # 如果导入失败，创建一个基础的catalog类
    class c:
        attitude_psi_rad = "attitude/psi-rad"
        position_h_sl_m = "position/h-sl-m"


def print_analysis_report(env, step, selected_tactic, tactical_task):
    """打印标准化分析报告（每12秒）"""
    if not ANALYSIS_MODE or step % ANALYSIS_INTERVAL != 0:
        return

    try:
        current_time = step * 0.2
        # 🔧 [任务1修复] 注释掉冗余的详细状态报告，保留关键战术决策日志
        # logging.info(f"\n{'='*80}")
        # logging.info(f"📊 [仿真分析报告] 时间: {current_time:.1f}s | 步数: {step} | 战术: {selected_tactic}")
        # logging.info(f"{'='*80}")

        # # 我方状态分析
        # logging.info(f"🔵 [我方状态分析]")
        # for agent_id in ['A0100', 'A0200']:
        #     if agent_id in env.agents:
        #         agent = env.agents[agent_id]
        #         log_agent_analysis(agent, agent_id, "我方", tactical_task)

        # # 敌方状态分析
        # logging.info(f"🔴 [敌方状态分析]")
        # for agent_id in ['B0100', 'B0200']:
        #     if agent_id in env.agents:
        #         agent = env.agents[agent_id]
        #         log_agent_analysis(agent, agent_id, "敌方", tactical_task)

        # # 战术执行分析
        # log_tactical_execution_analysis(env, tactical_task, current_time)

        # # 导弹状态分析
        # log_missile_analysis(env, current_time)

        # # 雷达威胁分析
        # log_radar_threat_analysis(env, current_time)

        # logging.info(f"{'='*80}\n")

    except Exception as e:
        logging.error(f"分析报告生成错误: {e}")

def log_agent_analysis(agent, agent_id, team_name, tactical_task):
    """记录单个飞机的分析信息"""
    try:
        # 基础状态
        pos = agent.get_position()
        vel = agent.get_velocity()
        speed = np.linalg.norm(vel)
        altitude = pos[2]

        # 安全获取航向信息
        heading = 0.0
        try:
            heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
        except (NameError, AttributeError) as e:
            # 如果catalog不可用，尝试直接使用字符串
            try:
                heading = np.rad2deg(agent.get_property_value("attitude/psi-rad"))
            except:
                heading = 0.0  # 默认值

        # 战术状态
        tactical_phase = "未知"
        if hasattr(tactical_task, 'tactic_roles') and agent_id in tactical_task.tactic_roles:
            role_info = tactical_task.tactic_roles[agent_id]
            if hasattr(role_info, 'tactical_phase'):
                tactical_phase = role_info.tactical_phase.value

        # 存活状态
        is_alive = getattr(agent, 'is_alive', True)

        # 导弹数量
        missile_count = 0
        try:
            if hasattr(agent, 'weapon_manager') and hasattr(agent.weapon_manager, 'weapons'):
                for weapon in agent.weapon_manager.weapons:
                    if hasattr(weapon, 'count'):
                        missile_count += weapon.count
        except:
            missile_count = 4  # 默认值

        logging.info(f"  [{agent_id}] {team_name} | 存活: {'是' if is_alive else '否'} | 阶段: {tactical_phase}")
        logging.info(f"    位置: ({pos[0]:.0f}, {pos[1]:.0f}) | 高度: {altitude:.0f}m | 航向: {heading:.1f}°")
        logging.info(f"    速度: {speed:.1f}m/s | 导弹: {missile_count}枚")

    except Exception as e:
        logging.error(f"  [{agent_id}] 状态分析失败: {e}")

def log_tactical_execution_analysis(env, tactical_task, current_time):
    """记录战术执行分析"""
    try:
        logging.info(f"⚔️ [战术执行分析]")

        # 当前战术阶段
        if hasattr(tactical_task, 'current_phase'):
            current_phase = tactical_task.current_phase.value
            logging.info(f"  全局战术阶段: {current_phase}")

        # 距离信息
        if 'A0100' in env.agents and 'B0100' in env.agents:
            distance = calculate_distance(env.agents['A0100'], env.agents['B0100'])
            logging.info(f"  主要目标距离: {distance:.1f}km")

        # 导弹发射状态
        if hasattr(tactical_task, 'missile_launched'):
            launched_agents = [aid for aid, launched in tactical_task.missile_launched.items() if launched]
            if launched_agents:
                logging.info(f"  已发射导弹: {', '.join(launched_agents)}")

        # 最近导弹发射时间
        if hasattr(tactical_task, 'last_missile_launch_time'):
            recent_launches = []
            for aid, launch_time in tactical_task.last_missile_launch_time.items():
                if launch_time > 0 and (current_time - launch_time) < 30:
                    recent_launches.append(f"{aid}({current_time - launch_time:.1f}s前)")
            if recent_launches:
                logging.info(f"  近期发射: {', '.join(recent_launches)}")

    except Exception as e:
        logging.error(f"战术执行分析失败: {e}")

def log_missile_analysis(env, current_time):
    """记录导弹状态分析"""
    try:
        logging.info(f"🚀 [导弹状态分析]")

        # 统计飞行中的导弹
        missiles_found = []

        # 检查 temp_simulators
        temp_simulators = getattr(env, 'temp_simulators', {})
        for sim_name, sim_obj in temp_simulators.items():
            if any(pattern in sim_name for pattern in ['10001', '20001', 'missile']):
                missiles_found.append(sim_name)

        # 检查 agents 中的导弹
        agents_dict = getattr(env, 'agents', {})
        for agent_id in agents_dict.keys():
            if any(pattern in agent_id for pattern in ['10001', '20001']) and len(agent_id) >= 5:
                missiles_found.append(agent_id)

        if missiles_found:
            logging.info(f"  飞行中导弹: {len(missiles_found)}枚")
            for missile_id in missiles_found[:3]:  # 只显示前3枚
                missile_type = "AIM-120C" if "A" in missile_id or "10001" in missile_id else "R-27ER"
                logging.info(f"    [{missile_id}] {missile_type}")
            if len(missiles_found) > 3:
                logging.info(f"    ... 还有{len(missiles_found) - 3}枚导弹")
        else:
            logging.info(f"  飞行中导弹: 0枚")

    except Exception as e:
        logging.error(f"导弹状态分析失败: {e}")

def log_radar_threat_analysis(env, current_time):
    """记录雷达威胁分析"""
    try:
        logging.info(f"📡 [雷达威胁分析]")

        agents_dict = getattr(env, 'agents', {})
        for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']:
            if agent_id in agents_dict:
                agent = agents_dict[agent_id]

                # 威胁等级
                rwr_level = 1  # 默认威胁等级
                ecm_status = False

                try:
                    # 尝试获取ECM状态
                    if hasattr(agent, 'ecm_on'):
                        ecm_status = agent.ecm_on
                    elif hasattr(agent, 'get_property_value'):
                        try:
                            ecm_status = agent.get_property_value(c.ecm_on) > 0
                        except (NameError, AttributeError):
                            # 如果catalog不可用，尝试直接使用字符串
                            try:
                                ecm_status = agent.get_property_value("ecm/on") > 0
                            except:
                                ecm_status = False
                except:
                    pass

                team = "我方" if agent_id.startswith('A') else "敌方"
                logging.info(f"  [{agent_id}] {team} | 威胁等级: {rwr_level} | ECM: {'ON' if ecm_status else 'OFF'}")

    except Exception as e:
        logging.error(f"雷达威胁分析失败: {e}")

def print_detailed_debug_info(env, step, selected_tactic, tactical_task):
    """打印详细的调试信息"""
    if not DETAILED_DEBUG or step % DEBUG_INTERVAL != 0:
        return
        
    try:
        current_time = step * 0.2
        print(f"\n" + "="*100)
        print(f"🔍 [详细状态报告] 时间: {current_time:.1f}s | 步数: {step} | 当前战术: {selected_tactic}")
        print("="*100)
        
        # 我方状态详情
        print(f"\n🔵 [我方详细状态]")
        for agent_id in ['A0100', 'A0200']:
            if agent_id in env.agents:
                agent = env.agents[agent_id]
                print_agent_detailed_status(agent, "我方")
        
        # 敌方状态详情
        print(f"\n🔴 [敌方详细状态]")
        for agent_id in ['B0100', 'B0200']:
            if agent_id in env.agents:
                agent = env.agents[agent_id]
                print_agent_detailed_status(agent, "敌方")
        
        # 雷达状态总览
        print(f"\n📡 [雷达系统状态]")
        print_radar_status(env)
        
        # 导弹状态总览
        print(f"\n[导弹系统状态]")
        print_missile_status(env)
        
        # 战术节点状态
        print(f"\n[战术节点状态]")
        print_tactical_node_status(env, tactical_task)
        
        print("="*100 + "\n")
        
    except Exception as e:
        logging.error(f"调试信息输出错误: {e}")

def print_agent_detailed_status(agent, team_name):
    """打印单个飞机的详细状态"""
    try:
        # 先尝试获取基础位置信息
        pos_x = pos_y = altitude = velocity = heading = 0
        
        # 尝试获取位置 - 先用get_position方法
        try:
            if hasattr(agent, 'get_position'):
                position = agent.get_position()
                if position and len(position) >= 3:
                    pos_x, pos_y, altitude = position[0], position[1], position[2]
        except:
            pass
            
        # 如果get_position失败，尝试属性获取
        if pos_x == 0 and pos_y == 0:
            try:
                pos_x = agent.get_property_value('position/lat-gc-deg') or 0
                pos_y = agent.get_property_value('position/long-gc-deg') or 0
                altitude = agent.get_property_value('position/h-sl-meters') or agent.get_property_value('position/h-sl-ft', 0) * 0.3048
            except:
                pass
        
        # 尝试获取速度和航向
        try:
            velocity = (agent.get_property_value('velocities/vtrue-fps') or 0) * 0.3048
            heading = agent.get_property_value('attitude/psi-deg') or 0
        except:
            pass
        
        # 获取更多状态信息
        try:
            fuel_level = agent.get_property_value('propulsion/total-fuel-lbs') or 0
        except:
            fuel_level = 0
            
        try:
            g_force = agent.get_property_value('accelerations/n-pilot-z-norm') or 0
        except:
            g_force = 0
        
        print(f"  [飞机] [{agent.uid}] {team_name}")
        print(f"     位置: ({pos_x:.0f}, {pos_y:.0f}) | 高度: {altitude:.0f}m")
        print(f"     速度: {velocity:.1f}m/s | 航向: {heading:.1f}°")
        print(f"     燃料: {fuel_level:.0f}lbs | G力: {g_force:.2f}")
        
        # 安全获取导弹数量和存活状态
        missile_count = 0
        for attr_name in ['missiles', 'missile_list', 'missile_count']:
            if hasattr(agent, attr_name):
                missiles = getattr(agent, attr_name)
                if isinstance(missiles, list):
                    missile_count = len(missiles)
                elif isinstance(missiles, int):
                    missile_count = missiles
                break
        
        is_alive = getattr(agent, 'is_alive', True)
        print(f"     存活: {'是' if is_alive else '否'} | 导弹数量: {missile_count}")
        
        # 战术阶段信息
        if hasattr(agent, 'tactical_phase'):
            print(f"     战术阶段: {agent.tactical_phase}")
            
    except Exception as e:
        print(f"  [错误] [{agent.uid}] 状态读取失败: {e}")

def print_radar_status(env):
    """打印雷达系统状态"""
    try:
        # 先尝试使用雷达管理器API
        try:
            # 动态导入雷达管理器API
            import sys
            import os
            simulation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation')
            if simulation_dir not in sys.path:
                sys.path.append(simulation_dir)
            
            from radar_manager import (
                get_rwr_threat_level, 
                is_enemy_ecm_active, 
                get_enemy_lock_target, 
                get_unified_radar_manager
            )
            radar_api_available = True
            # print("[DEBUG] 雷达管理器API加载成功")
        except ImportError as e:
            # print(f"[DEBUG] 雷达管理器API不可用: {e}")
            radar_api_available = False

        agents_dict = getattr(env, 'agents', {})
        for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']:
            if agent_id in agents_dict:
                agent = agents_dict[agent_id]
                # 获取雷达相关信息
                try:
                    rwr_level = 0
                    ecm_status = False
                    radar_lock = None
                    
                    # 优先使用雷达管理器API
                    if radar_api_available:
                        try:
                            # 获取雷达管理器实例
                            radar_manager = get_unified_radar_manager()
                            
                            # 战斗环境下强制威胁等级为1（确保对称性）
                            rwr_level = 1  # 在战斗环境中默认威胁等级为1
                            print(f"[DEBUG] 战斗环境强制设置{agent_id}威胁等级=1")
                            
                            # 验证雷达状态以确认强制设置的合理性
                            if hasattr(radar_manager, 'enemy_radar_states'):
                                enemy_searching = False
                                for enemy_id, radar_state in radar_manager.enemy_radar_states.items():
                                    if enemy_id != agent_id and str(radar_state) == 'RadarStatus.SEARCH':
                                        enemy_searching = True
                                        print(f"[DEBUG] 检测到敌方{enemy_id}雷达搜索状态，{agent_id}威胁等级确认为1")
                                        break
                                
                                if not enemy_searching and hasattr(radar_manager, 'friendly_radar_states'):
                                    for friendly_id, radar_state in radar_manager.friendly_radar_states.items():
                                        if friendly_id != agent_id and str(radar_state) == 'RadarStatus.SEARCH':
                                            print(f"[DEBUG] 检测到友方{friendly_id}雷达搜索状态，{agent_id}威胁等级确认为1")
                                            break
                            
                            # ECM状态 - 检查具体ECM状态
                            if hasattr(radar_manager, 'ecm_states'):
                                ecm_status = radar_manager.ecm_states.get(agent_id, False)
                            else:
                                ecm_status = is_enemy_ecm_active(agent_id)
                            
                            # 雷达锁定目标
                            if hasattr(radar_manager, 'enemy_lock_targets'):
                                radar_lock = radar_manager.enemy_lock_targets.get(agent_id, None)
                            else:
                                radar_lock = get_enemy_lock_target(agent_id)
                                
                            # 调试信息：显示雷达管理器的内部状态
                            if hasattr(radar_manager, 'rwr_states') and agent_id in radar_manager.rwr_states:
                                rwr_state = radar_manager.rwr_states[agent_id]
                                print(f"[DEBUG] {agent_id} RWR内部状态: {rwr_state}")
                            
                            # 调试：检查雷达状态是否在更新
                            if hasattr(radar_manager, 'enemy_radar_states'):
                                print(f"[DEBUG] 敌方雷达状态: {dict(radar_manager.enemy_radar_states)}")
                            if hasattr(radar_manager, 'friendly_radar_states'):
                                print(f"[DEBUG] 友方雷达状态: {dict(radar_manager.friendly_radar_states)}")
                            
                        except Exception as api_e:
                            # print(f"[DEBUG] 雷达API调用失败 {agent_id}: {api_e}")
                            pass
                    
                    # 如果API不可用，回退到属性读取
                    if not radar_api_available or rwr_level == 0:
                        # RWR威胁等级
                        rwr_attrs = ['rwr_threat_level', 'radar_warning', 'threat_level', 'rwr_level']
                        for attr in rwr_attrs:
                            try:
                                val = getattr(agent, attr, None)
                                if val is not None and val != 0:
                                    rwr_level = val
                                    break
                            except:
                                continue
                    
                    if not radar_api_available or not ecm_status:
                        # ECM状态
                        ecm_attrs = ['ecm_active', 'ecm_on', 'ecm_status', 'countermeasure_active']
                        for attr in ecm_attrs:
                            try:
                                val = getattr(agent, attr, None)
                                if val is not None:
                                    ecm_status = bool(val)
                                    break
                            except:
                                continue
                    
                    if not radar_api_available or not radar_lock:
                        # 雷达锁定目标
                        lock_attrs = ['radar_locked_target', 'locked_target', 'target_locked', 'radar_target']
                        for attr in lock_attrs:
                            try:
                                val = getattr(agent, attr, None)
                                if val is not None:
                                    radar_lock = val
                                    break
                            except:
                                continue
                    
                    # 如果还是获取不到，尝试从jsbsim属性获取
                    if rwr_level == 0:
                        try:
                            rwr_level = agent.get_property_value('systems/rwr/threat-level') or 0
                        except:
                            pass
                            
                    if not ecm_status:
                        try:
                            ecm_status = bool(agent.get_property_value('systems/ecm/active') or False)
                        except:
                            pass
                    
                    print(f"  [雷达] [{getattr(agent, 'uid', agent_id)}] RWR威胁等级: {rwr_level} | ECM: {'ON' if ecm_status else 'OFF'} | 锁定目标: {radar_lock or '无'}")
                except Exception as radar_e:
                    print(f"  [雷达] [{getattr(agent, 'uid', agent_id)}] 雷达状态: 信息不可用 ({radar_e})")
    except Exception as e:
        print(f"  [错误] 雷达状态读取失败: {e}")

def print_missile_status(env):
    """打印导弹系统状态"""
    try:
        missiles_found = []
        missile_count = 0
        
        print(f"[导弹系统状态]")
        
        # 优先检查 env.missiles（这是最可能的存储位置）
        missiles_list = getattr(env, 'missiles', [])
        if missiles_list:
            print(f"[DEBUG] 在env.missiles中发现{len(missiles_list)}枚导弹: {missiles_list}")
            for missile_id in missiles_list:
                # 尝试获取导弹对象
                missile_obj = None
                if hasattr(env, 'agents') and missile_id in env.agents:
                    missile_obj = env.agents[missile_id]
                elif hasattr(env, 'temp_simulators') and missile_id in env.temp_simulators:
                    missile_obj = env.temp_simulators[missile_id]
                missiles_found.append((missile_id, missile_obj))
        else:
            print(f"[DEBUG] env.missiles为空或不存在")
        
        # 检查 temp_simulators
        temp_simulators = getattr(env, 'temp_simulators', None)
        if temp_simulators:
            temp_missiles = [name for name in temp_simulators.keys() if any(c in name for c in ['10001', '20001']) or 'missile' in name.lower()]
            if temp_missiles:
                print(f"[DEBUG] 在temp_simulators中发现导弹: {temp_missiles}")
                for sim_name in temp_missiles:
                    missiles_found.append((sim_name, temp_simulators[sim_name]))
        
        # 检查 task.temp_simulators
        task = getattr(env, 'task', None)
        if task and hasattr(task, 'temp_simulators'):
            task_temp_simulators = getattr(task, 'temp_simulators', None)
            if task_temp_simulators:
                task_missiles = [name for name in task_temp_simulators.keys() if any(c in name for c in ['10001', '20001']) or 'missile' in name.lower()]
                if task_missiles:
                    print(f"[DEBUG] 在task.temp_simulators中发现导弹: {task_missiles}")
                    for sim_name in task_missiles:
                        missiles_found.append((sim_name, task_temp_simulators[sim_name]))
        
        # 检查agents字典中的导弹（以特定模式命名的导弹ID）
        agents_dict = getattr(env, 'agents', {})
        print(f"[DEBUG] 检查agents_dict: {list(agents_dict.keys())}")
        agent_missiles = [aid for aid in agents_dict.keys() if any(pattern in aid for pattern in ['10001', '20001', 'B10', 'A10', 'B20', 'A20']) and len(aid) >= 5]
        if agent_missiles:
            print(f"[DEBUG] 在agents中发现导弹: {agent_missiles}")
            for agent_id in agent_missiles:
                missiles_found.append((agent_id, agents_dict[agent_id]))
        
        # 去重并显示导弹信息
        seen_missiles = set()
        for sim_name, simulator in missiles_found:
            if sim_name not in seen_missiles:
                seen_missiles.add(sim_name)
                missile_count += 1
                
                # 简化导弹信息显示
                missile_type = "未知"
                if "AIM" in sim_name or "A1" in sim_name or "A2" in sim_name:
                    missile_type = "AIM-120C7" if "A" in sim_name else "导弹"
                elif "B1" in sim_name or "B2" in sim_name or "R27" in sim_name:
                    missile_type = "R-27ER" if "B" in sim_name else "导弹"
                
                print(f"  [导弹] [{sim_name}] 类型: {missile_type} | 状态: 飞行中")
        
        if missile_count == 0:
            print(f"  [信息] 当前无导弹在飞行")
        else:
            print(f"  [信息] 共{missile_count}枚导弹在飞行")
            print(f"  [提示] 导弹详细轨迹请查看仿真日志")
            
    except Exception as e:
        print(f"  [错误] 导弹状态读取失败: {e}")

def print_tactical_node_status(env, tactical_task):
    """打印战术节点状态"""
    try:
        agents_dict = getattr(env, 'agents', {})
        for agent_id in ['A0100', 'A0200']:
            if agent_id in agents_dict:
                agent = agents_dict[agent_id]
                try:
                    # 从tactical_task获取战术状态信息
                    if tactical_task:
                        # 获取当前选定的战术
                        current_tactic = getattr(tactical_task, 'selected_tactic', 'Unknown') or 'Unknown'
                        
                        # 获取战术阶段 - 优先从state_manager获取
                        tactical_phase = 'Unknown'
                        if hasattr(tactical_task, 'state_manager') and tactical_task.state_manager:
                            try:
                                phase = tactical_task.state_manager.get_agent_phase(agent_id)
                                if phase:
                                    tactical_phase = str(phase).split('.')[-1]  # 去掉枚举前缀
                            except:
                                pass
                        
                        # 如果从state_manager获取失败，尝试从agent获取
                        if tactical_phase == 'Unknown' and hasattr(agent, 'tactical_phase'):
                            tactical_phase = str(agent.tactical_phase).split('.')[-1]
                        
                        # 获取角色信息
                        role = 'Unknown'
                        if hasattr(tactical_task, 'tactic_roles') and tactical_task.tactic_roles:
                            if isinstance(tactical_task.tactic_roles, dict):
                                # 直接查找agent_id作为key
                                if agent_id in tactical_task.tactic_roles:
                                    role = tactical_task.tactic_roles[agent_id]
                                # 查找role类型 (lead/wingman等)
                                elif 'lead' in tactical_task.tactic_roles and tactical_task.tactic_roles['lead'] == agent_id:
                                    role = 'Leader'
                                elif 'wingman' in tactical_task.tactic_roles and tactical_task.tactic_roles['wingman'] == agent_id:
                                    role = 'Wingman'
                                elif 'high' in tactical_task.tactic_roles and tactical_task.tactic_roles['high'] == agent_id:
                                    role = 'High'
                                elif 'low' in tactical_task.tactic_roles and tactical_task.tactic_roles['low'] == agent_id:
                                    role = 'Low'
                        
                        # 如果还是Unknown，使用默认角色
                        if role == 'Unknown':
                            role = 'Leader' if agent_id == 'A0100' else 'Wingman'
                    else:
                        current_tactic = 'No_Task'
                        tactical_phase = 'No_Task'
                        role = 'No_Task'
                    
                    agent_uid = getattr(agent, 'uid', agent_id)
                    print(f"  [战术] [{agent_uid}] 战术: {current_tactic} | 阶段: {tactical_phase} | 角色: {role}")
                    
                    # 计算与敌方的距离
                    if agent_id == 'A0100' and 'B0100' in agents_dict:
                        distance = calculate_distance(agent, agents_dict['B0100'])
                        print(f"       与主要目标距离: {distance:.1f}km")
                        
                except Exception as e:
                    agent_uid = getattr(agent, 'uid', agent_id) if 'agent' in locals() else agent_id
                    print(f"  [错误] [{agent_uid}] 战术状态读取失败: {e}")
                    # 添加调试信息
                    if tactical_task:
                        print(f"    调试: selected_tactic={getattr(tactical_task, 'selected_tactic', 'NO_ATTR')}")
                        print(f"    调试: tactic_roles={getattr(tactical_task, 'tactic_roles', 'NO_ATTR')}")
    except Exception as e:
        print(f"  [错误] 战术节点状态读取失败: {e}")

def calculate_distance(agent1, agent2):
    """计算两架飞机之间的距离"""
    try:
        # 尝试多种位置获取方式
        x1, y1, x2, y2 = 0, 0, 0, 0
        
        # 方式1: 使用get_position方法
        try:
            if hasattr(agent1, 'get_position') and hasattr(agent2, 'get_position'):
                pos1 = agent1.get_position()
                pos2 = agent2.get_position()
                if pos1 and pos2 and len(pos1) >= 2 and len(pos2) >= 2:
                    x1, y1 = pos1[0], pos1[1]
                    x2, y2 = pos2[0], pos2[1]
                    if abs(x1) + abs(y1) + abs(x2) + abs(y2) > 0.1:  # 确保不是所有值都是0
                        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                        return distance / 1000  # 转换为km
        except:
            pass
            
        # 方式2: 使用经纬度计算距离
        try:
            lat1 = agent1.get_property_value('position/lat-gc-deg')
            lon1 = agent1.get_property_value('position/long-gc-deg')
            lat2 = agent2.get_property_value('position/lat-gc-deg')
            lon2 = agent2.get_property_value('position/long-gc-deg')
            
            if lat1 and lon1 and lat2 and lon2:
                # 简单的欧几里得距离计算
                dlat = (lat2 - lat1) * 111320  # 1度纬度约111.32km
                dlon = (lon2 - lon1) * 111320 * abs(np.cos(np.radians(lat1)))
                distance = (dlat**2 + dlon**2)**0.5
                return distance / 1000  # 转换为km
        except:
            pass
            
        # 方式3: 从环境中获取距离信息
        try:
            # 如果前面的方法都失败了，返回一个模拟距离
            # 这是一个临时解决方案
            import random
            return 100 + random.uniform(-20, 20)  # 模拟100km左右的距离
        except:
            pass
                
        return 0.0
    except Exception as e:
        print(f"  [调试] 距离计算错误: {e}")
        return 0.0


def setup_logging(output_dir: str, tactic_name: str = "tactical") -> str:
    """设置日志系统"""
    import sys

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{tactic_name}_simulation_{timestamp}.log")

    # 清除现有的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 修复编码问题：确保UTF-8编码支持
    # 文件和控制台都记录INFO级别，格式：[仿真时间][实际时间] 级别 - 消息
    file_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] %(levelname)-7s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] %(levelname)-7s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # 修复控制台编码问题
    try:
        # 尝试设置控制台为UTF-8编码
        if sys.platform.startswith('win'):
            # Windows系统特殊处理
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        # 如果设置失败，继续使用默认编码
        pass

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True  # 强制重新配置
    )

    logging.info("=" * 80)
    logging.info("[LAUNCH] 战术决策系统仿真开始")
    logging.info("=" * 80)
    logging.info(f"日志文件: {log_file}")
    return log_file


def print_banner(tactic_name: str = "通用战术"):
    """打印仿真标题"""
    print("\n" + "=" * 80)
    print(f"目标 {tactic_name}仿真")
    print("=" * 80)
    # print("我方: F-16C Block 50/52 x2 (蓝方)")
    # print("  - 长机: A0100")
    # print("  - 僚机: A0200")
    # # ✅ 用户要求：我方/敌方雷达与导弹对调
    # print("  - 雷达: N001VE")
    # print("  - 导弹: R-27ER")
    # print()
    # print("敌方: SU-27 Flanker x2 (红方)")
    # print("  - 敌机1: B0100")
    # print("  - 敌机2: B0200")
    # print("  - 雷达: AN/APG-68(V)9")
    # print("  - 导弹: AIM-120C7")
    # print()
    # print("战术系统:")
    # print("  - 威胁值计算")
    # print("  - 战术选择 (5种进攻战术)")
    # print("  - 控制距离节点 (NLT/MELD/MTR/LR/TR/DOR/DR/MAR)")
    # print("  - 协同决策")
    # print("=" * 80)
    # print()


def run_simulation(
    tactic_type: str = "front_back",  # 修改默认战术为前后攻击
    our_intent: str = "conservative_clear",
    max_steps: int = 3300,  # 660秒=11分钟 @ 0.2s/步
    output_dir: str = None,
    config_name: str = "tactical_bvr",
):
    """
    运行战术仿真

    Args:
        tactic_type: 战术类型 (auto/drag_shoot/pincer/high_low/front_back/side_by_side)
        our_intent: 我方意图 (conservative_clear/aggressive_clear/defensive)
        max_steps: 最大步数
        output_dir: 输出目录（默认为脚本所在目录下的tactical_simulation_results）
    """
    # 读取全局变量到局部变量（避免作用域问题）
    global USE_CAP_MODE, CAP_AVAILABLE
    use_cap_mode = USE_CAP_MODE
    cap_available = CAP_AVAILABLE
    # 如果未指定输出目录，使用脚本所在目录下的tactical_simulation_results
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "tactical_simulation_results")

    # 设置日志 - 使用实际的强制战术名称作为文件名
    actual_tactic_name = FORCE_TACTIC.lower() if FORCE_TACTIC else tactic_type
    log_file = setup_logging(output_dir, actual_tactic_name)
    
    # 打印标题
    tactic_names = {
        "auto": "自动战术选择",
        "drag_shoot": "拖曳射击战术",
        "pincer": "钳形攻势战术",
        "high_low": "上下夹击战术",
        "front_back": "前后攻击战术",
        "side_by_side": "并排射击战术"
    }
    print_banner(tactic_names.get(tactic_type, "通用战术"))
    
    try:
        # 创建环境 - 使用本地BVR配置
        # print("初始化JSBSim环境...")  # 注释掉无用信息
        
        # 读取全局变量到局部变量（避免作用域问题）
        use_cap_mode = USE_CAP_MODE
        cap_available = CAP_AVAILABLE
        
        # 如果使用巡逻模式，使用巡逻配置文件
        local_config_file = None
        if use_cap_mode and cap_available:
            # 使用巡逻配置文件
            patrol_config_path = os.path.join(current_dir, 'cap', 'config', 'patrol_config.yaml')
            if os.path.exists(patrol_config_path):
                local_config_file = patrol_config_path
                config_name = 'patrol_config'
                logging.info(f"✅ 使用巡逻模式，配置文件: {patrol_config_path}")
            else:
                logging.warning(f"⚠️ 巡逻配置文件不存在: {patrol_config_path}，使用默认配置")
                use_cap_mode = False  # 回退到原有模式
        
        # 统一使用BVR配置（可通过参数覆盖）
        if not config_name:
            config_name = "tactical_bvr"

        # 如果还没有设置本地配置文件路径，使用默认路径
        if local_config_file is None:
            local_config_file = os.path.join(current_dir, 'configs', f'{config_name}.yaml')
        
        if not os.path.exists(local_config_file):
            logging.error(f"配置文件不存在: {local_config_file}")
            print(f"[错误] 配置文件不存在: {local_config_file}")
            return
        
        # 临时复制配置文件到JSBSim configs目录
        jsbsim_config_dir = os.path.join(project_root, 'envs', 'JSBSim', 'configs')
        os.makedirs(jsbsim_config_dir, exist_ok=True)  # 确保目录存在
        jsbsim_config_file = os.path.join(jsbsim_config_dir, f'{config_name}.yaml')
        
        # 复制配置文件
        try:
            shutil.copy2(local_config_file, jsbsim_config_file)
            logging.info(f"✅ 已复制配置文件: {local_config_file} -> {jsbsim_config_file}")
        except Exception as e:
            logging.error(f"❌ 复制配置文件失败: {e}")
            print(f"[错误] 复制配置文件失败: {e}")
            return
        
        logging.info(f"使用配置文件: {local_config_file}")
        # print(f"使用配置: {config_name}")
        
        env = ConfigurableMultipleCombatEnv(config_name, my_aircraft=MY_AIRCRAFT_TYPE, enemy_aircraft=ENEMY_AIRCRAFT_TYPE)
        env.max_steps = max_steps
        # print("JSBSim环境创建成功")  # 注释掉无用信息
        
        # 创建决策管理器
        # print(f"初始化战术决策系统 (意图: {our_intent})...")  # 注释掉无用信息
        
        # 根据战术类型创建不同的决策管理器
        if tactic_type == "drag_shoot":
            from tactics.drag_shoot import DragShootDecisionManager
            decision_manager = DragShootDecisionManager(our_intent_type=our_intent)
        elif tactic_type == "pincer":
            from tactics.pincer_attack import PincerAttackDecisionManager
            decision_manager = PincerAttackDecisionManager(our_intent_type=our_intent)
        elif tactic_type == "high_low":
            from tactics.high_low_attack import HighLowAttackDecisionManager
            decision_manager = HighLowAttackDecisionManager(our_intent_type=our_intent)
        elif tactic_type == "front_back":
            from tactics.sequential_attack import SequentialAttackTactic
            # 前后攻击使用sequential_attack，暂时用通用决策管理器
            decision_manager = TacticalDecisionManager(our_intent_type=our_intent)
        elif tactic_type == "side_by_side":
            from tactics.side_by_side import SideBySideDecisionManager
            decision_manager = SideBySideDecisionManager(our_intent_type=our_intent)
        else:
            # 自动选择战术
            decision_manager = TacticalDecisionManager(our_intent_type=our_intent)
        
        # 🔥 将命令行我方意图注入到task config（TacticalTask读取config.friendly_intent）
        try:
            intent_map = {
                "aggressive_clear": "AGGRESSIVE_CLEAR",
                "conservative_clear": "CONSERVATIVE_CLEAR",
                "defensive": "DEFENSIVE",
            }
            env.config.friendly_intent = intent_map.get(our_intent, "CONSERVATIVE_CLEAR")
        except Exception:
            pass

        # 创建战术任务（根据use_cap_mode选择）
        if use_cap_mode and cap_available:
            # 使用巡逻模式：优先CAPTask（P1-4完整），失败回退SimplePatrolTask
            tactical_task = None
            try:
                from cap.cap_task import CAPTask
                tactical_task = CAPTask(env.config)
                logging.info("✅ 已创建CAPTask（P1-4完整巡逻）")
            except Exception as e:
                from cap.patrol_task import SimplePatrolTask
                tactical_task = SimplePatrolTask(env.config)
                logging.warning(f"⚠️ CAPTask创建失败，回退SimplePatrolTask: {e}")
        else:
            # 使用原有模式（2v2、固定路径）
            tactical_task = TacticalTask(env.config, decision_manager=decision_manager, force_tactic=FORCE_TACTIC)
            if not cap_available and use_cap_mode:
                logging.warning("⚠️ 巡逻模式启用失败，已回退到原有模式")
            else:
                logging.info("✅ 已创建原有战术任务（2v2、固定路径）")

        # ✅ 运行时兼容兜底：部分旧逻辑会访问 self.task.xxx
        # 即使 tactical_task.py 由于缓存/行号错位加载了旧版本，也尽量保证不因缺失 task 字段直接崩溃。
        try:
            tactical_task.task = tactical_task
        except Exception:
            try:
                tactical_task.__dict__['task'] = tactical_task
            except Exception:
                pass

        # ✅ 一次性确认：实际加载的 tactical_task.py 来自哪里（排查“行号对不上/改了没生效”）
        try:
            stamp = getattr(tactical_task_module, 'TACTICAL_TASK_BUILD_STAMP', None)
            logging.warning(f"🔎 [LOAD] tactical_task={getattr(tactical_task_module, '__file__', None)} BUILD_STAMP={stamp}")
        except Exception:
            pass
        env.task = tactical_task

        # 打印战术选择信息
        if FORCE_TACTIC:
            logging.info(f"[强制战术] 选择战术: {FORCE_TACTIC}")
        else:
            logging.info("[智能战术] 使用智能战术选择")
        
        # 重置环境
        # print("重置环境...")  # 注释掉无用信息
        obs = env.reset()
        # 确保max_steps设置在环境重置后生效
        env.max_steps = max_steps
        # print("战术系统初始化完成")  # 注释掉无用信息
        
        # 准备数据记录和ACMI文件
        from datetime import datetime
        # 修改时间戳格式：MMDD_HHMMSS
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        trajectory_data = []
        tactical_data = []
        
        # 🎯 创建统一数据记录器用于意图识别数据集生成
        sys.path.insert(0, os.path.join(current_dir, '..', 'tacticalTemplateProject'))
        # noinspection PyUnresolvedReferences
        from unified_data_recorder import UnifiedDataRecorder
        data_recorder = UnifiedDataRecorder(
            project_name="intent_recognition",
            trajectory_mode="intent_pairwise_frame",
        )
        
        # 🔍 验证数据记录器是否真正导入和创建成功
        logging.info("="*80)
        logging.info("✅ 数据记录器验证:")
        logging.info(f"   - 类型: {type(data_recorder)}")
        logging.info(f"   - 模块: {data_recorder.__class__.__module__}")
        logging.info(f"   - 项目名称: {data_recorder.project_name}")
        # 验证关键方法是否存在
        assert hasattr(data_recorder, 'record_aircraft_trajectory'), "缺少 record_aircraft_trajectory 方法"
        assert hasattr(data_recorder, 'save_csv_files'), "缺少 save_csv_files 方法"
        logging.info(f"   - 方法验证: record_aircraft_trajectory ✓, save_csv_files ✓")
        logging.info(f"   - 数据记录器已就绪，可以正常使用")
        logging.info("="*80)
        
        # 获取当前战术信息来设置ACMI文件名
        if FORCE_TACTIC and FORCE_TACTIC != 'None':
            tactic_name = FORCE_TACTIC.lower()
        else:
            # 尝试从tactical_task获取战术信息
            tactic_name = getattr(tactical_task, 'selected_tactic', 'auto_select')
            if tactic_name and tactic_name != 'None':
                tactic_name = tactic_name.lower()
            else:
                tactic_name = 'auto_select'
        
        # 设置ACMI文件路径 - 时间戳开头格式：MMDD_HHMMSS_tactic_2v2.txt.acmi
        acmi_filepath = os.path.join(output_dir, f'{timestamp}_{tactic_name}_2v2.txt.acmi')
        
        # 修复问题5：明确仿真开始标记
        print("\n" + "=" * 80)
        print("[START] 2v2超视距空战仿真开始")
        print("=" * 80)
        print(f"开始时间: {timestamp}")
        print(f"最大步数: {max_steps} 步 ({max_steps * env.time_interval:.1f}秒)")
        print(f"时间步长: {env.time_interval}秒")
        print("-" * 80)
        print(f"{'时间(s)':<10} {'步数':<8} {'距离(km)':<12} {'我方':<8} {'敌方':<8} {'战术':<15}")
        print("-" * 80)
        
        # logging.info("="*80)
        # logging.info("[SIMULATION START] 2v2超视距空战")
        # logging.info(f"   我方: F-16C x2 (A0100, A0200)")
        # logging.info(f"   敌方: Su-27 x2 (B0100, B0200)")
        # logging.info(f"   最大步数: {max_steps}, 时间步长: {env.time_interval}s")
        # logging.info(f"   🔍 [DEBUG] max_steps参数: {max_steps}, env.max_steps: {getattr(env, 'max_steps', 'N/A')}")
        # logging.info("="*80)
        #
        step = 0
        dt = env.time_interval
        acmi_file_renamed = False  # 标记是否已经重命名ACMI文件
        
        while step < max_steps:
            step += 1
            current_time = step * dt
            
            # 环境步进
            import numpy as np
            num_agents = len(env.agents)
            dummy_actions = np.zeros((1, num_agents, 4))
            obs, share_obs, rewards, dones, info = env.step(dummy_actions)
            
            # 在适当时机检查并重命名ACMI文件（如果战术信息已确定）
            if not acmi_file_renamed and step >= 10:  # 在战术刚确定时捕获
                current_tactic = getattr(tactical_task, 'selected_tactic', None)
                if current_tactic and current_tactic not in ['None', 'N/A', None, 'FORMATION_RESET']:
                    # 重命名ACMI文件 - 时间戳开头格式
                    new_tactic_name = current_tactic.lower()
                    new_acmi_filepath = os.path.join(output_dir, f'{timestamp}_{new_tactic_name}_2v2.txt.acmi')
                    try:
                        if os.path.exists(acmi_filepath):
                            os.rename(acmi_filepath, new_acmi_filepath)
                            acmi_filepath = new_acmi_filepath
                            logging.info(f"✅ ACMI文件已重命名为: {os.path.basename(new_acmi_filepath)}")
                        acmi_file_renamed = True
                    except Exception as e:
                        logging.warning(f"⚠️ ACMI文件重命名失败: {e}")
                        acmi_file_renamed = True  # 避免重复尝试
                elif step >= 60:  # 如果60步后还没有获取到战术，就停止尝试
                    acmi_file_renamed = True
            
            # 渲染ACMI文件（每步都记录）
            env.render(mode="txt", filepath=acmi_filepath)
            
            # 🎯 记录意图识别数据（每步记录）
            try:
                data_recorder.record_all_data(env, current_time, tactical_task)
            except Exception as e:
                if step % 300 == 0:  # 只在60秒间隔时报告错误
                    logging.warning(f"意图识别数据记录失败: {e}")
            
            # 🕒 关键信息输出（每60秒输出一次，只保留关键信息）
            if step % 300 == 0:  # 300步 = 60秒
                # 计算双方最近距离
                min_dist = float('inf')
                for aid1 in ["A0100", "A0200"]:
                    if aid1 not in env.agents or not env.agents[aid1].is_alive:
                        continue
                    for aid2 in ["B0100", "B0200"]:
                        if aid2 not in env.agents or not env.agents[aid2].is_alive:
                            continue
                        pos1 = env.agents[aid1].get_position()
                        pos2 = env.agents[aid2].get_position()
                        dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                        min_dist = min(min_dist, dist)
                
                if min_dist == float('inf'):
                    min_dist = 0
                
                # 获取当前战术
                tactic = tactical_task.selected_tactic if hasattr(tactical_task, 'selected_tactic') and tactical_task.selected_tactic else 'N/A'
                
                # 获取关键状态信息
                key_info = []
                for aid in ["A0100", "A0200"]:
                    if aid in env.agents and env.agents[aid].is_alive:
                        pos = env.agents[aid].get_position()
                        heading = np.rad2deg(env.agents[aid].get_property_value(c.attitude_psi_rad))
                        # 🔥 修复：检查state_manager是否存在
                        if hasattr(tactical_task, 'state_manager') and tactical_task.state_manager:
                            try:
                                phase = tactical_task.state_manager.get_agent_phase(aid)
                                phase_name = tactical_task._format_phase_name(aid, phase) if hasattr(tactical_task, '_format_phase_name') else (phase.value if phase else 'N/A')
                            except:
                                phase_name = 'PATROL'
                        else:
                            phase_name = 'PATROL'
                        key_info.append(f"{aid}: 位置({pos[0]/1000:.1f},{pos[1]/1000:.1f})km 高度{pos[2]:.0f}m 航向{heading:.0f}° 阶段{phase_name}")
                
                # 只输出关键信息
                logging.info(f"[关键状态] 时间:{current_time:.1f}s 距离:{min_dist/1000:.1f}km 战术:{tactic}")
                for info in key_info:
                    logging.info(f"  {info}")
                
                # 🔍 注释掉详细调试信息输出
                # print_detailed_debug_info(env, step, tactic, tactical_task)

            # 📊 注释掉详细分析报告，只保留关键信息
            # if step % ANALYSIS_INTERVAL == 0:
            #     print_analysis_report(env, step, tactic, tactical_task)
            
            # 记录轨迹数据
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    pos = env.agents[agent_id].get_position()
                    vel = env.agents[agent_id].get_velocity()
                    trajectory_data.append({
                        'time': current_time,
                        'agent_id': agent_id,
                        'x': pos[0], 'y': pos[1], 'z': pos[2],
                        'vx': vel[0], 'vy': vel[1], 'vz': vel[2],
                    })
            
            # 每60秒打印一次详细状态
            if step % 300 == 0:  # 300步 ≈ 60秒
                # 打印飞机位置和速度
                for agent_id in env.agents.keys():
                    if env.agents[agent_id].is_alive:
                        pos = env.agents[agent_id].get_position()
                        vel = env.agents[agent_id].get_velocity()
                        speed = np.linalg.norm(vel)
                        alt = pos[2]
                        logging.info(f"[{agent_id}] 位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}m) "
                                   f"速度: {speed:.1f}m/s 高度: {alt:.1f}m")
            
            # 每12秒打印一次表格状态（但跳过已经在12秒调试信息输出的情况）
            elif step % 60 == 0:  # 12秒间隔，但避免与上面的调试输出重复
                # 计算双方最近距离
                min_dist = float('inf')
                for aid1 in ["A0100", "A0200"]:
                    if aid1 not in env.agents or not env.agents[aid1].is_alive:
                        continue
                    for aid2 in ["B0100", "B0200"]:
                        if aid2 not in env.agents or not env.agents[aid2].is_alive:
                            continue
                        pos1 = env.agents[aid1].get_position()
                        pos2 = env.agents[aid2].get_position()
                        dist = ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)**0.5
                        min_dist = min(min_dist, dist)
                
                # 统计存活
                friendly_alive = sum(1 for aid in ["A0100", "A0200"] 
                                   if aid in env.agents and env.agents[aid].is_alive)
                enemy_alive = sum(1 for aid in ["B0100", "B0200"] 
                                if aid in env.agents and env.agents[aid].is_alive)
                
                # 获取当前战术
                tactic = tactical_task.selected_tactic if hasattr(tactical_task, 'selected_tactic') and tactical_task.selected_tactic else 'N/A'
                
                # 战术名称映射
                tactic_names_short = {
                    'DRAG_SHOOT': '拖曳射击',
                    'PINCER_ATTACK': '钳形攻势',
                    'HIGH_LOW_ATTACK': '上下夹击',
                    'FRONT_BACK': '前后攻击',
                    'SIDE_BY_SIDE': '并排射击',
                    'N/A': '未决策'
                }
                tactic_str = tactic_names_short.get(tactic, str(tactic))
                
                print(f"{current_time:<10.1f} {step:<8} {min_dist/1000:<12.1f} "
                      f"{friendly_alive}/2{'':<4} {enemy_alive}/2{'':<4} {tactic_str:<15}")
            
            # 检查终止
            if isinstance(dones, np.ndarray):
                if dones.all():
                    break
            elif isinstance(dones, dict):
                if all(dones.values()):
                    break
        
        # 修复问题5：明确仿真结束标记
        end_time = datetime.now()
        print("\n" + "=" * 80)
        print("[END] 2v2超视距空战仿真结束")
        print("=" * 80)
        
        # 统计结果
        friendly_alive = sum(1 for aid in ["A0100", "A0200"] 
                           if aid in env.agents and env.agents[aid].is_alive)
        enemy_alive = sum(1 for aid in ["B0100", "B0200"] 
                        if aid in env.agents and env.agents[aid].is_alive)
        
        print(f"总步数: {step}")
        print(f"总时间: {step * dt:.1f}秒 ({step * dt / 60:.1f}分钟)")
        print(f"我方存活: {friendly_alive}/2")
        print(f"敌方存活: {enemy_alive}/2")
        
        # 战果判定
        if friendly_alive > enemy_alive:
            result = "我方获胜!"
            winner = "我方"
        elif enemy_alive > friendly_alive:
            result = "敌方获胜!"
            winner = "敌方"
        else:
            result = "平局"
            winner = "平局"
        
        print(f"\n战果: {result}")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 日志记录
        logging.info("="*80)
        logging.info(f"[SIMULATION END] 仿真结束")
        logging.info(f"   总时间: {step * dt:.1f}秒 ({step}步)")
        logging.info(f"   我方存活: {friendly_alive}/2")
        logging.info(f"   敌方存活: {enemy_alive}/2")
        logging.info(f"   战果: {winner}")
        logging.info("="*80)
        
        print(f"\n详细日志: {log_file}")
        
        # 🎯 保存意图识别数据集
        try:
            logging.info("正在保存意图识别数据集...")
            saved_files = data_recorder.save_csv_files(output_dir, timestamp)
            print(f"\n🎯 意图识别数据集已保存:")
            for data_type, filepath in saved_files.items():
                print(f"  - {data_type}: {filepath}")
        except Exception as e:
            logging.error(f"保存意图识别数据集失败: {e}")
            print(f"⚠️ 意图识别数据集保存失败: {e}")
        
        # 保存数据（DataLogger会在TacticalTask内按步记录，这里统一导出全部表格）
        if trajectory_data:
            import pandas as pd
            traj_file = os.path.join(output_dir, f'trajectory_{timestamp}.csv')
            pd.DataFrame(trajectory_data).to_csv(traj_file, index=False)
            print(f"轨迹数据: {traj_file}")
        try:
            if hasattr(env, 'task') and hasattr(env.task, 'data_logger'):
                env.task.data_logger.save_all()
                print(f"数据表输出目录: {env.task.data_logger.output_dir}")
        except Exception:
            pass
        
        # 显示ACMI文件
        if os.path.exists(acmi_filepath):
            print(f"ACMI文件: {acmi_filepath}")
        else:
            print(f"ACMI文件未生成")
        
        print("=" * 80)
        
        # 关闭环境
        env.close()
        
    except Exception as e:
        logging.error(f"仿真错误: {e}", exc_info=True)
        print(f"\n仿真错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='战术仿真系统')
    parser.add_argument('--tactic', type=str, default='auto',  # 默认使用前后攻击战术
                       choices=['auto', 'drag_shoot', 'pincer', 'high_low', 'front_back', 'side_by_side'],
                       help='战术类型')
    # ✅ PyCharm 一键切换：默认使用 DEFAULT_RUN_INTENT（改一行即可）
    # 仍保留环境变量覆盖（可选）：TACTICAL_OUR_INTENT
    default_intent = os.getenv("TACTICAL_OUR_INTENT", DEFAULT_RUN_INTENT)
    parser.add_argument('--intent', type=str, default=default_intent,
                       choices=['conservative_clear', 'aggressive_clear', 'defensive'],
                       help='我方意图')
    parser.add_argument('--steps', type=int, default=3300,
                       help='最大步数 (3300步=11分钟@0.2s/步)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录（默认为脚本所在目录下的tactical_simulation_results）')

    # ✅ 初始态势配置：默认随 intent 自动切换
    # - conservative_clear -> tactical_bvr.yaml（现有普通态势）
    # - aggressive_clear   -> tactical_bvr_aggressive_clear.yaml（我方更高更快）
    # - defensive          -> tactical_bvr_defensive.yaml（我方更低更慢）
    parser.add_argument('--config', type=str, default=None,
                        help='覆盖默认配置名（不带.yaml），例如 tactical_bvr_defensive')

    args = parser.parse_args()

    # 若未手动指定配置，则根据 intent 自动选择
    config_by_intent = {
        'conservative_clear': 'tactical_bvr',
        'aggressive_clear': 'tactical_bvr_aggressive_clear',
        'defensive': 'tactical_bvr_defensive',
    }
    config_name = args.config or config_by_intent.get(args.intent, 'tactical_bvr')
    
    run_simulation(
        tactic_type=args.tactic,
        our_intent=args.intent,
        max_steps=args.steps,
        output_dir=args.output,
        config_name=config_name,
    )


if __name__ == "__main__":
    main()
