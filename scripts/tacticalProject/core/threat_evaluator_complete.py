"""
完整威胁评估模块 - 实现项目说明.md 6.3节的威胁评估算法

威胁等级计算基于多源传感器信息：
tl = w1·RWR_THREAT + w2·MISSILE_THREAT + w3·RADAR_THREAT + w4·RANGE_THREAT

权重：w1=0.4, w2=0.3, w3=0.2, w4=0.1
所有威胁分量归一化至[0,1]
"""
import logging
import numpy as np

try:
    from utils.trace_logger import trace_throttle, trace_event
except Exception:  # pragma: no cover
    trace_throttle = None
    trace_event = None


class CompleteThreatEvaluator:
    """完整威胁评估器"""
    
    def __init__(self):
        # 威胁权重（项目说明.md 6.3节）
        self.w_rwr = 0.4
        self.w_missile = 0.3
        self.w_radar = 0.2
        self.w_range = 0.1
        
        if trace_event is not None:
            trace_event(
                事件="威胁评估-初始化",
                模块="threat_evaluator_complete",
                类型="THREAT",
                状态="READY",
                说明="CompleteThreatEvaluator就绪（tl=0.4*RWR+0.3*MISSILE+0.2*RADAR+0.1*RANGE）",
                数据={"w_rwr": self.w_rwr, "w_missile": self.w_missile, "w_radar": self.w_radar, "w_range": self.w_range},
            )
        else:
            logging.info("✅ 完整威胁评估系统初始化完成")

    def calculate_total_threat_detail(self, my_aircraft, enemy_aircraft, env):
        """返回威胁总分的可解释细节（用于日志/答辩复盘）。"""
        rwr_threat = self._calculate_rwr_threat(my_aircraft, enemy_aircraft)
        missile_threat = self._calculate_missile_threat(my_aircraft, env)
        radar_threat = self._calculate_radar_threat(my_aircraft, enemy_aircraft)
        range_threat = self._calculate_range_threat(my_aircraft, enemy_aircraft)

        contrib_rwr = self.w_rwr * rwr_threat
        contrib_missile = self.w_missile * missile_threat
        contrib_radar = self.w_radar * radar_threat
        contrib_range = self.w_range * range_threat
        total_threat = contrib_rwr + contrib_missile + contrib_radar + contrib_range
        total_threat = float(np.clip(total_threat, 0.0, 1.0))

        return {
            "formula": "tl = 0.4*RWR + 0.3*MISSILE + 0.2*RADAR + 0.1*RANGE",
            "weights": {
                "rwr": float(self.w_rwr),
                "missile": float(self.w_missile),
                "radar": float(self.w_radar),
                "range": float(self.w_range),
            },
            "parts": {
                "rwr": float(rwr_threat),
                "missile": float(missile_threat),
                "radar": float(radar_threat),
                "range": float(range_threat),
            },
            "contrib": {
                "0.4*rwr": float(contrib_rwr),
                "0.3*missile": float(contrib_missile),
                "0.2*radar": float(contrib_radar),
                "0.1*range": float(contrib_range),
            },
            "total": float(total_threat),
        }
    
    def calculate_total_threat(self, my_aircraft, enemy_aircraft, env):
        """
        计算综合威胁等级
        
        Args:
            my_aircraft: 我方飞机
            enemy_aircraft: 敌方飞机
            env: 环境对象（用于获取导弹信息）
            
        Returns:
            float: 威胁等级 [0, 1]
        """
        try:
            detail = self.calculate_total_threat_detail(my_aircraft, enemy_aircraft, env)
            total = float(detail.get("total", 0.0))

            # ✅ 答辩级解释日志：每~10秒输出一次（按步数节流）
            if trace_throttle is not None:
                try:
                    my_id = str(getattr(my_aircraft, "uid", None) or getattr(my_aircraft, "agent_id", None) or "ME")
                    enemy_id = str(getattr(enemy_aircraft, "uid", None) or getattr(enemy_aircraft, "agent_id", None) or "ENEMY")
                    time_interval = float(getattr(env, "time_interval", 0.2) or 0.2)
                    min_steps = max(1, int(round(10.0 / time_interval)))
                    trace_throttle(
                        key=f"threat:formula:{my_id}:{enemy_id}",
                        min_steps=min_steps,
                        标题="威胁评估-总分公式",
                        env=env,
                        模块="threat_evaluator_complete",
                        类型="THREAT",
                        状态=f"tl={total:.3f}",
                        我机=my_id,
                        敌机=enemy_id,
                        说明="输出tl计算过程：分项(0~1)+权重+加权贡献",
                        数据=detail,
                    )
                except Exception:
                    pass

            return total
            
        except Exception as e:
            if trace_event is not None:
                trace_event(
                    事件="威胁评估-异常",
                    env=env,
                    模块="threat_evaluator_complete",
                    类型="THREAT",
                    状态="EXCEPTION",
                    说明="威胁评估发生异常，返回默认0.5",
                    数据={"exception": str(e)},
                )
            else:
                logging.error(f"威胁评估错误: {e}")
            return 0.5
    
    def _calculate_rwr_threat(self, my_aircraft, enemy_aircraft):
        """
        计算RWR告警威胁度
        
        基于项目说明.md表3.2的RWR威胁等级定义：
        - 0级：无威胁（未被探测）
        - 1级：被搜索（SEARCH模式）
        - 2级：被跟踪（TRACK/TWS模式）
        - 3级：被锁定（LOCK/STT模式）
        - 4级：导弹发射
        - 5级：导弹制导
        
        Returns:
            float: RWR威胁度 [0, 1]
        """
        try:
            if not enemy_aircraft or not enemy_aircraft.is_alive:
                return 0.0
            
            # 获取敌方雷达状态
            radar_mode = getattr(enemy_aircraft, 'radar_mode', 'SEARCH')
            
            # 根据雷达模式确定RWR威胁等级
            if radar_mode == 'LOCK' or radar_mode == 'STT':
                rwr_level = 3  # 被锁定
            elif radar_mode == 'TRACK' or radar_mode == 'TWS':
                rwr_level = 2  # 被跟踪
            elif radar_mode == 'SEARCH':
                # 检查是否在雷达探测范围内
                my_pos = my_aircraft.get_position()
                enemy_pos = enemy_aircraft.get_position()
                distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
                
                # 假设雷达探测范围120km
                if distance < 120000:
                    rwr_level = 1  # 被搜索
                else:
                    rwr_level = 0  # 未被探测
            else:
                rwr_level = 0
            
            # 归一化到[0, 1]（RWR等级0-5）
            rwr_threat = rwr_level / 5.0
            
            return rwr_threat
            
        except Exception as e:
            if trace_event is not None:
                trace_event(
                    事件="威胁评估-RWR异常",
                    模块="threat_evaluator_complete",
                    类型="THREAT",
                    状态="EXCEPTION",
                    数据={"exception": str(e)},
                )
            else:
                logging.error(f"RWR威胁评估错误: {e}")
            return 0.0
    
    def _calculate_missile_threat(self, my_aircraft, env):
        """
        计算导弹威胁度
        
        考虑因素：
        - 是否有导弹来袭
        - 导弹相对距离
        - 导弹接近速率
        
        Returns:
            float: 导弹威胁度 [0, 1]
        """
        try:
            my_uid = my_aircraft.uid
            my_pos = my_aircraft.get_position()
            
            max_missile_threat = 0.0
            
            # 检查所有导弹
            if hasattr(env, 'missiles'):
                for missile_id, missile in env.missiles.items():
                    if not missile.is_alive:
                        continue
                    
                    # 检查导弹目标是否是我机
                    if hasattr(missile, 'target_aircraft') and missile.target_aircraft:
                        if missile.target_aircraft.uid == my_uid:
                            # 计算导弹距离
                            missile_pos = missile.get_position()
                            distance = np.linalg.norm(np.array(missile_pos) - np.array(my_pos))
                            
                            # 导弹威胁度与距离负相关
                            # 20km以内高威胁，40km以外低威胁
                            if distance < 20000:
                                missile_threat = 1.0
                            elif distance < 40000:
                                missile_threat = 1.0 - (distance - 20000) / 20000
                            else:
                                missile_threat = 0.0
                            
                            max_missile_threat = max(max_missile_threat, missile_threat)
            
            return max_missile_threat
            
        except Exception as e:
            if trace_event is not None:
                trace_event(
                    事件="威胁评估-导弹异常",
                    模块="threat_evaluator_complete",
                    类型="THREAT",
                    状态="EXCEPTION",
                    数据={"exception": str(e)},
                )
            else:
                logging.error(f"导弹威胁评估错误: {e}")
            return 0.0
    
    def _calculate_radar_threat(self, my_aircraft, enemy_aircraft):
        """
        计算敌方雷达威胁度
        
        考虑因素：
        - 雷达工作模式
        - 照射时间
        - 信号强度（距离）
        
        Returns:
            float: 雷达威胁度 [0, 1]
        """
        try:
            if not enemy_aircraft or not enemy_aircraft.is_alive:
                return 0.0
            
            # 获取雷达状态
            radar_mode = getattr(enemy_aircraft, 'radar_mode', 'SEARCH')
            
            # 计算距离
            my_pos = my_aircraft.get_position()
            enemy_pos = enemy_aircraft.get_position()
            distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
            distance_km = distance / 1000.0

            # 雷达模式威胁
            if radar_mode == 'LOCK' or radar_mode == 'STT':
                mode_threat = 1.0
            elif radar_mode == 'TRACK' or radar_mode == 'TWS':
                mode_threat = 0.6
            elif radar_mode == 'SEARCH':
                mode_threat = 0.2
            else:
                mode_threat = 0.0

            # 距离威胁（距离越近威胁越大）
            if distance_km < 40:
                distance_threat = 1.0
            elif distance_km < 80:
                distance_threat = 1.0 - (distance_km - 40) / 40
            else:
                distance_threat = 0.0

            # 综合雷达威胁
            radar_threat = 0.7 * mode_threat + 0.3 * distance_threat

            return radar_threat

        except Exception as e:
            if trace_event is not None:
                trace_event(
                    事件="威胁评估-雷达异常",
                    模块="threat_evaluator_complete",
                    类型="THREAT",
                    状态="EXCEPTION",
                    数据={"exception": str(e)},
                )
            else:
                logging.error(f"雷达威胁评估错误: {e}")
            return 0.0

    def _calculate_range_threat(self, my_aircraft, enemy_aircraft):
        """
        计算距离威胁度

        基于敌我距离与武器有效区WEZ位置

        Returns:
            float: 距离威胁度 [0, 1]
        """
        try:
            if not enemy_aircraft or not enemy_aircraft.is_alive:
                return 0.0

            # 计算距离
            my_pos = my_aircraft.get_position()
            enemy_pos = enemy_aircraft.get_position()
            distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
            distance_km = distance / 1000.0

            # 武器有效区（WEZ）定义
            # AIM-120C: Rmax ≈ 100km, Ropt ≈ 40-80km, Rmin ≈ 5km
            # R-27ER: Rmax ≈ 130km, Ropt ≈ 50-90km, Rmin ≈ 5km

            # 最大威胁区：40-80km（双方都在最佳射程内）
            if 40 <= distance_km <= 80:
                range_threat = 1.0
            # 次威胁区：80-100km（我方在射程内）
            elif 80 < distance_km <= 100:
                range_threat = 0.8
            # 低威胁区：100-130km（敌方在射程内，我方不在）
            elif 100 < distance_km <= 130:
                range_threat = 0.4
            # 极近距离：5-40km（导弹可能已发射）
            elif 5 <= distance_km < 40:
                range_threat = 0.9
            # 超近距离：<5km（导弹最小射程内，但可能进入格斗）
            elif distance_km < 5:
                range_threat = 0.5
            # 超远距离：>130km（双方都不在射程内）
            else:
                range_threat = 0.1

            return range_threat

        except Exception as e:
            if trace_event is not None:
                trace_event(
                    事件="威胁评估-距离异常",
                    模块="threat_evaluator_complete",
                    类型="THREAT",
                    状态="EXCEPTION",
                    数据={"exception": str(e)},
                )
            else:
                logging.error(f"距离威胁评估错误: {e}")
            return 0.0

    def detect_incoming_missiles(self, aircraft, env):
        """
        检测来袭导弹（优化版本）

        检测逻辑：
        1. 检查导弹是否存活
        2. 检查导弹目标是否是当前飞机
        3. 检查导弹距离（40km以内认为是威胁）
        4. 检查导弹是否在接近（相对速度为负）

        Args:
            aircraft: 飞机对象
            env: 环境对象

        Returns:
            bool: 是否有导弹来袭
        """
        try:
            aircraft_uid = aircraft.uid
            my_pos = np.array(aircraft.get_position())
            my_vel = np.array(aircraft.get_velocity())

            if hasattr(env, 'missiles') and env.missiles:
                for missile_id, missile in env.missiles.items():
                    # 检查导弹是否存活
                    if not hasattr(missile, 'is_alive') or not missile.is_alive:
                        continue

                    # 检查导弹目标
                    if hasattr(missile, 'target_aircraft') and missile.target_aircraft:
                        if missile.target_aircraft.uid == aircraft_uid:
                            # 计算导弹距离
                            missile_pos = np.array(missile.get_position())
                            distance = np.linalg.norm(missile_pos - my_pos)

                            # 计算相对速度（检查是否在接近）
                            missile_vel = np.array(missile.get_velocity()) if hasattr(missile, 'get_velocity') else np.zeros(3)
                            rel_vel = missile_vel - my_vel
                            rel_pos = missile_pos - my_pos
                            closing_rate = -np.dot(rel_vel, rel_pos) / (distance + 1e-6)

                            # 如果导弹在40km以内且正在接近，认为是威胁
                            if distance < 40000 and closing_rate > 0:
                                logging.warning(f"🚨 [导弹威胁] {aircraft_uid} 检测到来袭导弹 {missile_id}: "
                                              f"距离={distance/1000:.1f}km, 接近速率={closing_rate:.1f}m/s")
                                return True

            return False

        except Exception as e:
            logging.error(f"导弹检测错误: {e}")
            return False

    def calculate_rwr_level(self, aircraft, enemy_aircraft_list, env=None):
        """
        计算RWR告警等级（优化版本）

        RWR等级定义（项目说明.md 表3.2）：
        - 0级: 无威胁（未被探测）
        - 1级: 被搜索（SEARCH模式）
        - 2级: 被跟踪（TRACK/TWS模式）
        - 3级: 被锁定（LOCK/STT模式）
        - 4级: 导弹发射
        - 5级: 导弹制导

        Args:
            aircraft: 我方飞机
            enemy_aircraft_list: 敌方飞机列表
            env: 环境对象（用于检测导弹）

        Returns:
            int: RWR告警等级 [0-5]
        """
        try:
            # ✅ 若仿真模型已提供RWR等级接口，优先使用（更贴近真实系统输出）
            if hasattr(aircraft, "get_rwr_threat_level"):
                try:
                    lvl = int(aircraft.get_rwr_threat_level())
                    return int(np.clip(lvl, 0, 5))
                except Exception:
                    pass

            max_level = 0
            aircraft_uid = aircraft.uid

            # 首先检查是否有导弹来袭（最高优先级）
            if env and hasattr(env, 'missiles') and env.missiles:
                for missile_id, missile in env.missiles.items():
                    if not hasattr(missile, 'is_alive') or not missile.is_alive:
                        continue

                    # 检查导弹目标
                    if hasattr(missile, 'target_aircraft') and missile.target_aircraft:
                        if missile.target_aircraft.uid == aircraft_uid:
                            # 计算导弹距离
                            my_pos = np.array(aircraft.get_position())
                            missile_pos = np.array(missile.get_position())
                            distance = np.linalg.norm(missile_pos - my_pos)

                            # 导弹在50km以内，RWR等级至少为4级
                            if distance < 50000:
                                # 检查导弹是否在制导阶段
                                missile_phase = getattr(missile, '_phase', None)
                                if missile_phase == 2:  # 末制导阶段
                                    max_level = max(max_level, 5)
                                    logging.warning(f"🚨 [RWR-5级] {aircraft_uid} 导弹制导中: {missile_id}, 距离={distance/1000:.1f}km")
                                else:
                                    max_level = max(max_level, 4)
                                    logging.warning(f"🚨 [RWR-4级] {aircraft_uid} 导弹来袭: {missile_id}, 距离={distance/1000:.1f}km")

            # 检查敌机雷达威胁
            for enemy in enemy_aircraft_list:
                if not enemy or not enemy.is_alive:
                    continue

                # 获取雷达状态
                radar_mode = getattr(enemy, 'radar_mode', 'SEARCH')

                # 计算距离
                my_pos = aircraft.get_position()
                enemy_pos = enemy.get_position()
                distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))

                # 根据雷达模式和距离确定RWR等级
                if radar_mode == 'LOCK' or radar_mode == 'STT':
                    if distance < 80000:  # 80km以内被锁定
                        level = 3
                    else:
                        level = 2
                elif radar_mode == 'TRACK' or radar_mode == 'TWS':
                    if distance < 100000:  # 100km以内被跟踪
                        level = 2
                    else:
                        level = 1
                elif radar_mode == 'SEARCH':
                    if distance < 120000:  # 120km以内被搜索
                        level = 1
                    else:
                        level = 0
                else:
                    level = 0

                max_level = max(max_level, level)

            return max_level

        except Exception as e:
            logging.error(f"RWR等级计算错误: {e}")
            return 0
    
    # ===== 兼容性方法 - 与基础ThreatEvaluator接口保持一致 =====
    
    def calculate_angle_threat(self, my_aircraft, enemy_aircraft):
        """
        计算角度威胁值（兼容性方法）
        
        考虑：
        - 敌机是否在我机前方
        - 我机是否在敌机雷达照射范围内
        - 相对角度优势
        
        Returns:
            float: 0-1之间的威胁值
        """
        try:
            my_pos = np.array(my_aircraft.get_position())
            enemy_pos = np.array(enemy_aircraft.get_position())
            
            # 获取航向
            from envs.JSBSim.core.catalog import Catalog as c
            my_heading = my_aircraft.get_property_value(c.attitude_psi_rad)
            enemy_heading = enemy_aircraft.get_property_value(c.attitude_psi_rad)
            
            # 计算相对位置向量
            rel_pos = enemy_pos - my_pos
            rel_pos_2d = rel_pos[:2]  # 只考虑水平面
            
            # 计算敌机相对于我机的方位角
            angle_to_enemy = np.arctan2(rel_pos_2d[1], rel_pos_2d[0])
            
            # 计算我机航向与敌机方位的夹角
            my_angle_diff = abs(self._normalize_angle(angle_to_enemy - my_heading))
            
            # 计算敌机航向与我机方位的夹角
            angle_to_me = np.arctan2(-rel_pos_2d[1], -rel_pos_2d[0])
            enemy_angle_diff = abs(self._normalize_angle(angle_to_me - enemy_heading))
            
            # 角度威胁：敌机在我前方且我在敌机前方时威胁最大
            my_angle_threat = 1.0 - (my_angle_diff / np.pi)  # 0度=1.0, 180度=0.0
            enemy_angle_threat = 1.0 - (enemy_angle_diff / np.pi)
            
            # 综合角度威胁
            angle_threat = 0.4 * my_angle_threat + 0.6 * enemy_angle_threat
            
            return np.clip(angle_threat, 0.0, 1.0)
            
        except Exception as e:
            logging.error(f"计算角度威胁值错误: {e}")
            return 0.5
    
    def calculate_altitude_threat(self, my_aircraft, enemy_aircraft):
        """
        计算高度威胁值（兼容性方法）
        
        考虑：
        - 高度差
        - 敌机是否占据高度优势
        
        Returns:
            float: 0-1之间的威胁值
        """
        try:
            my_alt = my_aircraft.get_position()[2]
            enemy_alt = enemy_aircraft.get_position()[2]
            
            # 高度差
            alt_diff = enemy_alt - my_alt
            
            # 高度威胁：敌机高度越高威胁越大
            # 高度差在-2000m到+2000m之间映射到0-1
            altitude_threat = (alt_diff + 2000) / 4000
            
            return np.clip(altitude_threat, 0.0, 1.0)
            
        except Exception as e:
            logging.error(f"计算高度威胁值错误: {e}")
            return 0.5
    
    def evaluate_situation(self, env):
        """
        评估整体态势（兼容性方法）
        
        Returns:
            dict: 包含各机威胁值和态势评估的字典
        """
        try:
            # 我方飞机
            my_aircraft = [env._jsbsims.get("A0100"), env._jsbsims.get("A0200")]
            enemy_aircraft = [env._jsbsims.get("B0100"), env._jsbsims.get("B0200")]
            
            # 计算长机威胁（敌机对A0100的威胁）
            lead_threats = []
            for enemy_ac in enemy_aircraft:
                if enemy_ac and enemy_ac.is_alive and my_aircraft[0] and my_aircraft[0].is_alive:
                    threat = self.calculate_total_threat(my_aircraft[0], enemy_ac, env)
                    lead_threats.append(threat)
            
            # 计算僚机威胁（敌机对A0200的威胁）
            wingman_threats = []
            for enemy_ac in enemy_aircraft:
                if enemy_ac and enemy_ac.is_alive and my_aircraft[1] and my_aircraft[1].is_alive:
                    threat = self.calculate_total_threat(my_aircraft[1], enemy_ac, env)
                    wingman_threats.append(threat)
            
            # 计算平均威胁
            lead_threat = max(lead_threats) if lead_threats else 0.5
            wingman_threat = max(wingman_threats) if wingman_threats else 0.5
            avg_threat = (lead_threat + wingman_threat) / 2
            
            # 态势判断
            situation = 'ADVANTAGE' if avg_threat < 0.5 else 'DISADVANTAGE'
            
            return {
                'lead_threat': lead_threat,
                'wingman_threat': wingman_threat,
                'avg_threat': avg_threat,
                'situation': situation
            }
                
        except Exception as e:
            logging.error(f"评估态势错误: {e}")
            return {
                'lead_threat': 0.5,
                'wingman_threat': 0.5,
                'avg_threat': 0.5,
                'situation': 'ADVANTAGE'
            }
    
    def _normalize_angle(self, angle):
        """标准化角度到[-π, π]（兼容性方法）"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

