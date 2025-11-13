"""
完整威胁评估模块 - 实现项目说明.md 6.3节的威胁评估算法

威胁等级计算基于多源传感器信息：
tl = w1·RWR_THREAT + w2·MISSILE_THREAT + w3·RADAR_THREAT + w4·RANGE_THREAT

权重：w1=0.4, w2=0.3, w3=0.2, w4=0.1
所有威胁分量归一化至[0,1]
"""
import logging
import numpy as np


class CompleteThreatEvaluator:
    """完整威胁评估器"""
    
    def __init__(self):
        # 威胁权重（项目说明.md 6.3节）
        self.w_rwr = 0.4
        self.w_missile = 0.3
        self.w_radar = 0.2
        self.w_range = 0.1
        
        logging.info("✅ 完整威胁评估系统初始化完成")
    
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
            # 计算各威胁分量
            rwr_threat = self._calculate_rwr_threat(my_aircraft, enemy_aircraft)
            missile_threat = self._calculate_missile_threat(my_aircraft, env)
            radar_threat = self._calculate_radar_threat(my_aircraft, enemy_aircraft)
            range_threat = self._calculate_range_threat(my_aircraft, enemy_aircraft)
            
            # 加权求和
            total_threat = (
                self.w_rwr * rwr_threat +
                self.w_missile * missile_threat +
                self.w_radar * radar_threat +
                self.w_range * range_threat
            )
            
            return np.clip(total_threat, 0.0, 1.0)
            
        except Exception as e:
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

