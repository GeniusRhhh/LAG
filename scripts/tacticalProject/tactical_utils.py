"""
战术工具函数模块
提供计算距离、角度归一化、坐标转换、CAP边界检查等工具函数
从tactical_task.py中提取，提高代码可维护性
"""
import logging
import numpy as np
from envs.JSBSim.core.catalog import Catalog as c


class TacticalUtils:
    """战术工具函数类"""
    
    @staticmethod
    def calculate_distance_between(aircraft1, aircraft2) -> float:
        """计算两架飞机之间的3D距离（米）"""
        if aircraft1 is None or aircraft2 is None:
            return float('inf')
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    @staticmethod
    def calculate_horizontal_distance(aircraft1, aircraft2) -> float:
        """计算两架飞机之间的水平距离（米，忽略高度差）"""
        if aircraft1 is None or aircraft2 is None:
            return float('inf')
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return np.sqrt(dx**2 + dy**2)
    
    @staticmethod
    def normalize_angle_diff(angle_diff: float) -> float:
        """
        标准化角度差到 [-180, 180] 区间
        
        Args:
            angle_diff: 角度差（度）
            
        Returns:
            标准化后的角度差（度）
        """
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff
    
    @staticmethod
    def convert_heading_to_index(heading_diff: float) -> int:
        """
        将航向变化（弧度）转换为离散指令索引
        
        Args:
            heading_diff: 航向变化量（弧度）
            
        Returns:
            航向指令索引 (0-16):
                0-3: 小幅左转
                4: 中幅左转
                5-7: 大幅左转
                8: 保持航向
                9-11: 小幅右转
                12: 中幅右转
                13-16: 大幅右转
        """
        heading_diff_deg = np.rad2deg(heading_diff)
        
        # 离散化航向指令
        if heading_diff_deg < -120:
            return 0  # 大幅左转
        elif heading_diff_deg < -90:
            return 1
        elif heading_diff_deg < -60:
            return 2
        elif heading_diff_deg < -45:
            return 3
        elif heading_diff_deg < -30:
            return 4  # 中幅左转
        elif heading_diff_deg < -20:
            return 5
        elif heading_diff_deg < -10:
            return 6  # 小幅左转
        elif heading_diff_deg < -5:
            return 7
        elif heading_diff_deg < 5:
            return 8  # 保持航向
        elif heading_diff_deg < 10:
            return 9
        elif heading_diff_deg < 20:
            return 10  # 小幅右转
        elif heading_diff_deg < 30:
            return 11
        elif heading_diff_deg < 45:
            return 12  # 中幅右转
        elif heading_diff_deg < 60:
            return 13
        elif heading_diff_deg < 90:
            return 14
        elif heading_diff_deg < 120:
            return 15
        else:
            return 16  # 大幅右转
    
    @staticmethod
    def convert_altitude_to_index(altitude_cmd_value: float) -> int:
        """
        将高度变化（米）转换为离散指令索引
        
        Args:
            altitude_cmd_value: 高度变化量（米）
            
        Returns:
            高度指令索引 (0-12):
                0-2: 大幅下降
                3-4: 中幅下降
                5-6: 小幅下降
                7: 保持高度
                8-10: 小幅爬升
                11-12: 大幅爬升
        """
        if altitude_cmd_value < -1000:
            return 0  # 大幅下降
        elif altitude_cmd_value < -500:
            return 2
        elif altitude_cmd_value < -300:
            return 3  # 中幅下降
        elif altitude_cmd_value < -150:
            return 4
        elif altitude_cmd_value < -100:
            return 5  # 小幅下降
        elif altitude_cmd_value < -50:
            return 6
        elif altitude_cmd_value < 50:
            return 7  # 保持高度
        elif altitude_cmd_value < 100:
            return 8
        elif altitude_cmd_value < 150:
            return 9  # 小幅爬升
        elif altitude_cmd_value < 300:
            return 10
        elif altitude_cmd_value < 500:
            return 11  # 中幅爬升
        else:
            return 12  # 大幅爬升
    
    @staticmethod
    def convert_velocity_to_index(velocity_cmd_value: float) -> int:
        """
        将速度变化（m/s）转换为离散指令索引
        
        Args:
            velocity_cmd_value: 速度变化量（m/s）
            
        Returns:
            速度指令索引 (0-6):
                0-1: 大幅减速
                2: 小幅减速
                3: 保持速度
                4: 小幅加速
                5-6: 大幅加速
        """
        if velocity_cmd_value < -100:
            return 0  # 大幅减速
        elif velocity_cmd_value < -50:
            return 1
        elif velocity_cmd_value < -20:
            return 2  # 小幅减速
        elif velocity_cmd_value < 20:
            return 3  # 保持速度
        elif velocity_cmd_value < 50:
            return 4  # 小幅加速
        elif velocity_cmd_value < 100:
            return 5
        else:
            return 6  # 大幅加速
    
    @staticmethod
    def get_enemy_bearing(env, agent_id: str) -> float:
        """
        获取敌机相对当前飞机的方位角（度）
        
        Args:
            env: 环境
            agent_id: 飞机ID
            
        Returns:
            敌机方位角（0-360度），如果没有敌机返回None
        """
        # 获取目标敌机
        from tactical_types import get_target_with_fallback
        target_id = get_target_with_fallback(agent_id, env)
        
        if target_id is None:
            return None
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            return None
        
        # 计算方位角
        pos_self = env.agents[agent_id].get_position()
        pos_target = target_aircraft.get_position()
        
        dx = pos_target[0] - pos_self[0]
        dy = pos_target[1] - pos_self[1]
        
        bearing = np.rad2deg(np.arctan2(dy, dx))  # 弧度转度数
        bearing = (bearing + 360) % 360  # 标准化到[0,360)
        
        return bearing
    
    @staticmethod
    def check_cap_boundary(pos: tuple, cap_center=(0, 0), cap_width=120000, cap_height=120000) -> bool:
        """
        检查飞机是否在CAP区域内
        
        Args:
            pos: 飞机位置 (x, y, z)
            cap_center: CAP中心坐标 (x, y)，默认(0,0)
            cap_width: CAP宽度（米），默认120km
            cap_height: CAP高度（米），默认120km
            
        Returns:
            True表示在边界内，False表示超出边界
        """
        x, y = pos[0], pos[1]
        cx, cy = cap_center
        
        # 使用Racetrack形状判断（矩形+两端半圆）
        half_width = cap_width / 2
        half_height = cap_height / 2
        
        # 距离CAP中心的偏移
        dx = abs(x - cx)
        dy = abs(y - cy)
        
        # 在矩形部分内
        if dx <= half_width and dy <= half_height:
            return True
        
        # 在半圆部分内（左右两端）
        if dx > half_width:
            # 计算到半圆中心的距离
            circle_center_x = cx + (half_width if x > cx else -half_width)
            dist_to_circle = np.sqrt((x - circle_center_x)**2 + (y - cy)**2)
            return dist_to_circle <= half_height
        
        return False
    
    @staticmethod
    def turn_to_heading(env, agent_id: str, target_heading: float, speed_cmd=3, altitude_cmd=7) -> tuple:
        """
        转向到目标航向的通用函数
        
        Args:
            env: 环境
            agent_id: 飞机ID
            target_heading: 目标航向（度，0-360）
            speed_cmd: 速度指令索引，默认3（保持速度）
            altitude_cmd: 高度指令索引，默认7（保持高度）
            
        Returns:
            (altitude_cmd, heading_cmd, velocity_cmd)
        """
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        heading_diff = TacticalUtils.normalize_angle_diff(target_heading - current_heading)
        
        # 转换为航向指令
        heading_cmd = TacticalUtils.convert_heading_to_index(np.deg2rad(heading_diff))
        
        return altitude_cmd, heading_cmd, speed_cmd
    
    @staticmethod
    def get_relative_position(aircraft1, aircraft2) -> dict:
        """
        获取aircraft2相对aircraft1的位置信息
        
        Args:
            aircraft1: 参考飞机
            aircraft2: 目标飞机
            
        Returns:
            字典包含: distance(距离), bearing(方位角), altitude_diff(高度差), 
                     relative_x, relative_y, relative_z
        """
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        bearing = (np.rad2deg(np.arctan2(dy, dx)) + 360) % 360
        
        return {
            'distance': distance,
            'horizontal_distance': horizontal_dist,
            'bearing': bearing,
            'altitude_diff': dz,
            'relative_x': dx,
            'relative_y': dy,
            'relative_z': dz
        }
    
    @staticmethod
    def calculate_aspect_angle(aircraft1, aircraft2) -> float:
        """
        计算aspect angle（目标机头指向与我机的夹角）
        
        Args:
            aircraft1: 我机
            aircraft2: 目标机
            
        Returns:
            aspect angle（度，0-180）
                0度：目标正对我
                90度：目标侧对我
                180度：目标背对我
        """
        # 获取目标航向
        target_heading = aircraft2.get_property_value(c.attitude_psi_deg)
        
        # 获取相对方位
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        dx = pos1[0] - pos2[0]  # 注意：从目标看我的方向
        dy = pos1[1] - pos2[1]
        bearing_to_me = (np.rad2deg(np.arctan2(dy, dx)) + 360) % 360
        
        # 计算夹角
        aspect = abs(TacticalUtils.normalize_angle_diff(bearing_to_me - target_heading))
        
        return aspect
    
    @staticmethod
    def calculate_antenna_train_angle(aircraft1, aircraft2) -> float:
        """
        计算ATA（天线指向角，Antenna Train Angle）
        即我机机头指向与目标的夹角
        
        Args:
            aircraft1: 我机
            aircraft2: 目标机
            
        Returns:
            ATA（度，0-180）
                0度：目标在我正前方
                90度：目标在我侧面
                180度：目标在我正后方
        """
        # 获取我机航向
        my_heading = aircraft1.get_property_value(c.attitude_psi_deg)
        
        # 获取目标相对方位
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        bearing_to_target = (np.rad2deg(np.arctan2(dy, dx)) + 360) % 360
        
        # 计算夹角
        ata = abs(TacticalUtils.normalize_angle_diff(bearing_to_target - my_heading))
        
        return ata
    
    @staticmethod
    def format_distance(distance_m: float) -> str:
        """
        格式化距离显示
        
        Args:
            distance_m: 距离（米）
            
        Returns:
            格式化字符串，例如 "12.5km" 或 "850m"
        """
        if distance_m >= 1000:
            return f"{distance_m/1000:.1f}km"
        else:
            return f"{distance_m:.0f}m"
    
    @staticmethod
    def calculate_closure_rate(aircraft1, aircraft2) -> float:
        """
        计算接近速率（m/s）
        正值表示接近，负值表示远离
        
        Args:
            aircraft1: 飞机1
            aircraft2: 飞机2
            
        Returns:
            接近速率（m/s）
        """
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        
        vel1 = aircraft1.get_property_value(c.velocities_u_fps) * 0.3048  # fps转m/s
        vel2 = aircraft2.get_property_value(c.velocities_u_fps) * 0.3048
        
        heading1 = aircraft1.get_property_value(c.attitude_psi_rad)
        heading2 = aircraft2.get_property_value(c.attitude_psi_rad)
        
        # 速度矢量
        vx1 = vel1 * np.cos(heading1)
        vy1 = vel1 * np.sin(heading1)
        vx2 = vel2 * np.cos(heading2)
        vy2 = vel2 * np.sin(heading2)
        
        # 位置矢量
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 1.0:
            return 0.0
        
        # 相对速度在连线方向上的投影
        dvx = vx2 - vx1
        dvy = vy2 - vy1
        
        closure_rate = -(dvx * dx + dvy * dy) / distance
        
        return closure_rate
