#!/usr/bin/env python3
"""
R-27ER导弹调试脚本
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.drag_shoot_2v2.r27er_missile import R27ERMissileSimulator

class MockAircraft:
    """模拟飞机类"""
    def __init__(self, uid, position, velocity, dt=1/60):
        self.uid = uid
        self.dt = dt
        self.color = "Blue" if uid.startswith('B') else "Red"
        self.is_alive = True
        self.launch_missiles = []
        self.under_missiles = []
        
        # 位置和速度
        self._position = np.array(position, dtype=float)
        self._velocity = np.array(velocity, dtype=float)
        self._geodetic = np.array([0.0, 0.0, position[2]], dtype=float)
        self._posture = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # 坐标系参考点
        self.lon0, self.lat0, self.alt0 = 0.0, 0.0, 0.0
        
    def get_position(self):
        return self._position.copy()
    
    def get_velocity(self):
        return self._velocity.copy()
    
    def get_geodetic(self):
        return self._geodetic.copy()
    
    def get_rpy(self):
        return self._posture.copy()
    
    def step(self):
        """更新飞机状态"""
        self._position += self._velocity * self.dt
        self._geodetic[2] = self._position[2]  # 更新高度
    
    def shotdown(self):
        self.is_alive = False
        print(f"💥 {self.uid} 被击落!")

def debug_r27er():
    """调试R-27ER导弹"""
    print("🔍 R-27ER导弹调试")
    print("=" * 40)
    
    # 简单的正面对冲场景
    launcher_pos = np.array([10000, 0, 6000])  # 10km距离
    launcher_vel = np.array([-200, 0, 0])  # 向目标方向飞行
    launcher = MockAircraft("B0100", launcher_pos, launcher_vel)
    
    target_pos = np.array([0, 0, 6000])  # 原点
    target_vel = np.array([0, 0, 0])  # 静止目标
    target = MockAircraft("A0100", target_pos, target_vel)
    
    # 创建R-27ER导弹
    missile = R27ERMissileSimulator.create(launcher, target, "B1001")
    
    print(f"初始状态:")
    print(f"  发射平台位置: {launcher_pos}")
    print(f"  目标位置: {target_pos}")
    print(f"  导弹位置: {missile.get_position()}")
    print(f"  导弹速度: {missile.get_velocity()}")
    print(f"  导弹姿态: {missile.get_rpy() * 180 / np.pi}")
    print(f"  初始距离: {missile.target_distance:.1f}m")
    
    # 运行几步看看
    for step in range(10):
        old_pos = missile.get_position().copy()
        old_distance = missile.target_distance
        
        missile.run()
        target.step()
        
        new_pos = missile.get_position()
        new_distance = missile.target_distance
        velocity = missile.get_velocity()
        
        print(f"\n步骤 {step+1}:")
        print(f"  时间: {missile._t:.2f}s")
        print(f"  导弹位置: [{new_pos[0]:.1f}, {new_pos[1]:.1f}, {new_pos[2]:.1f}]")
        print(f"  目标位置: [{target.get_position()[0]:.1f}, {target.get_position()[1]:.1f}, {target.get_position()[2]:.1f}]")
        print(f"  导弹速度: [{velocity[0]:.1f}, {velocity[1]:.1f}, {velocity[2]:.1f}] (|v|={np.linalg.norm(velocity):.1f})")
        print(f"  导弹姿态: {missile.get_rpy() * 180 / np.pi}")
        print(f"  距离变化: {old_distance:.1f} -> {new_distance:.1f} (Δ={new_distance-old_distance:.1f})")
        print(f"  位移: {np.linalg.norm(new_pos - old_pos):.1f}m")
        print(f"  阶段: {missile._phase}")
        print(f"  状态: {'ALIVE' if missile.is_alive else 'DONE'}")
        
        if not missile.is_alive:
            print(f"  导弹状态: {missile._R27ERMissileSimulator__status}")
            break

if __name__ == "__main__":
    debug_r27er()
