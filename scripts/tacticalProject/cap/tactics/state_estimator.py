"""
目标状态估计算法模块
- 简单版：直接使用测量位置
- 专业版：卡尔曼滤波状态估计
"""
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TargetState:
    """目标状态"""
    target_id: str
    position: np.ndarray  # [x, y, z] km
    velocity: np.ndarray  # [vx, vy, vz] m/s
    heading: float  # 度
    timestamp: float  # 秒
    covariance: Optional[np.ndarray] = None  # 协方差矩阵 (可选)


class SimpleStateEstimator:
    """简单状态估计器 - 直接使用测量值"""
    
    def __init__(self):
        self._states: Dict[str, TargetState] = {}
    
    def update(self, target_id: str, position: np.ndarray, 
               velocity: np.ndarray, heading: float, timestamp: float) -> TargetState:
        """更新目标状态 - 直接使用测量值"""
        state = TargetState(
            target_id=target_id,
            position=position.copy(),
            velocity=velocity.copy(),
            heading=heading,
            timestamp=timestamp
        )
        self._states[target_id] = state
        return state
    
    def predict(self, target_id: str, dt: float) -> Optional[TargetState]:
        """预测目标位置 - 简单线性外推"""
        if target_id not in self._states:
            return None
        
        state = self._states[target_id]
        new_pos = state.position + state.velocity * dt / 1000.0  # m/s to km/s
        
        return TargetState(
            target_id=target_id,
            position=new_pos,
            velocity=state.velocity.copy(),
            heading=state.heading,
            timestamp=state.timestamp + dt
        )
    
    def get_state(self, target_id: str) -> Optional[TargetState]:
        return self._states.get(target_id)
    
    def clear(self):
        self._states.clear()


class KalmanStateEstimator:
    """卡尔曼滤波状态估计器 - 专业版
    
    状态向量: [x, y, vx, vy] (位置和速度)
    观测向量: [x, y] (仅位置)
    """
    
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 2.5):
        self._states: Dict[str, TargetState] = {}
        self._P: Dict[str, np.ndarray] = {}  # 协方差矩阵
        self.Q = np.eye(4) * process_noise  # 过程噪声
        self.R = np.eye(2) * measurement_noise ** 2  # 测量噪声
        
        # 观测矩阵 H (只观测位置)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
    
    def _get_F(self, dt: float) -> np.ndarray:
        """状态转移矩阵"""
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def update(self, target_id: str, position: np.ndarray, 
               velocity: np.ndarray, heading: float, timestamp: float) -> TargetState:
        """卡尔曼滤波更新"""
        z = position[:2]  # 观测值 [x, y]
        
        if target_id not in self._states:
            # 初始化
            x = np.array([position[0], position[1], velocity[0]/1000, velocity[1]/1000])
            P = np.eye(4) * 10.0
        else:
            # 预测步骤
            old_state = self._states[target_id]
            dt = timestamp - old_state.timestamp
            if dt <= 0:
                dt = 0.2  # 默认步长
            
            F = self._get_F(dt)
            x_prior = np.array([
                old_state.position[0], old_state.position[1],
                old_state.velocity[0]/1000, old_state.velocity[1]/1000
            ])
            x_prior = F @ x_prior
            P_prior = F @ self._P[target_id] @ F.T + self.Q
            
            # 更新步骤
            y = z - self.H @ x_prior  # 残差
            S = self.H @ P_prior @ self.H.T + self.R
            K = P_prior @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
            
            x = x_prior + K @ y
            P = (np.eye(4) - K @ self.H) @ P_prior
        
        # 存储
        self._P[target_id] = P
        
        state = TargetState(
            target_id=target_id,
            position=np.array([x[0], x[1], position[2]]),
            velocity=np.array([x[2]*1000, x[3]*1000, velocity[2]]),
            heading=heading,
            timestamp=timestamp,
            covariance=P.copy()
        )
        self._states[target_id] = state
        return state
    
    def predict(self, target_id: str, dt: float) -> Optional[TargetState]:
        """卡尔曼预测"""
        if target_id not in self._states:
            return None
        
        state = self._states[target_id]
        F = self._get_F(dt)
        
        x = np.array([
            state.position[0], state.position[1],
            state.velocity[0]/1000, state.velocity[1]/1000
        ])
        x_pred = F @ x
        P_pred = F @ self._P[target_id] @ F.T + self.Q
        
        return TargetState(
            target_id=target_id,
            position=np.array([x_pred[0], x_pred[1], state.position[2]]),
            velocity=np.array([x_pred[2]*1000, x_pred[3]*1000, state.velocity[2]]),
            heading=state.heading,
            timestamp=state.timestamp + dt,
            covariance=P_pred.copy()
        )
    
    def get_state(self, target_id: str) -> Optional[TargetState]:
        return self._states.get(target_id)
    
    def clear(self):
        self._states.clear()
        self._P.clear()


def create_state_estimator(use_kalman: bool = False, **kwargs) -> SimpleStateEstimator:
    """工厂函数：创建状态估计器
    
    Args:
        use_kalman: True=使用高级估计(IMM/KF), False=使用简单估计
    """
    if use_kalman:
        # 尝试使用IMM估计器
        try:
            from .imm_estimator import IMMEstimatorWrapper
            return IMMEstimatorWrapper()
        except ImportError:
            # 回退到普通卡尔曼
            return KalmanStateEstimator(**kwargs)
    else:
        return SimpleStateEstimator()
