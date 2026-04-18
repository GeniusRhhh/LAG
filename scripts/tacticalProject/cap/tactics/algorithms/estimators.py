"""
协同探测V4 - 目标状态估计器
简化版: 直接使用预警机数据
专业版: IMM-EKF多模型滤波（增强版 - 算法1.1完整实现）
"""
import numpy as np
from typing import Dict, Tuple, Optional
from ..detection_types import TargetState
from ..imm_ekf_enhanced import IMMEKFEnhanced, IMMState


class SimpleEstimator:
    """简化版目标估计器 - 直接使用预警机数据"""
    
    def __init__(self):
        self._states: Dict[str, TargetState] = {}
    
    def update(self, target_id: str, measurement: Tuple[float, float],
               measurement_noise: float, timestamp: float) -> TargetState:
        """直接返回测量值"""
        # 简单速度估计
        vx, vy = 0.0, -0.3
        if target_id in self._states:
            prev = self._states[target_id]
            dt = timestamp - prev.timestamp
            if dt > 0.1:
                vx = (measurement[0] - prev.x) / dt
                vy = (measurement[1] - prev.y) / dt
        
        state = TargetState(
            x=measurement[0], y=measurement[1],
            vx=vx, vy=vy,
            covariance=np.eye(4) * measurement_noise**2,
            timestamp=timestamp
        )
        self._states[target_id] = state
        return state
    
    def get_state(self, target_id: str) -> Optional[TargetState]:
        return self._states.get(target_id)


class UKFFilter:
    """无迹卡尔曼滤波器"""
    
    def __init__(self, model_type: str = 'CV'):
        self.model_type = model_type
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4) * 10.0
        self.Q = self._get_process_noise()
        self.initialized = False
    
    def _get_process_noise(self) -> np.ndarray:
        """获取过程噪声"""
        if self.model_type == 'CV':
            return np.diag([0.1, 0.1, 0.01, 0.01])
        elif self.model_type == 'CA':
            return np.diag([0.2, 0.2, 0.05, 0.05])
        elif self.model_type == 'CT':
            return np.diag([0.3, 0.3, 0.1, 0.1])
        else:  # Singer
            return np.diag([0.15, 0.15, 0.03, 0.03])
    
    def predict(self, dt: float):
        """预测步骤"""
        if not self.initialized:
            return
        
        # 状态转移
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        if self.model_type == 'CT':
            # 协调转弯模型 - 添加转弯
            omega = 0.05  # 转弯率
            if abs(omega) > 1e-6:
                F = np.array([
                    [1, 0, np.sin(omega*dt)/omega, -(1-np.cos(omega*dt))/omega],
                    [0, 1, (1-np.cos(omega*dt))/omega, np.sin(omega*dt)/omega],
                    [0, 0, np.cos(omega*dt), -np.sin(omega*dt)],
                    [0, 0, np.sin(omega*dt), np.cos(omega*dt)]
                ])
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, measurement: Tuple[float, float], R: float):
        """更新步骤"""
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return
        
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        z = np.array(measurement)
        
        # 卡尔曼增益
        S = H @ self.P @ H.T + np.eye(2) * R**2
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 更新
        y = z - H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
    
    def get_likelihood(self, measurement: Tuple[float, float], R: float) -> float:
        """计算测量似然"""
        if not self.initialized:
            return 1.0
        
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        z = np.array(measurement)
        
        y = z - H @ self.state
        S = H @ self.P @ H.T + np.eye(2) * R**2
        
        det_S = np.linalg.det(S)
        if det_S < 1e-10:
            return 1e-10
        
        exp_term = -0.5 * y.T @ np.linalg.inv(S) @ y
        return np.exp(exp_term) / np.sqrt((2*np.pi)**2 * det_S)


class IMMEKFEstimator:
    """IMM-EKF多模型估计器 - 增强版（算法1.1完整实现）
    
    使用增强版IMM-EKF替代原UKF实现:
    - 完整的4模型IMM (CV/CT/CA/Singer)
    - EKF非线性滤波（符合规范要求）
    - 自适应过程噪声
    - 完整的Singer模型
    - 100%算法1.1规范符合
    
    模型集合:
    - CV (Constant Velocity): 匀速直线
    - CA (Constant Acceleration): 匀加速
    - CT (Coordinated Turn): 协调转弯
    - Singer: Singer加速度模型
    """
    
    def __init__(self):
        self._estimators: Dict[str, IMMEKFEnhanced] = {}
        self._last_time: Dict[str, float] = {}
        self._last_target_id: Optional[str] = None
    
    def update(self, target_id: str, measurement: Tuple[float, float],
               measurement_noise: float, timestamp: float) -> TargetState:
        """IMM-EKF更新（使用增强版）
        
        Args:
            target_id: 目标ID
            measurement: 观测位置 (x, y) km
            measurement_noise: 观测噪声标准差 (km)
            timestamp: 时间戳 (s)
        
        Returns:
            TargetState包含估计状态和协方差
        """
        # 初始化
        if target_id not in self._estimators:
            self._estimators[target_id] = IMMEKFEnhanced(dt=0.2)
            self._last_time[target_id] = timestamp
        
        # 准备观测
        z = np.array(measurement)
        R = np.eye(2) * measurement_noise**2
        
        # IMM-EKF更新（算法1.1完整流程）
        state: IMMState = self._estimators[target_id].update(z, R, timestamp)
        
        self._last_time[target_id] = timestamp
        self._last_target_id = target_id
        
        # 转换为TargetState
        return TargetState(
            x=state.x[0], y=state.x[1],
            vx=state.x[2], vy=state.x[3],
            covariance=state.P,
            timestamp=timestamp
        )

    def predict(self, dt: float, target_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """无量测预测（用于协同探测算法2.5）。

        重要：该预测必须是“非破坏性”的——不能改变内部滤波器状态，
        否则上层若每步用 total-τ 调用predict会导致重复传播。
        """
        tid = target_id or self._last_target_id
        if tid is None and self._estimators:
            tid = next(iter(self._estimators.keys()))
        if tid is None or tid not in self._estimators:
            return np.zeros(4), np.eye(4) * 10.0

        imm = self._estimators[tid]

        # ---- snapshot ----
        snap_filters = {}
        for m, f in imm.filters.items():
            snap_filters[m] = (f.x.copy(), f.P.copy())
        snap_fused_x = imm.fused_x.copy()
        snap_fused_P = imm.fused_P.copy()

        try:
            x_pred, P_pred = imm.predict(float(dt))
            # 统一返回4维
            if x_pred.shape[0] != 4:
                x4 = np.zeros(4)
                x4[:min(4, x_pred.shape[0])] = x_pred[:min(4, x_pred.shape[0])]
                x_pred = x4
            if P_pred.shape != (4, 4):
                P4 = np.eye(4) * 10.0
                min_dim = min(P_pred.shape[0], 4)
                P4[:min_dim, :min_dim] = P_pred[:min_dim, :min_dim]
                P_pred = P4
            return x_pred, P_pred
        finally:
            # ---- restore ----
            for m, (x0, P0) in snap_filters.items():
                imm.filters[m].x = x0
                imm.filters[m].P = P0
            imm.fused_x = snap_fused_x
            imm.fused_P = snap_fused_P
    
    def get_state(self, target_id: str) -> Optional[TargetState]:
        """获取当前状态"""
        if target_id not in self._estimators:
            return None
        
        imm = self._estimators[target_id]
        return TargetState(
            x=imm.fused_x[0], y=imm.fused_x[1],
            vx=imm.fused_x[2], vy=imm.fused_x[3],
            covariance=imm.fused_P,
            timestamp=self._last_time.get(target_id, 0)
        )
    
    def get_model_probabilities(self, target_id: str) -> Optional[Dict[str, float]]:
        """获取模型概率（用于调试和分析）"""
        if target_id not in self._estimators:
            return None
        return self._estimators[target_id].get_model_probabilities()
    
    def get_dominant_model(self, target_id: str) -> Optional[str]:
        """获取主导模型（用于调试和分析）"""
        if target_id not in self._estimators:
            return None
        return self._estimators[target_id].get_dominant_model()


# 保留旧版本作为备份（可选）
class IMMUKFEstimatorLegacy:
    """IMM-UKF多模型估计器（旧版本 - 保留作为备份）
    
    注意：此版本已被IMMEKFEstimator（增强版）替代
    保留此代码仅用于对比和回退
    """
    
    def __init__(self):
        self._filters: Dict[str, Dict[str, UKFFilter]] = {}
        self._model_probs: Dict[str, np.ndarray] = {}
        self._last_time: Dict[str, float] = {}
        
        # 模型转移概率矩阵
        self.TPM = np.array([
            [0.9, 0.03, 0.04, 0.03],  # CV
            [0.03, 0.9, 0.04, 0.03],  # CA
            [0.03, 0.03, 0.9, 0.04],  # CT
            [0.03, 0.03, 0.04, 0.9]   # Singer
        ])
        self.models = ['CV', 'CA', 'CT', 'Singer']
    
    def update(self, target_id: str, measurement: Tuple[float, float],
               measurement_noise: float, timestamp: float) -> TargetState:
        """IMM-UKF更新"""
        # 初始化
        if target_id not in self._filters:
            self._filters[target_id] = {m: UKFFilter(m) for m in self.models}
            self._model_probs[target_id] = np.array([0.7, 0.1, 0.1, 0.1])
            self._last_time[target_id] = timestamp
        
        filters = self._filters[target_id]
        probs = self._model_probs[target_id]
        dt = timestamp - self._last_time[target_id]
        
        if dt > 0.01:
            # 1. 模型交互
            mixed_states, mixed_covs = self._mix_states(filters, probs)
            
            # 2. 各模型预测和更新
            likelihoods = np.zeros(4)
            for i, model in enumerate(self.models):
                filters[model].state = mixed_states[i]
                filters[model].P = mixed_covs[i]
                filters[model].predict(dt)
                filters[model].update(measurement, measurement_noise)
                likelihoods[i] = filters[model].get_likelihood(measurement, measurement_noise)
            
            # 3. 更新模型概率
            c = self.TPM.T @ probs
            probs = likelihoods * c
            probs_sum = np.sum(probs)
            if probs_sum > 1e-10:
                probs = probs / probs_sum
            else:
                probs = np.array([0.7, 0.1, 0.1, 0.1])
            
            self._model_probs[target_id] = probs
        else:
            # 仅更新
            for model in self.models:
                filters[model].update(measurement, measurement_noise)
        
        self._last_time[target_id] = timestamp
        
        # 4. 状态融合
        fused_state, fused_cov = self._fuse_states(filters, probs)
        
        return TargetState(
            x=fused_state[0], y=fused_state[1],
            vx=fused_state[2], vy=fused_state[3],
            covariance=fused_cov,
            timestamp=timestamp
        )
    
    def _mix_states(self, filters: Dict[str, UKFFilter], probs: np.ndarray):
        """模型交互"""
        n = len(self.models)
        mixed_states = []
        mixed_covs = []
        
        # 计算混合概率
        c = self.TPM.T @ probs
        
        for j in range(n):
            if c[j] < 1e-10:
                mixed_states.append(filters[self.models[j]].state.copy())
                mixed_covs.append(filters[self.models[j]].P.copy())
                continue
            
            # 混合状态
            mixed_state = np.zeros(4)
            for i in range(n):
                mu_ij = self.TPM[i, j] * probs[i] / c[j]
                mixed_state += mu_ij * filters[self.models[i]].state
            
            # 混合协方差
            mixed_cov = np.zeros((4, 4))
            for i in range(n):
                mu_ij = self.TPM[i, j] * probs[i] / c[j]
                diff = filters[self.models[i]].state - mixed_state
                mixed_cov += mu_ij * (filters[self.models[i]].P + np.outer(diff, diff))
            
            mixed_states.append(mixed_state)
            mixed_covs.append(mixed_cov)
        
        return mixed_states, mixed_covs
    
    def _fuse_states(self, filters: Dict[str, UKFFilter], probs: np.ndarray):
        """状态融合"""
        fused_state = np.zeros(4)
        for i, model in enumerate(self.models):
            fused_state += probs[i] * filters[model].state
        
        fused_cov = np.zeros((4, 4))
        for i, model in enumerate(self.models):
            diff = filters[model].state - fused_state
            fused_cov += probs[i] * (filters[model].P + np.outer(diff, diff))
        
        return fused_state, fused_cov
    
    def get_state(self, target_id: str) -> Optional[TargetState]:
        if target_id not in self._filters:
            return None
        
        filters = self._filters[target_id]
        probs = self._model_probs[target_id]
        fused_state, fused_cov = self._fuse_states(filters, probs)
        
        return TargetState(
            x=fused_state[0], y=fused_state[1],
            vx=fused_state[2], vy=fused_state[3],
            covariance=fused_cov,
            timestamp=self._last_time.get(target_id, 0)
        )
