"""
IMM-EKF 交互多模型目标跟踪算法
Interacting Multiple Model (IMM) with Extended Kalman Filter (EKF)

包含三个模型：
1. CV (Constant Velocity): 匀速直线运动 (4维状态: x, y, vx, vy)
2. CT (Coordinated Turn): 协调转弯运动 (5维状态: x, y, vx, vy, omega)
3. CA (Constant Acceleration): 匀加速运动 (6维状态: x, y, vx, vy, ax, ay)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from .state_estimator import TargetState

# 状态维度映射
# CV: [x, y, vx, vy]
# CT: [x, y, vx, vy, w]
# CA: [x, y, vx, vy, ax, ay]

class KalmanFilter:
    """基础卡尔曼滤波器"""
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros(dim_x)     # 状态
        self.P = np.eye(dim_x)       # 协方差
        self.Q = np.eye(dim_x)       # 过程噪声
        self.R = np.eye(dim_z)       # 测量噪声
        self.H = np.zeros((dim_z, dim_x)) # 观测矩阵
        
    def predict(self, dt: float, F: np.ndarray = None, Q: np.ndarray = None):
        if F is None: F = np.eye(self.dim_x)
        if Q is None: Q = self.Q
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        
    def update(self, z: np.ndarray, H: np.ndarray = None, R: np.ndarray = None):
        if H is None: H = self.H
        if R is None: R = self.R
        
        y = z - H @ self.x  #/ residual
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P
        
        # 似然函数计算 (用于IMM)
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        norm_factor = 1.0 / np.sqrt((2 * np.pi) ** self.dim_z * det_S)
        likelihood = norm_factor * np.exp(-0.5 * (y.T @ inv_S @ y))
        return likelihood

class CVModel(KalmanFilter):
    """匀速直线运动模型 (CV)"""
    def __init__(self, process_noise: float = 0.5, meas_noise: float = 2.5):
        super().__init__(dim_x=4, dim_z=2)
        # 观测矩阵: z = [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        # 噪声初始化
        self.R = np.eye(2) * meas_noise**2
        self.q_std = process_noise

    def predict_step(self, dt: float):
        # 状态转移 F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # 离散过程噪声 Q (Piecewise White Noise Model)
        # q_std 表示加速度噪声强度
        q = self.q_std**2
        G = np.array([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])
        Q = G @ G.T * q
        
        self.predict(dt, F, Q)

class CTModel(KalmanFilter):
    """协调转弯模型 (CT) - EKF"""
    def __init__(self, process_noise: float = 1.0, meas_noise: float = 2.5):
        super().__init__(dim_x=5, dim_z=2) # [x, y, vx, vy, w]
        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        self.R = np.eye(2) * meas_noise**2
        self.q_std = process_noise

    def predict_step(self, dt: float):
        # EKF 状态预测 (非线性)
        x, y, vx, vy, w = self.x
        
        if abs(w) < 1e-4: # 近似直线
            # 避免除以零
            dx = vx * dt
            dy = vy * dt
            dvx = 0
            dvy = 0
            F_jacobian = np.eye(5)
            F_jacobian[0, 2] = dt
            F_jacobian[1, 3] = dt
        else:
            sin_wt = np.sin(w * dt)
            cos_wt = np.cos(w * dt)
            
            # 非线性状态更新
            dx = (vx * sin_wt - vy * (1 - cos_wt)) / w
            dy = (vx * (1 - cos_wt) + vy * sin_wt) / w
            dvx = vx * cos_wt - vy * sin_wt - vx
            dvy = vx * sin_wt + vy * cos_wt - vy
            
            # 更新状态
            self.x[0] += dx
            self.x[1] += dy
            self.x[2] += dvx + vx # dvx is delta
            self.x[3] += dvy + vy
            # w 保持不变
            
            # 雅可比矩阵 F (线性化) for P update
            # 简化版雅可比，省略w对pos的复杂导数，主要关注速度旋转
            F_jacobian = np.eye(5)
            F_jacobian[0, 2] = sin_wt / w
            F_jacobian[0, 3] = -(1 - cos_wt) / w
            F_jacobian[1, 2] = (1 - cos_wt) / w
            F_jacobian[1, 3] = sin_wt / w
            F_jacobian[2, 2] = cos_wt
            F_jacobian[2, 3] = -sin_wt
            F_jacobian[3, 2] = sin_wt
            F_jacobian[3, 3] = cos_wt
            
        # 过程噪声 (主要在速度和角速度上)
        Q = np.eye(5) * (self.q_std**2 * dt)
        Q[4, 4] = (0.1 * dt)**2 # 角速度噪声较小
        
        # 只更新P (x已经被非线性更新了)
        self.P = F_jacobian @ self.P @ F_jacobian.T + Q

class CAModel(KalmanFilter):
    """匀加速运动模型 (CA)"""
    def __init__(self, process_noise: float = 2.0, meas_noise: float = 2.5):
        super().__init__(dim_x=6, dim_z=2) # [x, y, vx, vy, ax, ay]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        self.R = np.eye(2) * meas_noise**2
        self.q_std = process_noise

    def predict_step(self, dt: float):
        # F matrix for CA
        F = np.eye(6)
        F[0, 2] = dt; F[0, 4] = 0.5*dt**2
        F[1, 3] = dt; F[1, 5] = 0.5*dt**2
        F[2, 4] = dt
        F[3, 5] = dt
        
        # Q matrix (jerk noise)
        q = self.q_std**2
        G = np.array([
            [dt**3/6, 0],
            [0, dt**3/6],
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])
        Q = G @ G.T * q
        
        self.predict(dt, F, Q)


class SingerModel(KalmanFilter):
    """Singer机动模型 (随机机动)
    
    假设目标加速度是一个时间相关的随机过程
    a(t) = -alpha * a(t) + w(t)
    
    状态: [x, y, vx, vy, ax, ay]
    """
    def __init__(self, alpha: float = 0.1, process_noise: float = 1.0, meas_noise: float = 2.5):
        super().__init__(dim_x=6, dim_z=2) # [x, y, vx, vy, ax, ay]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        self.R = np.eye(2) * meas_noise**2
        self.alpha = alpha     # 机动频率倒数 (1/tau)
        self.sigma = process_noise  # 机动方差
        
    def predict_step(self, dt: float):
        alpha = self.alpha
        alpha_dt = alpha * dt
        e_adt = np.exp(-alpha_dt)
        
        # 状态转移矩阵 F (Singer模型推导)
        # x(k+1) = x(k) + v(k)*dt + (alpha*dt - 1 + e^(-alpha*dt))/alpha^2 * a(k)
        # v(k+1) = v(k) + (1 - e^(-alpha*dt))/alpha * a(k)
        # a(k+1) = e^(-alpha*dt) * a(k)
        
        c1 = (alpha_dt - 1 + e_adt) / alpha**2
        c2 = (1 - e_adt) / alpha
        c3 = e_adt
        
        F = np.eye(6)
        F[0, 2] = dt; F[0, 4] = c1
        F[1, 3] = dt; F[1, 5] = c1
        F[2, 4] = c2
        F[3, 5] = c2
        F[4, 4] = c3
        F[5, 5] = c3
        
        # 过程噪声协方差 Q
        # 复杂推导结果，这里使用简化近似以提高计算效率
        q = 2 * alpha * self.sigma**2
        
        # 简化版Q矩阵 (主要噪声在加速度项)
        # 假设位置和速度噪声由加速度噪声积分得到
        Q = np.zeros((6, 6))
        Q[4, 4] = q * (1 - e_adt**2) / (2 * alpha)
        Q[5, 5] = Q[4, 4]
        
        # 传播至速度和位置 (近似)
        Q[2, 2] = Q[4, 4] * dt**2
        Q[3, 3] = Q[5, 5] * dt**2
        Q[0, 0] = Q[2, 2] * dt**2 / 3
        Q[1, 1] = Q[3, 3] * dt**2 / 3
        
        self.predict(dt, F, Q)


class IMMFilter:
    """交互多模型滤波器"""
    def __init__(self, start_pos: np.ndarray):
        # 初始化模型
        self.cv = CVModel(process_noise=0.5)  # 巡航
        self.ct = CTModel(process_noise=1.0)  # 转弯
        self.ca = CAModel(process_noise=3.0)  # 机动
        self.singer = SingerModel(alpha=0.1, process_noise=2.0) # 随机机动
        
        self.models = [self.cv, self.ct, self.ca, self.singer]
        
        # 模型转换概率矩阵 (4x4)
        # CV, CT, CA, Singer
        self.trans_prob = np.array([
            [0.85, 0.05, 0.05, 0.05], # CV -> CV high
            [0.10, 0.80, 0.05, 0.05], # CT -> CT high
            [0.10, 0.10, 0.70, 0.10], # CA -> CA high
            [0.10, 0.10, 0.10, 0.70]  # Singer -> Singer high
        ])
        
        # 模型概率
        self.model_probs = np.array([0.7, 0.1, 0.1, 0.1])
        
        # 初始化状态
        for model in self.models:
            model.x[:2] = start_pos
            model.P *= 10.0
        
    def _mix_states(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """交互步骤：根据转换概率混合状态"""
        n_models = len(self.models)
        mixed_estimates = []
        
        # 计算混合概率 mu_ij (从模型i到模型j的条件概率)
        c_j = np.zeros(n_models)
        for j in range(n_models):
            for i in range(n_models):
                c_j[j] += self.trans_prob[i, j] * self.model_probs[i]
        
        mu = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                mu[i, j] = (self.trans_prob[i, j] * self.model_probs[i]) / (c_j[j] + 1e-10)
        
        # 对每个目标模型j，混合所有模型i的状态
        for j in range(n_models):
            # 混合状态 x0j
            x0j = np.zeros(self.models[j].dim_x)
            for i in range(n_models):
                # 状态映射: i -> j
                x_i = self._map_state(self.models[i].x, self.models[i].dim_x, self.models[j].dim_x)
                x0j += x_i * mu[i, j]
                
            # 混合协方差 P0j
            P0j = np.zeros((self.models[j].dim_x, self.models[j].dim_x))
            for i in range(n_models):
                x_i = self._map_state(self.models[i].x, self.models[i].dim_x, self.models[j].dim_x)
                P_i = self._map_cov(self.models[i].P, self.models[i].dim_x, self.models[j].dim_x)
                
                diff = (x_i - x0j).reshape(-1, 1)
                P0j += mu[i, j] * (P_i + diff @ diff.T)
                
            mixed_estimates.append((x0j, P0j))
            
        return mixed_estimates

    def _map_state(self, x_src: np.ndarray, dim_src: int, dim_tgt: int) -> np.ndarray:
        """映射状态向量 (简单的截断或补零)"""
        x_tgt = np.zeros(dim_tgt)
        # 公共部分: x, y, vx, vy
        common_dim = min(dim_src, dim_tgt, 4)
        x_tgt[:common_dim] = x_src[:common_dim]
        # 特有部分处理 (CT的w, CA的ax,ay)
        # 简单处理：额外维度设为0
        return x_tgt

    def _map_cov(self, P_src: np.ndarray, dim_src: int, dim_tgt: int) -> np.ndarray:
        """映射协方差矩阵"""
        P_tgt = np.eye(dim_tgt) * 100.0 # 默认大方差
        common = min(dim_src, dim_tgt, 4)
        P_tgt[:common, :common] = P_src[:common, :common]
        return P_tgt

    def update(self, z: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """IMM更新循环
        Returns: (fused_state, fused_cov, model_probs)
        """
        # 1. 交互/混合
        mixed = self._mix_states()
        
        # 重新初始化各模型
        for j in range(len(self.models)):
            x0, P0 = mixed[j]
            self.models[j].x = x0
            self.models[j].P = P0
            
        # 2. 模型滤波 (预测 + 更新)
        likelihoods = np.zeros(len(self.models))
        for j, model in enumerate(self.models):
            # 预测
            if hasattr(model, 'predict_step'):
                model.predict_step(dt)
            else:
                model.predict(dt)
                
            # 更新并计算似然
            likelihoods[j] = model.update(z)
            
        # 3. 概率更新
        c_sum = 0.0
        c_j = np.zeros(len(self.models))
        # 先计算预测概率
        for j in range(len(self.models)):
            param = 0.0
            for i in range(len(self.models)):
                param += self.trans_prob[i, j] * self.model_probs[i]
            c_j[j] = param
            
        new_probs = likelihoods * c_j
        total_prob = np.sum(new_probs)
        
        if total_prob > 0:
            self.model_probs = new_probs / total_prob
        else:
            # 数值稳定保护
            self.model_probs = np.ones(len(self.models)) / len(self.models)
            
        # 4. 状态融合
        fused_dim = 6 # 统一输出到最高维度 [x, y, vx, vy, ax, ay]
        fused_x = np.zeros(fused_dim)
        fused_P = np.zeros((fused_dim, fused_dim))
        
        for j, model in enumerate(self.models):
            x_mapped = self._map_state(model.x, model.dim_x, fused_dim)
            if model.dim_x == 5: # CT model special mapping for ax, ay
                # CT 加速度 approx: a = v * w (向心力)
                v_mag = np.linalg.norm(model.x[2:4])
                w = model.x[4]
                # 加速度方向垂直于速度
                # ... 暂时简化为仅映射状态
                pass
                
            fused_x += self.model_probs[j] * x_mapped
            
        for j, model in enumerate(self.models):
            x_mapped = self._map_state(model.x, model.dim_x, fused_dim)
            P_mapped = self._map_cov(model.P, model.dim_x, fused_dim)
            diff = (x_mapped - fused_x).reshape(-1, 1)
            fused_P += self.model_probs[j] * (P_mapped + diff @ diff.T)
            
        return fused_x, fused_P, self.model_probs.tolist()

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """融合预测"""
        # 对每个模型单独预测
        preds = []
        for model in self.models:
            # 保存当前状态
            x_old = model.x.copy()
            P_old = model.P.copy()
            
            # 预测
            if hasattr(model, 'predict_step'):
                model.predict_step(dt)
            else:
                model.predict(dt)
            
            preds.append((model.x.copy(), model.P.copy()))
            
            # 恢复状态 (因为这只是预测查询，不应改变滤波器状态)
            model.x = x_old
            model.P = P_old
            
        # 融合预测结果
        fused_dim = 6
        x_pred = np.zeros(fused_dim)
        for j, (x, _) in enumerate(preds):
            x_mapped = self._map_state(x, self.models[j].dim_x, fused_dim)
            x_pred += self.model_probs[j] * x_mapped
            
        return x_pred, np.eye(6) # P not strictly needed for just position predict

@dataclass
class IMMEstimatorWrapper:
    """IMM状态估计器包装类 (适配SimpleStateEstimator接口)"""
    def __init__(self):
        self._estimators: Dict[str, IMMFilter] = {}
        self._states: Dict[str, TargetState] = {}
        
    def update(self, target_id: str, position: np.ndarray, 
               velocity: np.ndarray, heading: float, timestamp: float) -> TargetState:
               
        if target_id not in self._estimators:
            self._estimators[target_id] = IMMFilter(position[:2])
            last_time = timestamp - 0.2
        else:
            last_state = self._states[target_id]
            last_time = last_state.timestamp
            
        dt = timestamp - last_time
        if dt <= 0: dt = 0.2
        
        estimator = self._estimators[target_id]
        # 更新滤波器
        fused_x, fused_P, probs = estimator.update(position[:2], dt)
        
        # 构造输出状态
        # IMM输出 fused_x = [x, y, vx, vy, ax, ay]
        state = TargetState(
            target_id=target_id,
            position=np.array([fused_x[0], fused_x[1], position[2]]), # z保持测量值
            velocity=np.array([fused_x[2]*1000, fused_x[3]*1000, velocity[2]]), # 转回m/s
            heading=heading, # 航向暂时使用测量值，也可从速度矢量计算
            timestamp=timestamp,
            covariance=fused_P
        )
        self._states[target_id] = state
        
        # 调试打印 (偶尔)
        # if np.random.random() < 0.01:
        #    print(f"[{target_id}] IMM Probs: CV={probs[0]:.2f} CT={probs[1]:.2f} CA={probs[2]:.2f}")
            
        return state
        
    def predict(self, target_id: str, dt: float) -> Optional[TargetState]:
        if target_id not in self._estimators:
            return None
            
        estimator = self._estimators[target_id]
        x_pred, _ = estimator.predict(dt)
        state = self._states[target_id]
        
        return TargetState(
            target_id=target_id,
            position=np.array([x_pred[0], x_pred[1], state.position[2]]),
            velocity=np.array([x_pred[2]*1000, x_pred[3]*1000, state.velocity[2]]),
            heading=state.heading,
            timestamp=state.timestamp + dt,
            covariance=None
        )
        
    def get_state(self, target_id: str) -> Optional[TargetState]:
        return self._states.get(target_id)
        
    def clear(self):
        self._estimators.clear()
        self._states.clear()
