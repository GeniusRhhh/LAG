"""
IMM-EKF增强版 - 算法1.1完整实现
严格按照规范文档，包含自适应功能

特性：
1. 完整的4模型IMM (CV/CT/CA/Singer)
2. EKF非线性滤波（替代UKF）
3. 自适应过程噪声
4. 自适应转弯率估计
5. 完整的Singer模型
"""
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class IMMState:
    """IMM状态"""
    x: np.ndarray  # 融合状态
    P: np.ndarray  # 融合协方差
    model_states: Dict[str, np.ndarray]  # 各模型状态
    model_covs: Dict[str, np.ndarray]  # 各模型协方差
    model_probs: np.ndarray  # 模型概率
    timestamp: float


class EKFModel:
    """EKF模型基类"""
    
    def __init__(self, model_type: str, dt: float = 0.2):
        self.model_type = model_type
        self.dt = dt
        self.state_dim = self._get_state_dim()
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10.0
        self.Q = self._init_process_noise()
        self.initialized = False
        
        # 自适应参数
        self.adaptive_Q = True
        self.innovation_history = []
        self.max_history = 10
    
    def _get_state_dim(self) -> int:
        """获取状态维度"""
        if self.model_type == 'CV':
            return 4  # [x, y, vx, vy]
        elif self.model_type == 'CA':
            return 6  # [x, y, vx, vy, ax, ay]
        elif self.model_type == 'CT':
            return 5  # [x, y, vx, vy, omega]
        elif self.model_type == 'Singer':
            return 6  # [x, y, vx, vy, ax, ay]
        else:
            return 4
    
    def _init_process_noise(self) -> np.ndarray:
        """初始化过程噪声"""
        if self.model_type == 'CV':
            # CV模型：位置和速度噪声
            q = 0.1
            return np.diag([q, q, q*0.1, q*0.1])
        elif self.model_type == 'CA':
            # CA模型：加速度噪声
            q = 0.2
            return np.diag([q, q, q*0.5, q*0.5, q*0.1, q*0.1])
        elif self.model_type == 'CT':
            # CT模型：转弯噪声
            q = 0.3
            return np.diag([q, q, q*0.3, q*0.3, q*0.01])
        elif self.model_type == 'Singer':
            # Singer模型：时间相关加速度噪声
            q = 0.15
            return np.diag([q, q, q*0.2, q*0.2, q*0.05, q*0.05])
        else:
            return np.eye(4) * 0.1
    
    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """预测步骤"""
        if not self.initialized:
            return self.x.copy(), self.P.copy()
        
        # 状态转移
        F = self._get_state_transition_matrix(dt)
        self.x = self._state_transition_function(self.x, dt)
        self.P = F @ self.P @ F.T + self._get_process_noise_matrix(dt)
        
        return self.x.copy(), self.P.copy()

    
    def _get_state_transition_matrix(self, dt: float) -> np.ndarray:
        """获取状态转移矩阵（线性化雅可比）"""
        if self.model_type == 'CV':
            # CV模型：F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
            F = np.eye(4)
            F[0, 2] = dt
            F[1, 3] = dt
            return F
        
        elif self.model_type == 'CA':
            # CA模型：包含加速度
            F = np.eye(6)
            F[0, 2] = dt
            F[0, 4] = 0.5 * dt**2
            F[1, 3] = dt
            F[1, 5] = 0.5 * dt**2
            F[2, 4] = dt
            F[3, 5] = dt
            return F
        
        elif self.model_type == 'CT':
            # CT模型：协调转弯（非线性，需要线性化）
            omega = self.x[4] if len(self.x) > 4 else 0.05
            if abs(omega) < 1e-6:
                # 退化为CV
                F = np.eye(5)
                F[0, 2] = dt
                F[1, 3] = dt
                return F
            
            # 雅可比矩阵
            vx, vy = self.x[2], self.x[3]
            sin_w = np.sin(omega * dt)
            cos_w = np.cos(omega * dt)
            
            F = np.eye(5)
            F[0, 2] = sin_w / omega
            F[0, 3] = -(1 - cos_w) / omega
            F[0, 4] = (vx * (cos_w * dt * omega - sin_w) + vy * (sin_w * dt * omega + cos_w - 1)) / omega**2
            
            F[1, 2] = (1 - cos_w) / omega
            F[1, 3] = sin_w / omega
            F[1, 4] = (vx * (-sin_w * dt * omega - cos_w + 1) + vy * (cos_w * dt * omega - sin_w)) / omega**2
            
            F[2, 2] = cos_w
            F[2, 3] = -sin_w
            F[2, 4] = -vx * sin_w * dt - vy * cos_w * dt
            
            F[3, 2] = sin_w
            F[3, 3] = cos_w
            F[3, 4] = vx * cos_w * dt - vy * sin_w * dt
            
            return F
        
        elif self.model_type == 'Singer':
            # Singer模型：一阶马尔可夫加速度
            alpha = 0.5  # 时间常数倒数 (1/τ)
            exp_alpha = np.exp(-alpha * dt)
            
            F = np.eye(6)
            F[0, 2] = dt
            F[0, 4] = (1 - exp_alpha) / alpha - dt
            F[1, 3] = dt
            F[1, 5] = (1 - exp_alpha) / alpha - dt
            F[2, 4] = (1 - exp_alpha) / alpha
            F[3, 5] = (1 - exp_alpha) / alpha
            F[4, 4] = exp_alpha
            F[5, 5] = exp_alpha
            return F
        
        else:
            return np.eye(self.state_dim)
    
    def _state_transition_function(self, x: np.ndarray, dt: float) -> np.ndarray:
        """非线性状态转移函数"""
        x_new = x.copy()
        
        if self.model_type == 'CV':
            # x_new = [x + vx*dt, y + vy*dt, vx, vy]
            x_new[0] += x[2] * dt
            x_new[1] += x[3] * dt
        
        elif self.model_type == 'CA':
            # x_new = [x + vx*dt + 0.5*ax*dt^2, ..., vx + ax*dt, ..., ax, ay]
            x_new[0] += x[2] * dt + 0.5 * x[4] * dt**2
            x_new[1] += x[3] * dt + 0.5 * x[5] * dt**2
            x_new[2] += x[4] * dt
            x_new[3] += x[5] * dt
        
        elif self.model_type == 'CT':
            # 协调转弯非线性方程
            omega = x[4] if len(x) > 4 else 0.05
            if abs(omega) < 1e-6:
                x_new[0] += x[2] * dt
                x_new[1] += x[3] * dt
            else:
                sin_w = np.sin(omega * dt)
                cos_w = np.cos(omega * dt)
                x_new[0] += (x[2] * sin_w - x[3] * (1 - cos_w)) / omega
                x_new[1] += (x[2] * (1 - cos_w) + x[3] * sin_w) / omega
                x_new[2] = x[2] * cos_w - x[3] * sin_w
                x_new[3] = x[2] * sin_w + x[3] * cos_w
        
        elif self.model_type == 'Singer':
            # Singer一阶马尔可夫加速度模型
            alpha = 0.5
            exp_alpha = np.exp(-alpha * dt)
            x_new[0] += x[2] * dt + x[4] * ((1 - exp_alpha) / alpha - dt)
            x_new[1] += x[3] * dt + x[5] * ((1 - exp_alpha) / alpha - dt)
            x_new[2] += x[4] * (1 - exp_alpha) / alpha
            x_new[3] += x[5] * (1 - exp_alpha) / alpha
            x_new[4] *= exp_alpha
            x_new[5] *= exp_alpha
        
        return x_new
    
    def _get_process_noise_matrix(self, dt: float) -> np.ndarray:
        """获取过程噪声矩阵（自适应）"""
        if self.adaptive_Q and len(self.innovation_history) > 3:
            # 基于创新序列自适应调整Q
            innovations = np.array(self.innovation_history[-5:])
            innovation_cov = np.cov(innovations.T)
            scale = np.trace(innovation_cov) / np.trace(self.Q[:2, :2])
            scale = np.clip(scale, 0.5, 2.0)
            return self.Q * scale * dt
        else:
            return self.Q * dt
    
    def update(self, z: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """更新步骤
        
        Returns:
            (更新后状态, 更新后协方差, 创新)
        """
        if not self.initialized:
            # 初始化
            self.x[:2] = z
            self.initialized = True
            return self.x.copy(), self.P.copy(), np.zeros(2)
        
        # 观测矩阵 H = [[1, 0, 0, ...], [0, 1, 0, ...]]
        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        
        # 创新
        innovation = z - H @ self.x
        
        # 创新协方差
        S = H @ self.P @ H.T + R
        
        # 卡尔曼增益
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((self.state_dim, 2))
        
        # 状态更新
        self.x = self.x + K @ innovation
        
        # 协方差更新（Joseph形式，数值稳定）
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # 保存创新历史（用于自适应）
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)
        
        return self.x.copy(), self.P.copy(), innovation
    
    def get_likelihood(self, z: np.ndarray, R: np.ndarray) -> float:
        """计算测量似然"""
        if not self.initialized:
            return 1.0
        
        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        
        try:
            det_S = np.linalg.det(S)
            if det_S < 1e-10:
                return 1e-10
            inv_S = np.linalg.inv(S)
            exp_term = -0.5 * innovation.T @ inv_S @ innovation
            likelihood = np.exp(exp_term) / np.sqrt((2 * np.pi)**2 * det_S)
            return max(likelihood, 1e-10)
        except:
            return 1e-10


class IMMEKFEnhanced:
    """IMM-EKF增强版 - 算法1.1完整实现
    
    严格按照规范文档实现：
    1. 4个模型：CV/CT/CA/Singer
    2. 模型转移概率矩阵
    3. 混合→预测→更新→融合
    4. 自适应功能
    """
    
    def __init__(self, dt: float = 0.2):
        self.dt = dt
        self.models = ['CV', 'CT', 'CA', 'Singer']
        self.n_models = len(self.models)
        
        # 模型转移概率矩阵（规范公式1.5）
        self.TPM = np.array([
            [0.90, 0.03, 0.04, 0.03],  # CV
            [0.03, 0.90, 0.04, 0.03],  # CA
            [0.03, 0.03, 0.90, 0.04],  # CT
            [0.03, 0.03, 0.04, 0.90]   # Singer
        ])
        
        # 初始化各模型
        self.filters: Dict[str, EKFModel] = {
            model: EKFModel(model, dt) for model in self.models
        }
        
        # 模型概率（初始偏向CV）
        self.model_probs = np.array([0.7, 0.1, 0.1, 0.1])
        
        # 融合状态
        self.fused_x = np.zeros(4)  # [x, y, vx, vy]
        self.fused_P = np.eye(4) * 10.0
        
        self.initialized = False
        self.last_update_time = 0.0
    
    def update(self, z: np.ndarray, R: np.ndarray, timestamp: float) -> IMMState:
        """IMM-EKF完整更新（算法1.1）
        
        Args:
            z: 观测 [x, y]
            R: 观测噪声协方差 2x2
            timestamp: 时间戳
        
        Returns:
            IMMState包含融合状态和各模型信息
        """
        dt = timestamp - self.last_update_time if self.initialized else self.dt
        dt = max(dt, 0.01)  # 防止dt过小
        
        if not self.initialized:
            # 初始化
            for model_name, filter in self.filters.items():
                filter.x[:2] = z
                filter.initialized = True
            self.fused_x[:2] = z
            self.initialized = True
            self.last_update_time = timestamp
            
            return IMMState(
                x=self.fused_x.copy(),
                P=self.fused_P.copy(),
                model_states={m: self.filters[m].x.copy() for m in self.models},
                model_covs={m: self.filters[m].P.copy() for m in self.models},
                model_probs=self.model_probs.copy(),
                timestamp=timestamp
            )
        
        # ===== 步骤1：模型混合（算法1.1步骤1-5）=====
        mixed_states, mixed_covs = self._model_mixing()
        
        # ===== 步骤2：各模型预测和更新（算法1.1步骤6-10）=====
        likelihoods = np.zeros(self.n_models)
        innovations = {}
        
        for i, model_name in enumerate(self.models):
            filter = self.filters[model_name]
            
            # 设置混合初值
            filter.x = mixed_states[i]
            filter.P = mixed_covs[i]
            
            # 预测
            filter.predict(dt)
            
            # 更新
            _, _, innovation = filter.update(z, R)
            innovations[model_name] = innovation
            
            # 计算似然
            likelihoods[i] = filter.get_likelihood(z, R)
        
        # ===== 步骤3：模型概率更新（算法1.1步骤11）=====
        self.model_probs = self._update_model_probabilities(likelihoods)
        
        # ===== 步骤4：状态融合（算法1.1步骤12-13）=====
        self.fused_x, self.fused_P = self._fuse_states()
        
        self.last_update_time = timestamp
        
        return IMMState(
            x=self.fused_x.copy(),
            P=self.fused_P.copy(),
            model_states={m: self.filters[m].x.copy() for m in self.models},
            model_covs={m: self.filters[m].P.copy() for m in self.models},
            model_probs=self.model_probs.copy(),
            timestamp=timestamp
        )

    
    def _model_mixing(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """模型混合（算法1.1步骤1-5）"""
        mixed_states = []
        mixed_covs = []
        
        # 计算混合权重归一化常数 c_bar(j)
        c_bar = self.TPM.T @ self.model_probs  # 步骤2
        
        for j in range(self.n_models):
            if c_bar[j] < 1e-10:
                # 避免除零
                mixed_states.append(self.filters[self.models[j]].x.copy())
                mixed_covs.append(self.filters[self.models[j]].P.copy())
                continue
            
            # 计算混合概率 μ_{k-1}^{(i|j)} (步骤3)
            mu_ij = np.zeros(self.n_models)
            for i in range(self.n_models):
                mu_ij[i] = self.TPM[i, j] * self.model_probs[i] / c_bar[j]
            
            # 混合状态 x_{0,j} (步骤4)
            # 需要统一状态维度到最大维度
            max_dim = max(self.filters[m].state_dim for m in self.models)
            mixed_x = np.zeros(max_dim)
            
            for i in range(self.n_models):
                x_i = self.filters[self.models[i]].x
                # 扩展到统一维度
                x_i_ext = np.zeros(max_dim)
                x_i_ext[:len(x_i)] = x_i
                mixed_x += mu_ij[i] * x_i_ext
            
            # 截取到目标模型维度
            target_dim = self.filters[self.models[j]].state_dim
            mixed_x = mixed_x[:target_dim]
            
            # 混合协方差 P_{0,j} (步骤5)
            mixed_P = np.zeros((target_dim, target_dim))
            for i in range(self.n_models):
                x_i = self.filters[self.models[i]].x
                P_i = self.filters[self.models[i]].P
                
                # 扩展到目标维度
                x_i_ext = np.zeros(target_dim)
                x_i_ext[:min(len(x_i), target_dim)] = x_i[:min(len(x_i), target_dim)]
                
                P_i_ext = np.zeros((target_dim, target_dim))
                min_dim = min(P_i.shape[0], target_dim)
                P_i_ext[:min_dim, :min_dim] = P_i[:min_dim, :min_dim]
                
                # 误差外积
                diff = x_i_ext - mixed_x
                mixed_P += mu_ij[i] * (P_i_ext + np.outer(diff, diff))
            
            mixed_states.append(mixed_x)
            mixed_covs.append(mixed_P)
        
        return mixed_states, mixed_covs
    
    def _update_model_probabilities(self, likelihoods: np.ndarray) -> np.ndarray:
        """更新模型概率（算法1.1步骤11）"""
        # c_bar(j) = Σ_i π_{ij} μ_{k-1}^{(i)}
        c_bar = self.TPM.T @ self.model_probs
        
        # μ_k^{(j)} = Λ_j c_bar(j) / Σ_ℓ Λ_ℓ c_bar(ℓ)
        numerator = likelihoods * c_bar
        denominator = np.sum(numerator)
        
        if denominator < 1e-10:
            # 避免除零，保持原概率
            return self.model_probs
        
        new_probs = numerator / denominator
        
        # 确保概率和为1
        new_probs = new_probs / np.sum(new_probs)
        
        return new_probs
    
    def _fuse_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """状态融合（算法1.1步骤12-13）"""
        # 融合状态 x̂_k = Σ_j μ_k^{(j)} x̂_k^{(j)} (步骤12)
        fused_x = np.zeros(4)  # 统一到4维 [x, y, vx, vy]
        
        for i, model_name in enumerate(self.models):
            x_i = self.filters[model_name].x
            # 提取位置和速度
            fused_x[0] += self.model_probs[i] * x_i[0]  # x
            fused_x[1] += self.model_probs[i] * x_i[1]  # y
            fused_x[2] += self.model_probs[i] * x_i[2]  # vx
            fused_x[3] += self.model_probs[i] * x_i[3]  # vy
        
        # 融合协方差 P_k = Σ_j μ_k^{(j)} [P_k^{(j)} + (x̂_k^{(j)} - x̂_k)(·)^T] (步骤13)
        fused_P = np.zeros((4, 4))
        
        for i, model_name in enumerate(self.models):
            x_i = self.filters[model_name].x
            P_i = self.filters[model_name].P
            
            # 提取4x4子矩阵
            P_i_sub = np.zeros((4, 4))
            min_dim = min(P_i.shape[0], 4)
            P_i_sub[:min_dim, :min_dim] = P_i[:min_dim, :min_dim]
            
            # 状态差
            x_i_sub = np.zeros(4)
            x_i_sub[:min_dim] = x_i[:min_dim]
            diff = x_i_sub - fused_x
            
            # 累加
            fused_P += self.model_probs[i] * (P_i_sub + np.outer(diff, diff))
        
        return fused_x, fused_P
    
    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """无量测预测（用于算法2.5）
        
        Args:
            dt: 预测时间步长
        
        Returns:
            (预测状态, 预测协方差)
        """
        if not self.initialized:
            return self.fused_x.copy(), self.fused_P.copy()
        
        # 各模型预测
        for model_name in self.models:
            self.filters[model_name].predict(dt)
        
        # 融合预测结果
        fused_x, fused_P = self._fuse_states()
        
        return fused_x, fused_P
    
    def get_position_covariance(self) -> np.ndarray:
        """获取位置协方差子矩阵（用于算法2.5）"""
        return self.fused_P[:2, :2]
    
    def get_model_probabilities(self) -> Dict[str, float]:
        """获取各模型概率"""
        return {model: prob for model, prob in zip(self.models, self.model_probs)}
    
    def get_dominant_model(self) -> str:
        """获取主导模型"""
        idx = np.argmax(self.model_probs)
        return self.models[idx]


# ===== 使用示例 =====
if __name__ == "__main__":
    print("IMM-EKF增强版测试")
    print("=" * 60)
    
    # 初始化
    imm = IMMEKFEnhanced(dt=0.2)
    
    # 模拟观测序列（目标做转弯机动）
    true_trajectory = []
    observations = []
    
    # 生成真实轨迹（协调转弯）
    x, y, vx, vy = 100.0, 200.0, 0.0, -0.3  # 初始状态
    omega = 0.05  # 转弯率 rad/s
    dt = 0.2
    
    for t in np.arange(0, 20, dt):
        # 真实运动（CT模型）
        x += (vx * np.sin(omega * dt) - vy * (1 - np.cos(omega * dt))) / omega
        y += (vx * (1 - np.cos(omega * dt)) + vy * np.sin(omega * dt)) / omega
        vx_new = vx * np.cos(omega * dt) - vy * np.sin(omega * dt)
        vy_new = vx * np.sin(omega * dt) + vy * np.cos(omega * dt)
        vx, vy = vx_new, vy_new
        
        true_trajectory.append([x, y, vx, vy])
        
        # 添加观测噪声
        z = np.array([x, y]) + np.random.normal(0, 2.5, 2)
        observations.append(z)
    
    # IMM-EKF滤波
    R = np.eye(2) * 2.5**2  # 观测噪声协方差
    
    print("\n时间  |  真实位置  |  估计位置  |  误差  |  主导模型  |  模型概率")
    print("-" * 80)
    
    for i, (z, true_state) in enumerate(zip(observations, true_trajectory)):
        t = i * dt
        
        # IMM更新
        state = imm.update(z, R, t)
        
        # 计算误差
        pos_error = np.sqrt((state.x[0] - true_state[0])**2 + (state.x[1] - true_state[1])**2)
        
        # 主导模型
        dominant = imm.get_dominant_model()
        probs = imm.get_model_probabilities()
        
        if i % 10 == 0:  # 每2秒打印一次
            print(f"{t:5.1f} | ({true_state[0]:6.1f}, {true_state[1]:6.1f}) | "
                  f"({state.x[0]:6.1f}, {state.x[1]:6.1f}) | {pos_error:5.2f} | "
                  f"{dominant:6s} | CV:{probs['CV']:.2f} CT:{probs['CT']:.2f}")
    
    print("\n测试完成！")
    print(f"最终主导模型: {imm.get_dominant_model()}")
    print(f"最终模型概率: {imm.get_model_probabilities()}")
