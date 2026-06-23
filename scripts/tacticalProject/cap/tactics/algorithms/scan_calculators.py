"""
协同探测V4 - 扫描范围计算器
简化版: arctan(误差/距离)
专业版: 协方差椭圆99%置信区间
"""
import numpy as np
from typing import Optional
from ..detection_types import TargetInfo, FighterInfo


class ArctanCalculator:
    """简化版扫描范围计算 - arctan(误差/距离)"""
    
    def calc_scan_range(self, target: TargetInfo, fighter: FighterInfo,
                        covariance: Optional[np.ndarray] = None) -> float:
        """计算扫描范围"""
        distance = fighter.distance_to(target)
        if distance < 1.0:
            return 30.0
        
        error = target.error_radius
        
        # 基础角度
        base_angle = np.degrees(np.arctan2(error, distance)) * 2
        
        # 机动余量
        maneuver_margin = 2.0
        
        return max(5.0, min(15.0, base_angle + maneuver_margin))


class CovarianceEllipseCalc:
    """专业版扫描范围计算 - 协方差椭圆99%置信区间"""
    
    def calc_scan_range(self, target: TargetInfo, fighter: FighterInfo,
                        covariance: Optional[np.ndarray] = None) -> float:
        """基于协方差椭圆计算99%置信区间覆盖角度"""
        distance = fighter.distance_to(target)
        if distance < 1.0:
            return 30.0
        
        if covariance is None or covariance.shape[0] < 2:
            # 回退到简化版
            return ArctanCalculator().calc_scan_range(target, fighter, None)
        
        # 提取位置协方差 (2x2)
        pos_cov = covariance[:2, :2]
        
        # 确保正定
        pos_cov = (pos_cov + pos_cov.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(pos_cov))
        if min_eig < 0.01:
            pos_cov += np.eye(2) * (0.01 - min_eig)
        
        # 计算椭圆参数
        eigenvalues = np.linalg.eigvalsh(pos_cov)
        
        # 99%置信区间 (chi-square, df=2, p=0.99 → 9.21)
        chi2_99 = 9.21
        
        # 椭圆半轴
        a = np.sqrt(chi2_99 * max(eigenvalues))
        
        # 计算从战机看椭圆的角度范围
        max_angle = np.degrees(np.arctan2(a, distance))
        
        # 安全余量
        return max(5.0, min(20.0, max_angle * 2 + 2.0))
