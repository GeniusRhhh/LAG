"""
协同探测V4验证 - 敌方场景生成器
生成多种敌方行为场景用于验证协同探测算法
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


class FormationType(Enum):
    """编队类型"""
    TIGHT = "tight"           # 密集编队 (间距<5km)
    STANDARD = "standard"     # 标准编队 (间距10-20km)
    SPREAD = "spread"         # 分散编队 (2+2分群, 间距>30km)
    LINE = "line"             # 纵队 (4机纵向排列)
    WIDE = "wide"             # 宽正面 (4机横向排列)


class AltitudeProfile(Enum):
    """高度分布"""
    HIGH = "high"             # 高空 (>10km)
    MEDIUM = "medium"         # 中空 (5-10km)
    LOW = "low"               # 低空 (<3km)
    MIXED = "mixed"           # 混合 (2高+2低)
    STAIRCASE = "staircase"   # 阶梯 (12/10/8/6km)
    POPUP_LOW = "popup_low"   # 低空突防起始（约4.5km），用于早期跃升场景


class ApproachDirection(Enum):
    """来袭方向"""
    FRONT = "front"           # 正面 (180°)
    LEFT_FLANK = "left"       # 左侧 (210°)
    RIGHT_FLANK = "right"     # 右侧 (150°)
    PINCER = "pincer"         # 钳形 (2左+2右)
    OBLIQUE = "oblique"       # 斜向 (165°)


class SpeedProfile(Enum):
    """速度配置 (基于1251.1 fps = 381 m/s)"""
    NORMAL = "normal"         # 普通 1251.1 fps = 381 m/s = 0.381 km/s
    LOW_SPEED = "low"         # 低速 900 fps = 274 m/s = 0.274 km/s
    HIGH_SPEED = "high"       # 高速 1500 fps = 457 m/s = 0.457 km/s
    ACCELERATING = "accel"    # 加速 1100→1400 fps
    MIXED = "mixed"           # 混合 前2机快，后2机慢


class ManeuverType(Enum):
    """Enemy maneuver families used by the verification harness."""

    STRAIGHT = "straight"
    WEAVE = "weave"
    TURN = "turn"
    DIVE = "dive"
    CLIMB = "climb"
    SCATTER = "scatter"
    DECOY = "decoy"
    CROSSING = "crossing"

    EXPAND = "expand"
    FLANK_TURN = "flank_turn"
    POPUP_TURN = "popup_turn"

    CH6_LOW_RISK_FEINT = "ch6_low_risk_feint"
    CH6_MEDIUM_RISK_PRESSURE = "ch6_medium_risk_pressure"
    CH6_HIGH_RISK_PENETRATION = "ch6_high_risk_penetration"

    EARLY_TURN_BACK = "early_turn_back"
    EARLY_SPLIT_TURN_BACK = "early_split_turn_back"
    EARLY_COMPLEX_SEQUENCE = "early_complex_sequence"


class AwacsStatus(Enum):
    """预警机状态（模拟干扰效果）"""
    NORMAL = "normal"              # 正常
    INTERMITTENT = "intermittent"  # 间歇丢失 (每30秒丢失10秒)
    UNAVAILABLE = "unavailable"    # 完全不可用
    DELAYED = "delayed"            # 延迟可用 (100km后丢失)


# 速度常量 (单位: km/s)
SPEED_NORMAL = 0.381      # 1251.1 fps
SPEED_LOW = 0.274         # 900 fps
SPEED_HIGH = 0.457        # 1500 fps


@dataclass
class EnemyScenario:
    """敌方场景配置"""
    scenario_id: str
    formation: FormationType
    altitude: AltitudeProfile
    direction: ApproachDirection
    speed: SpeedProfile
    maneuver: ManeuverType
    awacs_status: AwacsStatus
    difficulty: int = 1  # 1-5星难度
    description: str = ""
    validation_points: str = ""  # 标注该场景主要验证点
    initial_distance: float = 250.0 # 初始距离(km)

    def reset_runtime_state(self) -> None:
        """Reset cached maneuver state before a new verification run."""
        for attr in (
            '_hdg_offsets_deg',
            '_early_stage',
            '_turn_executed',
            '_scatter_executed',
            '_decoy_executed',
            '_crossing_executed',
            '_expand_stage',
            '_flank_stage',
            '_popup_stage',
            '_ch6_phase',
            '_ch6_phase_time',
        ):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
    
    def generate_initial_positions(self, base_y: float = None) -> List[Tuple[float, float, float]]:
        """
        生成4架敌机初始位置 (x, y, altitude) 单位km
        
        Args:
            base_y: 基准Y坐标(距离)，默认None(使用initial_distance)
        """
        # 使用场景配置的距离，除非强制指定
        if base_y is None:
            base_y = self.initial_distance
            
        # 位置偏移 (相对于编队中心)
        if self.formation == FormationType.TIGHT:
            offsets = [(-2, 0), (2, 0), (-2, -3), (2, -3)]
        elif self.formation == FormationType.STANDARD:
            offsets = [(-10, 0), (10, 0), (-10, -10), (10, -10)]
        elif self.formation == FormationType.SPREAD:
            # 2+2 分群分散（左右两群）
            offsets = [(-30, 0), (-30, -10), (30, 0), (30, -10)]
        elif self.formation == FormationType.LINE:
            offsets = [(0, 0), (0, -10), (0, -20), (0, -30)]
        else:  # WIDE
            offsets = [(-60, 0), (-20, 0), (20, 0), (60, 0)]
        
        # 航向调整基准X位置
        if self.direction == ApproachDirection.LEFT_FLANK:
            base_x = -50
        elif self.direction == ApproachDirection.RIGHT_FLANK:
            base_x = 50
        elif self.direction == ApproachDirection.PINCER:
            # 钳形攻击：左右各2架
            offsets = [(-80, 0), (-80, -10), (80, 0), (80, -10)]
            base_x = 0
        else:
            base_x = 0  # FRONT, OBLIQUE
        
        # 高度分布 (单位km)
        # 保留S21用于回归；验证V01-V03使用AltitudeProfile驱动
        if self.scenario_id == 'S21':
            alts = [4.0, 4.0, 4.0, 4.0]
        elif self.altitude == AltitudeProfile.HIGH:
            alts = [12.0, 11.5, 12.0, 11.0]
        elif self.altitude == AltitudeProfile.MEDIUM:
            alts = [8.0, 8.5, 8.0, 8.0]  # B0400 was 7.5km
        elif self.altitude == AltitudeProfile.LOW:
            alts = [2.0, 2.5, 2.0, 1.5]
        elif self.altitude == AltitudeProfile.MIXED:
            # 低于约3.5km在部分机型/初始状态下更容易触发不稳定/NaN，这里抬高低空层
            alts = [11.0, 4.5, 10.0, 4.5]
        elif self.altitude == AltitudeProfile.POPUP_LOW:
            # 低空突防起始高度（不采用LOW档2km级，避免不稳定/NaN）
            alts = [3.5, 3.5, 3.5, 3.5]
        else:  # STAIRCASE
            alts = [12.0, 10.0, 8.0, 6.0]
        
        positions = []
        for i, (dx, dy) in enumerate(offsets):
            positions.append((base_x + dx, base_y + dy, alts[i]))
        
        return positions
    
    def generate_initial_velocities(self) -> List[Tuple[float, float, float]]:
        """
        生成4架敌机初始速度 (vx, vy, vz) 单位km/s
        """
        # 基础速度
        if self.speed == SpeedProfile.NORMAL:
            speeds = [SPEED_NORMAL] * 4
        elif self.speed == SpeedProfile.LOW_SPEED:
            speeds = [SPEED_LOW] * 4
        elif self.speed == SpeedProfile.HIGH_SPEED:
            speeds = [SPEED_HIGH] * 4
        elif self.speed == SpeedProfile.ACCELERATING:
            speeds = [0.335] * 4  # 起始速度
        else:  # MIXED
            speeds = [SPEED_HIGH, SPEED_HIGH, SPEED_LOW, SPEED_LOW]
        
        # 航向 (度，北=0，顺时针)
        if self.direction == ApproachDirection.FRONT:
            headings = [180, 180, 180, 180]
        elif self.direction == ApproachDirection.LEFT_FLANK:
            headings = [210, 210, 210, 210]
        elif self.direction == ApproachDirection.RIGHT_FLANK:
            headings = [150, 150, 150, 150]
        elif self.direction == ApproachDirection.PINCER:
            headings = [150, 150, 210, 210]  # 左侧向右飞，右侧向左飞
        else:  # OBLIQUE
            headings = [165, 165, 165, 165]
        
        velocities = []
        for i in range(4):
            hdg_rad = np.radians(headings[i])
            # 航向180°时: vx=0, vy=-speed (向南飞)
            vx = speeds[i] * np.sin(hdg_rad)
            vy = speeds[i] * np.cos(hdg_rad)  # cos(180)=-1，所以vy为负=向南
            velocities.append((vx, vy, 0.0))
        
        return velocities
    
    def get_initial_headings(self) -> List[float]:
        """获取初始航向 (度)"""
        if self.direction == ApproachDirection.FRONT:
            return [180, 180, 180, 180]
        elif self.direction == ApproachDirection.LEFT_FLANK:
            return [210, 210, 210, 210]
        elif self.direction == ApproachDirection.RIGHT_FLANK:
            return [150, 150, 150, 150]
        elif self.direction == ApproachDirection.PINCER:
            return [150, 150, 210, 210]
        else:  # OBLIQUE
            return [165, 165, 165, 165]
    
    def get_maneuver_commands(self, distance: float, time: float) -> List[Dict]:
        """
        根据当前距离和时间返回机动指令
        
        Args:
            distance: 当前与我方的距离 (km)
            time: 仿真时间 (秒)
        
        Returns:
            4架飞机的机动指令列表
        """
        commands = [{} for _ in range(4)]

        # 内部：记录每架机当前“相对初始航向(180°)”的偏置，便于多阶段机动后回正
        # 说明：heading_change 在 run_detection 侧会被当作“单次偏转指令”处理并更新基础航向。
        if not hasattr(self, '_hdg_offsets_deg') or not isinstance(getattr(self, '_hdg_offsets_deg', None), list):
            self._hdg_offsets_deg = [0.0, 0.0, 0.0, 0.0]

        def _apply_heading_deltas(deltas: List[float]):
            for i in range(4):
                d = float(deltas[i]) if i < len(deltas) else 0.0
                if abs(d) > 1e-6:
                    commands[i]['heading_change'] = d
                self._hdg_offsets_deg[i] = float(self._hdg_offsets_deg[i]) + d

        def _turn_back_to_180(indices: Optional[List[int]] = None):
            """????????180??????????

            ??????4??? indices ????????????????????
            """
            if indices is None:
                indices = [0, 1, 2, 3]

            deltas = [0.0, 0.0, 0.0, 0.0]
            for i in indices:
                try:
                    deltas[i] = -float(self._hdg_offsets_deg[i])
                except Exception:
                    deltas[i] = 0.0

            _apply_heading_deltas(deltas)

            for i in indices:
                self._hdg_offsets_deg[i] = 0.0

        def _set_heading_offsets(target_offsets: List[float]):
            deltas = []
            for i in range(4):
                current = float(self._hdg_offsets_deg[i])
                target = float(target_offsets[i]) if i < len(target_offsets) else current
                deltas.append(target - current)
            _apply_heading_deltas(deltas)

        def _set_target_altitudes(target_altitudes: List[float]):
            for i in range(4):
                target_alt = float(target_altitudes[i]) if i < len(target_altitudes) else None
                if target_alt is not None:
                    commands[i]['target_altitude'] = target_alt

        def _ensure_ch6_state():
            if not hasattr(self, '_ch6_phase'):
                self._ch6_phase = 0
            if not hasattr(self, '_ch6_phase_time'):
                self._ch6_phase_time = float(time)

        def _enter_ch6_phase(phase: int):
            self._ch6_phase = int(phase)
            self._ch6_phase_time = float(time)

        def _ch6_phase_elapsed() -> float:
            return float(time) - float(getattr(self, '_ch6_phase_time', time))
        
        if self.maneuver == ManeuverType.STRAIGHT:
            pass  # ???

        elif self.maneuver == ManeuverType.CH6_LOW_RISK_FEINT:
            # S1: deterministic low-risk presentation.
            # The scene is fixed at the macro level only: mild split, re-center,
            # then an early turn-away before a deep medium-risk push.
            _ensure_ch6_state()
            _set_target_altitudes([8.8, 8.4, 7.6, 7.2])

            if self._ch6_phase == 0 and distance <= 292.0:
                _set_heading_offsets([14.0, 8.0, -8.0, -14.0])
                _enter_ch6_phase(1)
            elif self._ch6_phase == 1 and distance <= 255.0:
                _set_heading_offsets([0.0, 0.0, 0.0, 0.0])
                _enter_ch6_phase(2)
            elif self._ch6_phase == 2 and distance <= 188.0:
                _set_heading_offsets([155.0, 165.0, -165.0, -155.0])
                _enter_ch6_phase(3)
            elif self._ch6_phase == 3 and _ch6_phase_elapsed() >= 28.0:
                _set_heading_offsets([120.0, 130.0, -130.0, -120.0])
                _enter_ch6_phase(4)

        elif self.maneuver == ManeuverType.CH6_MEDIUM_RISK_PRESSURE:
            # S2: repeatable medium-risk pressure under intermittent AWACS.
            # One pair turns out first while the other keeps pressing; both then recommit.
            _ensure_ch6_state()
            _set_target_altitudes([10.5, 6.2, 9.8, 6.0])

            if self._ch6_phase == 0 and distance <= 235.0:
                _set_heading_offsets([38.0, 22.0, -22.0, -38.0])
                _enter_ch6_phase(1)
            elif self._ch6_phase == 1 and distance <= 205.0:
                _set_heading_offsets([14.0, 8.0, -8.0, -14.0])
                _enter_ch6_phase(2)
            elif self._ch6_phase == 2 and distance <= 150.0:
                _set_heading_offsets([160.0, 170.0, -18.0, -28.0])
                _enter_ch6_phase(3)
            elif self._ch6_phase == 3 and _ch6_phase_elapsed() >= 26.0:
                _set_heading_offsets([150.0, 160.0, -165.0, -170.0])
                _enter_ch6_phase(4)
            elif self._ch6_phase == 4 and _ch6_phase_elapsed() >= 38.0:
                _set_heading_offsets([25.0, 12.0, -12.0, -25.0])
                _enter_ch6_phase(5)
            elif self._ch6_phase == 5 and distance <= 130.0:
                _set_heading_offsets([165.0, 170.0, -170.0, -165.0])
                _enter_ch6_phase(6)

        elif self.maneuver == ManeuverType.CH6_HIGH_RISK_PENETRATION:
            # S3: one pair remains low and keeps penetrating long enough to trigger
            # the high-risk branch; the other pair pops up later to raise urgency.
            _ensure_ch6_state()

            if self._ch6_phase < 2:
                _set_target_altitudes([3.5, 3.5, 4.5, 4.5])
            else:
                _set_target_altitudes([3.5, 3.5, 8.5, 8.5])

            if self._ch6_phase == 0 and distance <= 150.0:
                _set_heading_offsets([0.0, 0.0, -14.0, -20.0])
                _enter_ch6_phase(1)
            elif self._ch6_phase == 1 and distance <= 95.0:
                _set_heading_offsets([0.0, 0.0, 6.0, 12.0])
                _enter_ch6_phase(2)
            elif self._ch6_phase == 2 and distance <= 35.0:
                _set_heading_offsets([165.0, 170.0, -170.0, -165.0])
                _set_target_altitudes([5.5, 5.5, 8.5, 8.5])
                _enter_ch6_phase(3)

        elif self.maneuver == ManeuverType.EARLY_TURN_BACK:
            # V01（简单）：两编队“开局括号机动 + 立体高度分层 + 一次单侧钩形 + 回正”
            # 目标：轨迹差异立刻出现，但动作次数少（便于对比 proposed/baseline）。
            if not hasattr(self, '_early_stage'):
                self._early_stage = 0

            # 阶段1（8~12s）：括号机动（两编队反向偏转，立刻拉开轨迹）
            if 8.0 <= time < 12.0 and self._early_stage == 0:
                # 左编队(B01/B02)右偏，右编队(B03/B04)左偏
                _apply_heading_deltas([+75, +65, -65, -75])

                # 轻量立体分层（短促，不持续）：左编队略爬升，右编队略下降
                commands[0]['altitude_rate'] = 0.06
                commands[1]['altitude_rate'] = 0.05
                commands[2]['altitude_rate'] = -0.04
                commands[3]['altitude_rate'] = -0.04
                self._early_stage = 1

            # 阶段2（16~20s）：锁定不同高度层（防止高度速率持续积分）
            elif 16.0 <= time < 20.0 and self._early_stage == 1:
                # 中高度场景默认约8km，这里形成 9km / 7km 分层
                commands[0]['target_altitude'] = 9.5
                commands[1]['target_altitude'] = 9.5
                commands[2]['target_altitude'] = 6.5
                commands[3]['target_altitude'] = 6.5
                self._early_stage = 2

            # 阶段3（38~42s）：仅右编队做一次“钩形”机动（增强轨迹差异，仍低频）
            elif 38.0 <= time < 42.0 and self._early_stage == 2:
                # 右编队在原来左偏基础上反向大幅拉一把，形成明显折线
                _apply_heading_deltas([0, 0, +110, +110])
                self._early_stage = 3

            # 阶段4（95~100s）：仅左编队再做一次反向钩形（增加动作次数/幅度，仍保持可对比）
            elif 95.0 <= time < 100.0 and self._early_stage == 3:
                _apply_heading_deltas([-105, -95, 0, 0])
                self._early_stage = 4

            # 阶段5（140~148s）：全体回正到180°，进入稳定对头
            elif 140.0 <= time < 148.0 and self._early_stage == 4:
                _turn_back_to_180()
                self._early_stage = 5

        elif self.maneuver == ManeuverType.EARLY_SPLIT_TURN_BACK:
            # V02（中等）：宽正面 + 两编队不同步/不同幅度的“分裂-交叉-回正-高度层转换”
            # 目标：让目标中心/方差随时间显著变化，检验 sigma_man/R_target 自适应与覆盖恢复。
            if not hasattr(self, '_early_stage'):
                self._early_stage = 0

            # 阶段1（8~12s）：左右分裂（同编队不同幅度，形成更丰富轨迹）
            if 8.0 <= time < 12.0 and self._early_stage == 0:
                _apply_heading_deltas([+85, +65, -65, -85])
                self._early_stage = 1

            # 阶段2（30~36s）：交叉趋势（向中间拉回），同时做高度层对换
            elif 30.0 <= time < 36.0 and self._early_stage == 1:
                _apply_heading_deltas([-55, -45, +45, +55])
                commands[0]['altitude_rate'] = -0.04
                commands[1]['altitude_rate'] = -0.03
                commands[2]['altitude_rate'] = 0.04
                commands[3]['altitude_rate'] = 0.03
                self._early_stage = 2

            # 阶段3（42~48s）：锁定高度层（防止持续积分漂移）
            elif 42.0 <= time < 48.0 and self._early_stage == 2:
                # 右编队更高，左编队更低（高度层对换完成）
                commands[0]['target_altitude'] = 7.0
                commands[1]['target_altitude'] = 7.0
                commands[2]['target_altitude'] = 10.0
                commands[3]['target_altitude'] = 10.0
                self._early_stage = 3

            # 阶段4（75~82s）：全体回正到180°
            elif 75.0 <= time < 82.0 and self._early_stage == 3:
                _turn_back_to_180()
                self._early_stage = 4

            # 阶段5（100~106s）：第二轮分裂（更大角度，增加动作次数/幅度）
            elif 100.0 <= time < 106.0 and self._early_stage == 4:
                _apply_heading_deltas([+95, +75, -75, -95])
                commands[0]['altitude_rate'] = 0.05
                commands[1]['altitude_rate'] = 0.04
                commands[2]['altitude_rate'] = -0.04
                commands[3]['altitude_rate'] = -0.05
                self._early_stage = 5

            # 阶段6（112~118s）：锁定高度层（避免持续积分漂移）
            elif 112.0 <= time < 118.0 and self._early_stage == 5:
                commands[0]['target_altitude'] = 9.0
                commands[1]['target_altitude'] = 9.0
                commands[2]['target_altitude'] = 6.5
                commands[3]['target_altitude'] = 6.5
                self._early_stage = 6

            # 阶段7（132~140s）：第二轮交叉趋势（向中间拉回）
            elif 132.0 <= time < 140.0 and self._early_stage == 6:
                _apply_heading_deltas([-85, -65, +65, +85])
                self._early_stage = 7

            # 阶段8（170~178s）：最终回正到180°
            elif 170.0 <= time < 178.0 and self._early_stage == 7:
                _turn_back_to_180()
                self._early_stage = 8

        elif self.maneuver == ManeuverType.EARLY_COMPLEX_SEQUENCE:
            # V03（复杂）：低空起始 + 两编队异步跃升/交叉/回正（配合AWACS间歇丢失）。
            # 目标：制造“方位双峰 + 高度层变化 + 短时目标缺失”组合，检验协同探测对覆盖恢复/首探全探的收益。
            if not hasattr(self, '_early_stage'):
                self._early_stage = 0

            # 阶段1（8~12s）：强分裂偏转（建立方位双峰）
            if 8.0 <= time < 12.0 and self._early_stage == 0:
                _apply_heading_deltas([+95, +85, -85, -95])
                self._early_stage = 1

            # 阶段2（24~32s）：左编队先跃升+折线，右编队保持低空并继续扩散（异步）
            elif 24.0 <= time < 32.0 and self._early_stage == 1:
                # 左编队：向内折一点并跃升
                _apply_heading_deltas([-55, -45, 0, 0])
                commands[0]['altitude_rate'] = 0.10
                commands[1]['altitude_rate'] = 0.10

                # 右编队：继续外扩一点（增强分离），高度保持
                _apply_heading_deltas([0, 0, -20, -20])
                self._early_stage = 2

            # 阶段3（36~44s）：右编队延迟跃升+交叉趋势
            elif 36.0 <= time < 44.0 and self._early_stage == 2:
                _apply_heading_deltas([0, 0, +70, +60])
                commands[2]['altitude_rate'] = 0.10
                commands[3]['altitude_rate'] = 0.10
                self._early_stage = 3

            # 阶段4（50~56s）：锁定目标高度（防止持续积分漂移）
            elif 50.0 <= time < 56.0 and self._early_stage == 3:
                # 统一跃升到9km（便于后续稳定飞行与对比）
                for i in range(4):
                    commands[i]['target_altitude'] = 9.0
                self._early_stage = 4

            # 阶段5（70~76s）：左编队先回正，右编队继续机动一会儿（保持两编队轨迹差异）
            elif 70.0 <= time < 76.0 and self._early_stage == 4:
                _turn_back_to_180(indices=[0, 1])
                self._early_stage = 5

            # 阶段6（86~92s）：右编队回正，态势收敛
            elif 86.0 <= time < 92.0 and self._early_stage == 5:
                _turn_back_to_180(indices=[2, 3])
                self._early_stage = 6

            # 阶段7（120~126s）：第二轮强分裂偏转（更大角度，增加动作次数/幅度）
            elif 120.0 <= time < 126.0 and self._early_stage == 6:
                _apply_heading_deltas([+115, +95, -95, -115])
                # 小幅高度层分离，避免过低高度导致不稳定
                commands[0]['target_altitude'] = 10.0
                commands[1]['target_altitude'] = 10.0
                commands[2]['target_altitude'] = 8.0
                commands[3]['target_altitude'] = 8.0
                self._early_stage = 7

            # 阶段8（150~156s）：交叉趋势（向中间拉回），强化折线与双峰合并过程
            elif 150.0 <= time < 156.0 and self._early_stage == 7:
                _apply_heading_deltas([-85, -70, +70, +85])
                self._early_stage = 8

            # 阶段9（190~198s）：全体回正到180°，后续进入稳定对头
            elif 190.0 <= time < 198.0 and self._early_stage == 8:
                _turn_back_to_180()
                self._early_stage = 9

            # 阶段10：稳定阶段，持续保持目标高度（低频，仅保持）
            elif self._early_stage >= 9:
                for i in range(4):
                    commands[i] = {'target_altitude': 9.0}
        
        elif self.maneuver == ManeuverType.WEAVE:
            # S形摆动：周期约60秒，±30°
            phase = np.sin(time * 0.1)
            for i in range(4):
                commands[i] = {'heading_change': 30 * phase}
        
        elif self.maneuver == ManeuverType.TURN:
            # 战术转向：单次动作
            # 使用静态属性记录是否已执行（注意：这是一个简单hack，理想情况应由外部控制）
            # 但由于ScenarioGenerator是重新实例化的，或者这个方法被循环调用
            # 更好的做法是：只在进入区间的 *第一帧* 发指令
            # 这里通过检查距离区间的边缘来实现
            
            # 在180km处转向60° (区间设窄一点，或者只在进入时触发)
            # 假设速度0.38km/s，仿真步长0.2s -> 每步走0.076km
            # 只要区间小于这个距离，就不会重复 triggering
            # 但为了稳健，我们使用 getattr 检查实例状态
            
            if not hasattr(self, '_turn_executed'):
                self._turn_executed = False
            
            if 175 < distance < 185 and not self._turn_executed:
                commands = [
                    {'heading_change': 60},
                    {'heading_change': 60},
                    {'heading_change': -60},
                    {'heading_change': -60}
                ]
                self._turn_executed = True
        
        elif self.maneuver == ManeuverType.DIVE:
             # 俯冲 (持续动作设定速率即可，无需状态锁定)
            if distance < 140:
                for i in range(4):
                    commands[i] = {'altitude_rate': -0.1}  # km/s下降率
        
        elif self.maneuver == ManeuverType.CLIMB:
            if distance < 130:
                for i in range(4):
                    commands[i] = {'altitude_rate': 0.08}
        
        elif self.maneuver == ManeuverType.SCATTER:
            if not hasattr(self, '_scatter_executed'):
                self._scatter_executed = False

            if 125 < distance < 135 and not self._scatter_executed:
                commands = [
                    {'heading_change': -30},
                    {'heading_change': 30},
                    {'heading_change': -45},
                    {'heading_change': 45}
                ]
                self._scatter_executed = True
        
        elif self.maneuver == ManeuverType.DECOY:
            if not hasattr(self, '_decoy_executed'):
                self._decoy_executed = False

            if distance < 160 and not self._decoy_executed:
                commands[0] = {'heading_change': 90}
                self._decoy_executed = True
        
        elif self.maneuver == ManeuverType.CROSSING:
            if not hasattr(self, '_crossing_executed'):
                self._crossing_executed = False

            if 140 < distance < 150 and not self._crossing_executed:
                commands = [
                    {'heading_change': 30},
                    {'heading_change': -30},
                    {'heading_change': 30},
                    {'heading_change': -30}
                ]
                self._crossing_executed = True
        
        elif self.maneuver == ManeuverType.EXPAND:
            # [场景1] 疏开机动: 280km处僚机外转20度，220km由CAPTask自动恢复(因为是 heading_change)
            # 这里我们设计为: 280km触发外转，220km触发回转
            # 注意: get_maneuver_commands返回的是delta或rate，run_detection里是累积还是单次?
            # run_detection里: maneuver != WEAVE时，heading_change只执行一次(通过_turn_executed标记)
            # 但这里我们需要两阶段动作，所以不能只依赖简单的_turn_executed
            
            # [场景1] 增强版疏开机动: 多阶段立体机动
            if not hasattr(self, '_expand_stage'):
                self._expand_stage = 0
            
            # Phase 1: 疏开 + 高度分离 (280km)
            # 左组爬升左转，右组轻微下降右转（减小下降速率避免坠毁）
            if 275 < distance < 285 and self._expand_stage == 0:
                commands[0] = {'heading_change': -20, 'altitude_rate': 0.03} # B01 Left + Climb
                commands[2] = {'heading_change': -20, 'altitude_rate': 0.03} # B03 Left + Climb
                commands[1] = {'heading_change': 20, 'altitude_rate': -0.02} # B02 Right + Slight Dive (减小)
                commands[3] = {'heading_change': 20, 'altitude_rate': -0.02} # B04 Right + Slight Dive (减小)
                self._expand_stage = 1
            
            # Phase 1.5: 停止高度变化 (250km)
            # 经过约30km (约75s)，高度变化约 3.75km
            elif 245 < distance < 255 and self._expand_stage == 1:
                for i in range(4):
                    commands[i] = {'altitude_rate': 0.0}
                self._expand_stage = 2

            # Phase 2: 蛇形内切 (230km)
            # 向内急转40度，形成交叉趋势
            elif 225 < distance < 235 and self._expand_stage == 2:
                commands[0] = {'heading_change': 40} # B01 Right
                commands[2] = {'heading_change': 40} # B03 Right
                commands[1] = {'heading_change': -40} # B02 Left
                commands[3] = {'heading_change': -40} # B04 Left
                self._expand_stage = 3
            
            # Phase 3: 恢复平行 (190km)
            elif 185 < distance < 195 and self._expand_stage == 3:
                commands[0] = {'heading_change': -20}
                commands[2] = {'heading_change': -20}
                commands[1] = {'heading_change': 20}
                commands[3] = {'heading_change': 20}
                self._expand_stage = 4

        elif self.maneuver == ManeuverType.FLANK_TURN:
            # [场景2] 增强侧翼机动: 大角度偏置+高度变化（7阶段立体机动）
            # 
            # 编队定义：
            # - 编队1: B0100 + B0200 (左侧编队)
            # - 编队2: B0300 + B0400 (右侧编队)
            # 
            # 设计原则：
            # 1. 大角度偏置（45度），明显可见
            # 2. 配合高度变化，形成立体机动
            # 3. 7阶段：编队1右转+爬升 → 回正 → 编队2左转+下降 → 回正 → 高度恢复 → 停止
            if not hasattr(self, '_flank_stage'):
                self._flank_stage = 0
                
            # 阶段1: 编队1(B0100/B0200)大角度右转45° + 爬升 (400km) 180 -> 225
            if 395 < distance < 405 and self._flank_stage == 0:
                commands[0] = {'heading_change': 45, 'altitude_rate': 0.05}   # B0100 右转+爬升
                commands[1] = {'heading_change': 45, 'altitude_rate': 0.05}   # B0200 右转+爬升
                # B0300/B0400保持180度
                self._flank_stage = 1
            
            # 阶段2: 编队1回正到180度 (360km) 225 -> 180，停止爬升
            elif 355 < distance < 365 and self._flank_stage == 1:
                commands[0] = {'heading_change': -45, 'altitude_rate': 0.0}  # B0100 左转回正+停止爬升
                commands[1] = {'heading_change': -45, 'altitude_rate': 0.0}  # B0200 左转回正+停止爬升
                self._flank_stage = 2
            
            # 阶段3: 编队2(B0300/B0400)大角度左转45° + 下降 (320km) 180 -> 135
            elif 315 < distance < 325 and self._flank_stage == 2:
                commands[2] = {'heading_change': -45, 'altitude_rate': -0.03}  # B0300 左转+轻微下降
                commands[3] = {'heading_change': -45, 'altitude_rate': -0.03}  # B0400 左转+轻微下降
                self._flank_stage = 3
            
            # 阶段4: 编队2回正到180度 (280km) 135 -> 180，停止下降
            elif 275 < distance < 285 and self._flank_stage == 3:
                commands[2] = {'heading_change': 45, 'altitude_rate': 0.0}   # B0300 右转回正+停止下降
                commands[3] = {'heading_change': 45, 'altitude_rate': 0.0}   # B0400 右转回正+停止下降
                self._flank_stage = 4
            
            # 阶段5: 编队1恢复初始高度 (240km) 下降回到初始高度
            elif 235 < distance < 245 and self._flank_stage == 4:
                commands[0] = {'altitude_rate': -0.05}  # B0100 下降恢复
                commands[1] = {'altitude_rate': -0.05}  # B0200 下降恢复
                self._flank_stage = 5
            
            # 阶段6: 编队2恢复初始高度 (200km) 爬升回到初始高度
            elif 195 < distance < 205 and self._flank_stage == 5:
                commands[2] = {'altitude_rate': 0.03}   # B0300 爬升恢复
                commands[3] = {'altitude_rate': 0.03}   # B0400 爬升恢复
                self._flank_stage = 6
            
            # 阶段7: 停止所有高度变化 (160km)
            elif 155 < distance < 165 and self._flank_stage == 6:
                for i in range(4):
                    commands[i] = {'altitude_rate': 0.0}
                self._flank_stage = 7
            
            # 完成：保持180度直飞，高度稳定
                
        elif self.maneuver == ManeuverType.POPUP_TURN:
            # [场景3] 低空突防跃升: 230km处快速爬升至9km并左转15度
            # 
            # 设计原则：
            # 1. 初始3.5km低空潜伏，避开雷达探测
            # 2. 230km触发快速拉升至9km（与我方同高度）
            # 3. 同时左转15度，形成突防态势
            # 4. 使用altitude_rate实现快速爬升（与V02保持一致）
            # 5. ✅ 修复：爬升完成后明确设置目标高度，防止AI误下降导致失速
            # 6. ✅ 修复V2：爬升完成后持续发送target_altitude，覆盖AI的下降指令
            if not hasattr(self, '_popup_stage'):
                self._popup_stage = 0
            
            # 🔥 诊断日志：追踪场景状态
            import os
            if os.environ.get('V03_ALTITUDE_DEBUG') == '1' and int(time_s * 5) % 10 == 0:
                import logging
                logging.info(f"[V03诊断-场景] time={time_s:.1f}s distance={distance:.1f}km stage={self._popup_stage}")
            
            # 阶段1：持续快速跃升 (230km ~ 170km)
            # 目标: 3.5km -> 9km, delta = 5.5km
            # 爬升速率: 0.10 km/s = 328 ft/s (快速爬升)
            # 预计时间: 55秒，距离变化: 230km - 21km = 209km
            if 170 <= distance < 230 and self._popup_stage == 0:
                for i in range(4):
                    commands[i] = {
                        'heading_change': -15,      # 左转15度
                        'altitude_rate': 0.10,      # 持续快速爬升 0.10 km/s
                    }
                if os.environ.get('V03_ALTITUDE_DEBUG') == '1' and int(time_s * 5) % 10 == 0:
                    import logging
                    logging.info(f"[V03诊断-场景] 阶段1爬升: altitude_rate=0.10 km/s")
            
            # 阶段2：到达目标高度后停止爬升并保持 (< 170km)
            # ✅ 修复：明确设置目标高度9km，防止AI接管后误下降
            elif distance < 170 and self._popup_stage == 0:
                for i in range(4):
                    commands[i] = {
                        'altitude_rate': 0.0,       # 停止爬升速率
                        'target_altitude': 9.0,     # ✅ 明确目标高度9km
                    }
                self._popup_stage = 1  # 标记完成
                import logging
                logging.info(f"[V03诊断-场景] ✅ 阶段2完成: 设置target_altitude=9.0, stage={self._popup_stage}")
            
            # 阶段3：爬升完成后持续保持目标高度 (< 170km, stage=1)
            # ✅ 修复V2：持续发送target_altitude，覆盖AI可能发出的下降指令
            elif distance < 170 and self._popup_stage == 1:
                for i in range(4):
                    commands[i] = {
                        'target_altitude': 9.0,     # ✅ 持续发送，覆盖AI指令
                    }
                if os.environ.get('V03_ALTITUDE_DEBUG') == '1' and int(time_s * 5) % 10 == 0:
                    import logging
                    logging.info(f"[V03诊断-场景] 阶段3保持: 持续发送target_altitude=9.0")

        return commands
    
    def is_awacs_available(self, time: float, distance: float) -> bool:
        """
        判断预警机信息是否可用
        
        Args:
            time: 仿真时间 (秒)
            distance: 当前距我方距离 (km)
        """
        if self.awacs_status == AwacsStatus.NORMAL:
            return True
        elif self.awacs_status == AwacsStatus.INTERMITTENT:
            # 每30秒丢失10秒
            return (time % 30) < 20
        elif self.awacs_status == AwacsStatus.UNAVAILABLE:
            return False
        elif self.awacs_status == AwacsStatus.DELAYED:
            # 100km后丢失
            return distance > 100
        return True


class ScenarioGenerator:
    """场景生成器"""
    
    @staticmethod
    def get_core_scenarios() -> List[EnemyScenario]:
        """生成验证场景（仅V01-V03）"""
        scenarios = []
        
        # ========== 验证报告专用场景 (V01-V03) ==========
        # V01：基准对头接敌（检验协调收益）
        scenarios.append(EnemyScenario(
            scenario_id="V01", formation=FormationType.STANDARD,
            altitude=AltitudeProfile.MEDIUM, direction=ApproachDirection.FRONT,
            speed=SpeedProfile.NORMAL, maneuver=ManeuverType.EARLY_TURN_BACK,
            awacs_status=AwacsStatus.NORMAL, difficulty=1,
            initial_distance=320.0,
            description="[V01] 简单：对头来袭(中高度) + 两编队括号机动(开局) + 9/7km分层 + 单侧钩形 + 回正（检验首探/全探与稳态收益）"
        ))

        # V02：机动不确定性（检验sigma_man/R_target自适应与搜索恢复能力）
        scenarios.append(EnemyScenario(
            scenario_id="V02", formation=FormationType.WIDE,
            altitude=AltitudeProfile.MEDIUM, direction=ApproachDirection.FRONT,
            speed=SpeedProfile.NORMAL, maneuver=ManeuverType.EARLY_SPLIT_TURN_BACK,
            awacs_status=AwacsStatus.NORMAL, difficulty=3,
            initial_distance=320.0,
            description="[V02] 中等：对头来袭+宽正面 + 分裂→交叉趋势→高度层对换(7/10km)→回正（检验覆盖自适应/恢复能力）"
        ))

        # V03：复杂态势（双峰方位+混合高度），同时压测AWACS低空掉线鲁棒性
        scenarios.append(EnemyScenario(
            scenario_id="V03", formation=FormationType.WIDE,
            altitude=AltitudeProfile.POPUP_LOW, direction=ApproachDirection.FRONT,
            speed=SpeedProfile.NORMAL, maneuver=ManeuverType.EARLY_COMPLEX_SEQUENCE,
            awacs_status=AwacsStatus.INTERMITTENT, difficulty=4,
            initial_distance=320.0,
            description="[V03] 复杂：低空起始+预警机间歇丢失 + 两编队异步跃升/折线/回正(保持9km)（压测搜索恢复/覆盖鲁棒性）"
        ))
        
        return scenarios
    
    @staticmethod
    def get_verification_scenarios() -> List[EnemyScenario]:
        """获取验证场景（V01-V03）- 与get_core_scenarios相同"""
        return ScenarioGenerator.get_core_scenarios()


if __name__ == "__main__":
    # 测试场景生成
    scenarios = ScenarioGenerator.get_core_scenarios()
    print(f"生成 {len(scenarios)} 个核心场景：\n")
    
    for s in scenarios:
        print(f"{s.scenario_id} [{'⭐' * s.difficulty}] {s.description}")
        positions = s.generate_initial_positions()
        velocities = s.generate_initial_velocities()
        print(f"    初始位置: {positions[0]}")
        print(f"    初始速度: {velocities[0]}")
        print()
