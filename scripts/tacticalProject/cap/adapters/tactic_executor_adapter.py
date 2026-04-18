"""
战术执行适配器 - 封装TacticalExecutor
支持7种战术模板执行
"""
import logging
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)

# 🔥 禁用TacticalExecutor导入（需要tactical_task参数，与CAP架构不兼容）
# 使用本地实现的战术执行逻辑
EXECUTOR_AVAILABLE = False
TacticalExecutor = None


class TacticType(Enum):
    """战术类型"""
    T_DS = "drag_shoot"       # 拖射战术
    T_PA = "pincer_attack"    # 钳形攻击
    T_HL = "high_low"         # 高低位攻击
    T_SBS = "side_by_side"    # 并列射击
    T_FB = "front_back"       # 前后夹击
    T_TE = "tactical_evade"   # 战术规避
    T_TT = "tactical_turn"    # 战术转弯


@dataclass
class TacticResult:
    """战术执行结果"""
    alt_cmd: int
    hdg_cmd: int
    vel_cmd: int
    should_launch: bool = False
    message: str = ""


class TacticExecutorAdapter:
    """战术执行适配器"""
    
    def __init__(self, control_ranges=None):
        from ..control_ranges import DEFAULT_RANGES
        self.ranges = control_ranges or DEFAULT_RANGES
        
        # 动作空间
        self.norm_hdg = np.array([-np.pi,-2*np.pi/3,-np.pi/2,-5*np.pi/12,-np.pi/3,
                                  -np.pi/4,-np.pi/6,-np.pi/12,0,np.pi/12,np.pi/6,
                                  np.pi/4,np.pi/3,5*np.pi/12,np.pi/2,2*np.pi/3,np.pi])
        
        # 战术状态
        self._tactic_state: Dict[str, Dict] = {}
        
        # 尝试使用现有系统
        self._executor = None
        if EXECUTOR_AVAILABLE:
            try:
                self._executor = TacticalExecutor()
                log.info("✅ 战术执行适配器: 使用TacticalExecutor")
            except Exception as e:
                log.warning(f"⚠️ TacticalExecutor初始化失败: {e}")
    
    def get_available_tactics(self, distance: float) -> List[TacticType]:
        """根据距离获取可用战术
        
        Args:
            distance: 到目标距离(km)
        
        Returns:
            可用战术列表
        """
        # 低风险区(200-300km): T_DS, T_PA, T_HL, T_SBS
        # 中风险区(100-200km): T_FB, T_PA
        # 高风险区(0-100km): T_TE, T_TT
        
        if distance > 200:
            return [TacticType.T_DS, TacticType.T_PA, TacticType.T_HL, TacticType.T_SBS]
        elif distance > 100:
            return [TacticType.T_FB, TacticType.T_PA, TacticType.T_TE, TacticType.T_TT]
        else:
            return [TacticType.T_TE, TacticType.T_TT]
    
    def execute_tactic(self, tactic: TacticType, agent_id: str,
                       my_pos: Tuple[float, float], my_hdg: float,
                       target_pos: Tuple[float, float],
                       is_lead: bool = True,
                       wingman_pos: Optional[Tuple[float, float]] = None) -> TacticResult:
        """执行指定战术
        
        Args:
            tactic: 战术类型
            agent_id: 执行飞机ID
            my_pos: 我方位置(x, y) km
            my_hdg: 我方航向(度)
            target_pos: 目标位置(x, y) km
            is_lead: 是否为长机
            wingman_pos: 僚机位置(用于编队战术)
        
        Returns:
            TacticResult
        """
        if tactic == TacticType.T_DS:
            return self._execute_drag_shoot(agent_id, my_pos, my_hdg, target_pos)
        elif tactic == TacticType.T_PA:
            return self._execute_pincer_attack(agent_id, my_pos, my_hdg, target_pos, is_lead)
        elif tactic == TacticType.T_HL:
            return self._execute_high_low(agent_id, my_pos, my_hdg, target_pos, is_lead)
        elif tactic == TacticType.T_SBS:
            return self._execute_side_by_side(agent_id, my_pos, my_hdg, target_pos, is_lead)
        elif tactic == TacticType.T_FB:
            return self._execute_front_back(agent_id, my_pos, my_hdg, target_pos, is_lead, wingman_pos)
        elif tactic == TacticType.T_TE:
            return self._execute_tactical_evade(agent_id, my_pos, my_hdg, target_pos)
        elif tactic == TacticType.T_TT:
            return self._execute_tactical_turn(agent_id, my_pos, my_hdg, target_pos)
        
        return TacticResult(7, 8, 3, message="未知战术")
    
    def _execute_drag_shoot(self, agent_id: str, my_pos: Tuple, my_hdg: float,
                            target_pos: Tuple) -> TacticResult:
        """拖射战术：向目标前进，发射后偏移50°脱离"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        target_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        
        # 初始化状态
        if agent_id not in self._tactic_state:
            self._tactic_state[agent_id] = {'phase': 'approach', 'launched': False}
        
        state = self._tactic_state[agent_id]
        
        if state['phase'] == 'approach':
            # 接近阶段：直飞向目标
            if distance <= self.ranges.LR + 5:
                state['phase'] = 'launch'
                state['launched'] = True
                return TacticResult(*self._hdg_to_cmd(my_hdg, target_hdg), 
                                   should_launch=True, message="拖射-发射")
            return TacticResult(*self._hdg_to_cmd(my_hdg, target_hdg), message="拖射-接近")
        
        elif state['phase'] == 'launch':
            # 发射后偏移50°脱离
            drag_hdg = (target_hdg + 50) % 360  # 右偏50°
            state['phase'] = 'drag'
            return TacticResult(*self._hdg_to_cmd(my_hdg, drag_hdg), message="拖射-脱离")
        
        else:  # drag
            # 继续脱离
            drag_hdg = (target_hdg + 50) % 360
            return TacticResult(*self._hdg_to_cmd(my_hdg, drag_hdg), message="拖射-脱离中")
    
    def _execute_pincer_attack(self, agent_id: str, my_pos: Tuple, my_hdg: float,
                               target_pos: Tuple, is_lead: bool) -> TacticResult:
        """钳形攻击：左右编队分开包抄"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        base_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        
        # 长机向左偏30°，僚机向右偏30°
        offset = -30 if is_lead else 30
        target_hdg = (base_hdg + offset) % 360
        
        return TacticResult(*self._hdg_to_cmd(my_hdg, target_hdg), message="钳形攻击")
    
    def _execute_high_low(self, agent_id: str, my_pos: Tuple, my_hdg: float,
                          target_pos: Tuple, is_lead: bool) -> TacticResult:
        """高低位攻击：长机高位，僚机低位"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        target_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        
        # 高度命令：长机爬升(高位)，僚机下降(低位)
        alt_cmd = 10 if is_lead else 4  # 爬升/下降
        
        return TacticResult(alt_cmd, self._hdg_to_cmd(my_hdg, target_hdg)[1], 3, 
                           message="高低位攻击")
    
    def _execute_side_by_side(self, agent_id: str, my_pos: Tuple, my_hdg: float,
                              target_pos: Tuple, is_lead: bool) -> TacticResult:
        """并列射击：保持横向间距同时攻击"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        target_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        
        # 轻微横向偏移
        offset = -10 if is_lead else 10
        target_hdg = (target_hdg + offset) % 360
        
        return TacticResult(*self._hdg_to_cmd(my_hdg, target_hdg), message="并列射击")
    
    def _execute_front_back(self, agent_id: str, my_pos: Tuple, my_hdg: float,
                            target_pos: Tuple, is_lead: bool,
                            wingman_pos: Optional[Tuple]) -> TacticResult:
        """前后夹击：长机在前，僚机在后"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        target_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        
        if is_lead:
            # 长机：直接向目标
            return TacticResult(*self._hdg_to_cmd(my_hdg, target_hdg), message="前后夹击-前")
        else:
            # 僚机：保持在长机后方
            if wingman_pos:
                # 跟随长机但保持后方
                return TacticResult(*self._hdg_to_cmd(my_hdg, target_hdg), message="前后夹击-后")
            return TacticResult(*self._hdg_to_cmd(my_hdg, target_hdg), message="前后夹击")
    
    def _execute_tactical_evade(self, agent_id: str, my_pos: Tuple, my_hdg: float,
                                target_pos: Tuple) -> TacticResult:
        """战术规避：三九Beam机动"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        threat_bearing = np.degrees(np.arctan2(dx, dy)) % 360
        
        # Beam机动：垂直于威胁方向
        beam_left = (threat_bearing - 90) % 360
        beam_right = (threat_bearing + 90) % 360
        
        # 选择转向角度较小的
        diff_left = abs(beam_left - my_hdg)
        if diff_left > 180: diff_left = 360 - diff_left
        diff_right = abs(beam_right - my_hdg)
        if diff_right > 180: diff_right = 360 - diff_right
        
        target_hdg = beam_left if diff_left < diff_right else beam_right
        
        return TacticResult(*self._hdg_to_cmd(my_hdg, target_hdg), message="战术规避-Beam")
    
    def _execute_tactical_turn(self, agent_id: str, my_pos: Tuple, my_hdg: float,
                               target_pos: Tuple) -> TacticResult:
        """战术转弯：快速掉头脱离"""
        target_hdg = (my_hdg + 180) % 360
        
        # 使用最大转向率
        diff = target_hdg - my_hdg
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        
        hdg_cmd = 0 if diff < 0 else 16  # -π 或 +π
        
        # 硬转弯阶段轻微下降，降低模型因转弯耦合导致的抬头爬升概率。
        return TacticResult(6, hdg_cmd, 3, message="战术转弯")
    
    def _hdg_to_cmd(self, current: float, target: float) -> Tuple[int, int, int]:
        """航向转命令"""
        diff = target - current
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        
        if abs(diff) < 5:
            return 7, 8, 3
        
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        return 7, hdg_cmd, 3
    
    def reset_state(self, agent_id: str):
        """重置战术状态"""
        if agent_id in self._tactic_state:
            del self._tactic_state[agent_id]
