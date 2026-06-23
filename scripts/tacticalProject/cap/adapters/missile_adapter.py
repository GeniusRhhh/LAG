"""
导弹系统适配器 - 封装MissileManager
提供导弹发射、跟踪和制导功能
"""
import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

log = logging.getLogger(__name__)

# 尝试导入现有导弹系统
try:
    import sys
    import os
    tactical_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if tactical_path not in sys.path:
        sys.path.insert(0, tactical_path)
    from missile_manager import MissileManager
    from tactical_state_manager import TacticalStateManager
    MISSILE_AVAILABLE = True
except ImportError:
    MISSILE_AVAILABLE = False
    MissileManager = None
    TacticalStateManager = None

# 🔥 导入R27ER导弹模拟器（我方使用半主动制导）
try:
    from simulation.r27er_missile import R27ERMissileSimulator
    R27ER_AVAILABLE = True
except ImportError:
    R27ER_AVAILABLE = False
    R27ERMissileSimulator = None


class MissileState(Enum):
    """导弹状态"""
    READY = "ready"           # 待发射
    LAUNCHED = "launched"     # 已发射
    GUIDING = "guiding"       # 制导中
    TERMINAL = "terminal"     # 末制导
    HIT = "hit"              # 命中
    MISS = "miss"            # 脱靶


@dataclass
class MissileStatus:
    """导弹状态信息"""
    missile_id: str
    state: MissileState
    target_id: Optional[str]
    guide_agent_id: Optional[str]
    distance_to_target: float
    time_of_flight: float


class MissileAdapter:
    """导弹系统适配器
    
    我方使用R-27ER（半主动雷达制导，需要中制导）
    敌方使用AIM-120C（主动雷达制导，射后不管）
    """
    
    # 🔥 发射间隔限制（秒）
    MIN_LAUNCH_INTERVAL = 10.0
    
    def __init__(self, control_ranges=None):
        """初始化导弹适配器
        
        Args:
            control_ranges: 控制距离配置，用于判断发射时机
        """
        from ..control_ranges import DEFAULT_RANGES
        self.ranges = control_ranges or DEFAULT_RANGES
        
        # 导弹库存 {agent_id: remaining_count}
        self._inventory: Dict[str, int] = {}
        # 已发射导弹 {missile_id: MissileStatus}
        self._missiles: Dict[str, MissileStatus] = {}
        # 真实导弹对象 {missile_id: missile_obj}
        self._real_missiles: Dict[str, object] = {}
        # 发射计数
        self._launch_count: Dict[str, int] = {}
        # 制导分配 {missile_id: guide_agent_id}
        self._guidance: Dict[str, str] = {}
        # 🔥 上次发射时间 {agent_id: last_launch_time}
        self._last_launch_time: Dict[str, float] = {}
        self._launch_events: List[Dict[str, object]] = []
        self._outcome_events: List[Dict[str, object]] = []
        self._state_log_last: Dict[str, MissileState] = {}
        self._hit_targets_recorded: set[str] = set()
        self._hit_target_by_missile: Dict[str, str] = {}
        
        # 尝试使用现有系统
        self._state_manager = None
        self._missile_manager = None
        if MISSILE_AVAILABLE:
            try:
                self._state_manager = TacticalStateManager()
                self._missile_manager = MissileManager(self._state_manager)
                log.info("✅ 导弹适配器: 使用MissileManager")
            except Exception as e:
                log.warning(f"⚠️ MissileManager初始化失败: {e}")
    
    def init_inventory(self, agent_ids: List[str], missiles_per_agent: int = 4):
        """初始化导弹库存"""
        for aid in agent_ids:
            self._inventory[aid] = missiles_per_agent
            self._launch_count[aid] = 0

    def _count_alive_enemy_aircraft(self, env, agent_id: str) -> int:
        enemy_prefix = 'B' if str(agent_id).startswith('A') else 'A'
        return sum(
            1
            for other_id, other_aircraft in getattr(env, 'agents', {}).items()
            if str(other_id).startswith(enemy_prefix) and getattr(other_aircraft, 'is_alive', False)
        )

    def _extract_frozen_target_id(self, missile_obj, env=None) -> Optional[str]:
        for attr_name in (
            'launch_target_real_id',
            'launch_target_id',
            'launch_target_uid',
            'frozen_target_real_id',
            'frozen_target_id',
            'frozen_target_uid',
        ):
            try:
                candidate = getattr(missile_obj, attr_name, None)
            except Exception:
                candidate = None
            resolved = self._resolve_real_agent_id(candidate, env)
            if resolved:
                return resolved
        return None

    def _get_team_remaining_inventory(self, agent_id: str) -> int:
        team_prefix = str(agent_id)[:1]
        return sum(
            max(0, int(remaining))
            for other_id, remaining in self._inventory.items()
            if str(other_id).startswith(team_prefix)
        )

    def _get_active_missiles_on_target(self, target_id: str) -> int:
        active_states = {
            MissileState.LAUNCHED,
            MissileState.GUIDING,
            MissileState.TERMINAL,
        }
        return sum(
            1
            for status in self._missiles.values()
            if status.target_id == target_id and status.state in active_states
        )

    def _get_effective_missiles_on_target(
        self,
        target_id: str,
        max_distance_km: float = 45.0,
        max_age_s: float = 34.0,
    ) -> int:
        active_states = {
            MissileState.LAUNCHED,
            MissileState.GUIDING,
            MissileState.TERMINAL,
        }
        count = 0
        for status in self._missiles.values():
            if status.target_id != target_id or status.state not in active_states:
                continue
            if status.state == MissileState.TERMINAL:
                count += 1
                continue
            try:
                distance_to_target = float(getattr(status, 'distance_to_target', 999.0))
            except Exception:
                distance_to_target = 999.0
            try:
                time_of_flight = float(getattr(status, 'time_of_flight', 0.0))
            except Exception:
                time_of_flight = 0.0
            if distance_to_target <= max_distance_km or time_of_flight <= max_age_s:
                count += 1
        return count

    def _sync_aircraft_inventory(self, aircraft, agent_id: str):
        inventory = max(0, int(self._inventory.get(agent_id, 0)))
        try:
            aircraft.num_missiles = inventory
        except Exception:
            pass
        try:
            aircraft.num_left_missiles = inventory
        except Exception:
            pass

    def _resolve_real_agent_id(self, agent_id: Optional[str], env=None) -> Optional[str]:
        if not agent_id:
            return agent_id
        aid = str(agent_id)
        if env is not None and hasattr(env, 'resolve_real_agent_id'):
            try:
                resolved = env.resolve_real_agent_id(aid)
                if resolved:
                    return str(resolved)
            except Exception:
                pass
        return aid

    def _resolve_guidance_aircraft(self, guide_agent_id: Optional[str], env=None, aircraft=None):
        if aircraft is not None:
            return aircraft
        if not guide_agent_id or env is None:
            return None

        real_id = self._resolve_real_agent_id(guide_agent_id, env)
        candidate_ids = []
        for candidate in (real_id, str(guide_agent_id)):
            if candidate and candidate not in candidate_ids:
                candidate_ids.append(candidate)

        for candidate_env in (env, getattr(env, "_original", None)):
            if candidate_env is None:
                continue
            for collection_name in ("_jsbsims", "agents"):
                collection = getattr(candidate_env, collection_name, None)
                if not isinstance(collection, dict):
                    continue
                for candidate_id in candidate_ids:
                    if candidate_id in collection:
                        return collection.get(candidate_id)
        return None

    def _extract_missile_guide_id(self, missile_obj, env=None) -> Optional[str]:
        guide_id = None
        try:
            guide_id = getattr(missile_obj, 'guide_agent_id', None)
        except Exception:
            guide_id = None

        try:
            guide_aircraft = getattr(missile_obj, 'guide_aircraft', None)
            if guide_aircraft is not None:
                guide_id = (
                    getattr(guide_aircraft, 'real_id', None)
                    or getattr(guide_aircraft, 'uid', None)
                    or guide_id
                )
        except Exception:
            pass

        return self._resolve_real_agent_id(guide_id, env)

    def _get_target_battlefield_y_km(self, env, target_id: str) -> float:
        target = getattr(env, "agents", {}).get(target_id)
        if target is None:
            return float("nan")
        task = getattr(env, "task", None)
        if task is not None and hasattr(task, "coord_sys"):
            try:
                geo = target.get_geodetic()
                _, y_km = task.coord_sys.geodetic_to_battlefield(geo[0], geo[1])
                return float(y_km)
            except Exception:
                pass
        try:
            pos = target.get_position()
            return float(pos[0]) / 1000.0
        except Exception:
            return float("nan")

    def _classify_target_risk_zone(self, env, target_id: str) -> tuple[str, float]:
        y_km = self._get_target_battlefield_y_km(env, target_id)
        if not np.isfinite(y_km):
            return "UNKNOWN", y_km
        if y_km < 0.0 or y_km > 300.0:
            return "OUTSIDE", y_km
        if y_km < 100.0:
            return "HIGH", y_km
        if y_km < 200.0:
            return "MEDIUM", y_km
        return "LOW", y_km

    def _get_coop_prelaunch_state(
        self,
        env,
        target_id: str,
        current_time: float,
        shooter_id: str = None,
        distance_km: float = None,
        is_second_attack: bool = False,
        target_zone: str = None,
    ) -> tuple[bool, dict]:
        owner_task = getattr(env, "task", None)
        if owner_task is None or not hasattr(owner_task, "check_external_prelaunch_gate"):
            return False, {}
        try:
            passed, details = owner_task.check_external_prelaunch_gate(
                target_id,
                current_time,
                shooter_id=shooter_id,
                distance_km=distance_km,
                is_second_attack=is_second_attack,
                target_zone=target_zone,
            )
            if not isinstance(details, dict):
                details = {}
            return bool(passed), details
        except Exception:
            return False, {}

    def _record_launch_event(
        self,
        shooter_id: str,
        target_id: str,
        missile_id: str,
        missile_model: str,
        guidance_mode: str,
        current_time: float,
        distance_km: float,
    ) -> None:
        missiles_left = int(self._inventory.get(shooter_id, 0))
        self._launch_events.append(
            {
                "time": float(current_time),
                "shooter": str(shooter_id),
                "target": str(target_id),
                "missile_id": str(missile_id),
                "missile_model": str(missile_model),
                "guidance_mode": str(guidance_mode),
                "distance_km": float(distance_km),
                "missiles_left": missiles_left,
            }
        )
        log.info(
            "[导弹发射] t=%.0fs shooter=%s target=%s missile=%s model=%s guide=%s dist=%.1fkm left=%d",
            current_time,
            shooter_id,
            target_id,
            missile_id,
            missile_model,
            guidance_mode,
            distance_km,
            missiles_left,
        )

    def sync_external_launch_event(
        self,
        shooter_id: str,
        target_id: str,
        missile_id: str,
        current_time: float,
        distance_km: float = 0.0,
        missile_model: Optional[str] = None,
        guidance_mode: Optional[str] = None,
    ) -> bool:
        """Mirror a launch created outside MissileAdapter into summary accounting."""
        shooter = str(shooter_id or "").strip()
        target = str(target_id or "").strip()
        missile = str(missile_id or "").strip()
        if not shooter or not target or not missile:
            return False

        for event in self._launch_events:
            if str(event.get("missile_id", "")).strip() == missile:
                return False

        if shooter in self._inventory:
            self._inventory[shooter] = max(0, int(self._inventory.get(shooter, 0)) - 1)
        self._launch_count[shooter] = int(self._launch_count.get(shooter, 0)) + 1
        self._last_launch_time[shooter] = float(current_time)
        self._guidance.setdefault(missile, shooter)

        inferred_model = missile_model or ("R27ERMissileSimulator" if shooter.startswith("A") else "MissileSimulator")
        inferred_guidance = guidance_mode or ("semi_active" if shooter.startswith("A") else "active")
        self._record_launch_event(
            shooter,
            target,
            missile,
            inferred_model,
            inferred_guidance,
            float(current_time),
            float(distance_km),
        )
        return True

    def register_external_missile(
        self,
        missile_id: str,
        missile_obj,
        shooter_id: Optional[str] = None,
        target_id: Optional[str] = None,
        env=None,
    ) -> bool:
        """Register a real missile created outside MissileAdapter so relay can act on the real object."""
        mid = str(missile_id or "").strip()
        if not mid or missile_obj is None:
            return False

        self._real_missiles[mid] = missile_obj

        frozen_target_id = self._resolve_real_agent_id(target_id, env)
        if frozen_target_id:
            for attr_name in (
                'launch_target_real_id',
                'launch_target_id',
                'launch_target_uid',
                'frozen_target_real_id',
                'frozen_target_id',
                'frozen_target_uid',
            ):
                try:
                    setattr(missile_obj, attr_name, frozen_target_id)
                except Exception:
                    pass

        resolved_guide_id = self._extract_missile_guide_id(missile_obj, env=env) or self._resolve_real_agent_id(shooter_id, env)
        resolved_target_id = self._extract_frozen_target_id(missile_obj, env=env)
        try:
            target_aircraft = getattr(missile_obj, 'target_aircraft', None)
            if target_aircraft is not None and not resolved_target_id:
                resolved_target_id = (
                    getattr(target_aircraft, 'real_id', None)
                    or getattr(target_aircraft, 'uid', None)
                )
        except Exception:
            pass
        if not resolved_target_id:
            try:
                resolved_target_id = getattr(missile_obj, 'target_id', None)
            except Exception:
                resolved_target_id = None
        if not resolved_target_id:
            resolved_target_id = self._resolve_real_agent_id(target_id, env)
        else:
            resolved_target_id = self._resolve_real_agent_id(resolved_target_id, env)

        distance_km = 100.0
        try:
            if hasattr(missile_obj, 'target_distance'):
                distance_km = float(getattr(missile_obj, 'target_distance')) / 1000.0
        except Exception:
            distance_km = 100.0

        time_of_flight = 0.0
        try:
            time_of_flight = float(getattr(missile_obj, '_t', 0.0))
        except Exception:
            time_of_flight = 0.0

        state = MissileState.LAUNCHED
        try:
            if hasattr(missile_obj, 'is_success') and getattr(missile_obj, 'is_success', False):
                state = MissileState.HIT
            elif hasattr(missile_obj, 'is_done') and getattr(missile_obj, 'is_done', False):
                state = MissileState.MISS
            elif hasattr(missile_obj, 'is_alive') and getattr(missile_obj, 'is_alive', False):
                phase = getattr(missile_obj, '_phase', None)
                terminal_phase = getattr(missile_obj.__class__, 'TERMINAL_PHASE', None)
                if phase is not None and terminal_phase is not None and phase == terminal_phase:
                    state = MissileState.TERMINAL
                elif time_of_flight > 0.0:
                    state = MissileState.GUIDING
        except Exception:
            state = MissileState.LAUNCHED

        status = self._missiles.get(mid)
        if status is None:
            self._missiles[mid] = MissileStatus(
                missile_id=mid,
                state=state,
                target_id=resolved_target_id,
                guide_agent_id=resolved_guide_id,
                distance_to_target=distance_km,
                time_of_flight=time_of_flight,
            )
        else:
            status.state = state
            status.target_id = resolved_target_id or status.target_id
            status.guide_agent_id = resolved_guide_id or status.guide_agent_id
            status.distance_to_target = distance_km
            status.time_of_flight = time_of_flight

        if resolved_guide_id:
            self._guidance[mid] = resolved_guide_id
        return True

    def _record_state_transition(self, status: MissileStatus, prev_state: MissileState, new_state: MissileState) -> None:
        if new_state == prev_state:
            return
        self._state_log_last[status.missile_id] = new_state
        if new_state == MissileState.TERMINAL:
            log.info(
                "[导弹阶段] missile=%s state=TERMINAL target=%s tof=%.1fs dist=%.1fkm guide=%s",
                status.missile_id,
                status.target_id,
                float(status.time_of_flight),
                float(status.distance_to_target),
                status.guide_agent_id,
            )
            return
        if new_state in (MissileState.HIT, MissileState.MISS):
            if new_state == MissileState.HIT and status.target_id:
                target_key = str(status.target_id)
                self._hit_targets_recorded.add(target_key)
                self._hit_target_by_missile[target_key] = str(status.missile_id)
            self._outcome_events.append(
                {
                    "missile_id": status.missile_id,
                    "target": status.target_id,
                    "guide": status.guide_agent_id,
                    "time_of_flight": float(status.time_of_flight),
                    "distance_km": float(status.distance_to_target),
                    "state": new_state.value,
                }
            )
            log.info(
                "[导弹结束] missile=%s state=%s target=%s tof=%.1fs guide=%s",
                status.missile_id,
                new_state.value.upper(),
                status.target_id,
                float(status.time_of_flight),
                status.guide_agent_id,
            )

    def _suppress_duplicate_hit(self, status: MissileStatus, current_time: float) -> bool:
        target_key = str(status.target_id or "")
        if not target_key or target_key not in self._hit_targets_recorded:
            return False
        first_hit_missile = str(self._hit_target_by_missile.get(target_key, ""))
        if first_hit_missile == str(status.missile_id):
            return False
        if not hasattr(self, "_duplicate_hit_log_last"):
            self._duplicate_hit_log_last = {}
        last_log_t = float(self._duplicate_hit_log_last.get(status.missile_id, -9999.0))
        if float(current_time) - last_log_t >= 3.0:
            self._duplicate_hit_log_last[status.missile_id] = float(current_time)
            log.info(
                "[DUPLICATE_HIT_SUPPRESS] missile=%s target=%s tof=%.1fs dist=%.1fkm",
                status.missile_id,
                status.target_id,
                float(status.time_of_flight),
                float(status.distance_to_target),
            )
        return True

    def get_summary_snapshot(self) -> Dict[str, object]:
        by_shooter: Dict[str, Dict[str, object]] = {}
        for event in self._launch_events:
            shooter = str(event.get("shooter"))
            bucket = by_shooter.setdefault(
                shooter,
                {
                    "launches": 0,
                    "avg_distance_km": 0.0,
                    "targets": {},
                },
            )
            bucket["launches"] = int(bucket["launches"]) + 1
            bucket["avg_distance_km"] += float(event.get("distance_km", 0.0))
            targets = bucket["targets"]
            target_id = str(event.get("target"))
            targets[target_id] = int(targets.get(target_id, 0)) + 1

        for bucket in by_shooter.values():
            launches = max(1, int(bucket["launches"]))
            bucket["avg_distance_km"] = float(bucket["avg_distance_km"]) / float(launches)

        state_counts = {
            "ready": 0,
            "launched": 0,
            "guiding": 0,
            "terminal": 0,
            "hit": 0,
            "miss": 0,
        }
        for status in self._missiles.values():
            state_counts[status.state.value] = int(state_counts.get(status.state.value, 0)) + 1

        return {
            "launch_count": len(self._launch_events),
            "outcome_count": len(self._outcome_events),
            "state_counts": state_counts,
            "by_shooter": by_shooter,
        }
    
    def should_launch(self, agent_id: str, target_id: str, distance: float,
                      is_second_attack: bool = False, current_time: float = 0.0, env=None) -> bool:
        """判断是否应该发射导弹
        
        发射时机：
        - 首次进攻: 距离 <= LR(100km)
        - 二次进攻: 距离 <= LR'(42km)
        - 发射间隔: >= 10秒
        """
        # 检查库存
        if self._inventory.get(agent_id, 0) <= 0:
            return False
        
        # 🔥 检查发射间隔（防止连续发射）
        last_time = self._last_launch_time.get(agent_id, -999)
        if current_time - last_time < self.MIN_LAUNCH_INTERVAL:
            return False
        
        # 检查是否已对该目标发射（同一目标最多2枚）
        target_missiles = self._get_active_missiles_on_target(target_id)
        target_risk_zone = "UNKNOWN"
        priority_target = False
        coop_prelaunch_ready = False
        is_friendly = str(agent_id).startswith('A')
        if env is not None:
            target_risk_zone, _ = self._classify_target_risk_zone(env, target_id)
            priority_target = target_risk_zone in ("HIGH", "MEDIUM")
            coop_prelaunch_ready, _ = self._get_coop_prelaunch_state(
                env,
                target_id,
                current_time,
                shooter_id=agent_id,
                distance_km=float(distance),
                is_second_attack=bool(is_second_attack),
                target_zone=str(target_risk_zone),
            )

        saturation_limit = 2
        saturation_distance_km = 42.0
        saturation_age_s = 34.0
        if target_risk_zone == "MEDIUM":
            saturation_limit = 3
            saturation_distance_km = 48.0
            saturation_age_s = 30.0
        elif target_risk_zone == "HIGH":
            saturation_limit = 4
            saturation_distance_km = 56.0
            saturation_age_s = 26.0
        if coop_prelaunch_ready:
            saturation_age_s = max(22.0, saturation_age_s - 4.0)
        effective_target_missiles = self._get_effective_missiles_on_target(
            target_id,
            max_distance_km=saturation_distance_km,
            max_age_s=saturation_age_s,
        )
        if effective_target_missiles >= saturation_limit:
            return False

        if env is not None:
            alive_enemy_count = self._count_alive_enemy_aircraft(env, agent_id)
            team_inventory = self._get_team_remaining_inventory(agent_id)
            shooter_inventory = max(0, int(self._inventory.get(agent_id, 0)))

            if alive_enemy_count > 1 and effective_target_missiles >= saturation_limit and distance > 18.0:
                return False
            if (not priority_target) and alive_enemy_count > 1 and (shooter_inventory - 1) < 1 and distance > 35.0:
                return False
            if (not priority_target) and alive_enemy_count > 1 and team_inventory <= alive_enemy_count and distance > 45.0:
                return False
            if is_friendly and (not priority_target) and (not is_second_attack):
                if alive_enemy_count >= 3 and shooter_inventory <= 3:
                    return False
                if shooter_inventory <= 2 and distance > 44.0:
                    return False
                if team_inventory <= (alive_enemy_count + 3) and distance > 48.0:
                    return False
                if effective_target_missiles >= 1 and distance > 50.0:
                    return False
            if is_friendly:
                reserve_floor = max(5, alive_enemy_count * 2 + 1)
                if team_inventory <= reserve_floor:
                    if distance > 56.0:
                        return False
                    if (not priority_target) and (not coop_prelaunch_ready) and distance > 44.0:
                        return False
                if (
                    (not is_second_attack)
                    and alive_enemy_count >= 3
                    and shooter_inventory <= 3
                    and effective_target_missiles >= 1
                    and distance > 40.0
                ):
                    return False
                if alive_enemy_count >= 3 and shooter_inventory <= 2 and not ((priority_target or coop_prelaunch_ready) and distance <= 58.0):
                    return False
                if alive_enemy_count >= 2 and shooter_inventory <= 1 and not ((priority_target or coop_prelaunch_ready) and distance <= 48.0):
                    return False

        # 发射时机判断
        if is_second_attack:
            # 二次进攻区：LR'(42km)发射
            if is_friendly:
                low_km = 36.0
                high_km = 54.0
                if priority_target:
                    low_km = 35.0
                    high_km = 56.0
                if coop_prelaunch_ready:
                    low_km = max(34.0, low_km - 1.0)
                    high_km += 2.0 if priority_target else 1.0
            else:
                low_km = 42.0
                high_km = 56.0
                if coop_prelaunch_ready:
                    low_km = 40.0
                    high_km = 58.0 if priority_target else 56.0
                elif priority_target:
                    low_km = 41.0
                    high_km = 58.0
            return low_km <= distance <= high_km
        else:
            # 首次进攻：R-27ER 对我方做更保守的首轮窗口，避免 90km 级浪费性远射
            if is_friendly:
                low_km = 70.0
                high_km = 84.0
                if target_risk_zone == "MEDIUM":
                    low_km = 68.0
                    high_km = 85.0
                elif target_risk_zone == "HIGH":
                    low_km = 66.0
                    high_km = 86.0
                if coop_prelaunch_ready:
                    low_km = max(66.0, low_km - 1.0)
                    high_km += 1.0 if priority_target else 0.5
            else:
                low_km = float(self.ranges.TR)
                high_km = float(self.ranges.LR)
                if coop_prelaunch_ready:
                    low_km = max(68.0, low_km - (8.0 if priority_target else 4.0))
                    high_km = high_km + (8.0 if priority_target else 4.0)
                elif priority_target:
                    low_km = max(72.0, low_km - 4.0)
                    high_km = high_km + 4.0
            return low_km <= distance <= high_km
    
    def execute_launch(self, agent_id: str, target_id: str, 
                       current_time: float, env=None) -> Optional[str]:
        """执行导弹发射
        
        Args:
            agent_id: 发射机ID
            target_id: 目标ID
            current_time: 当前时间
            env: 环境对象（用于真实导弹发射）
        
        Returns:
            导弹ID，失败返回None
        """
        if self._inventory.get(agent_id, 0) <= 0:
            log.warning(f"⚠️ {agent_id}导弹库存不足")
            return None
        
        # 生成导弹ID - 依附母体命名，不使用下划线（Tacview兼容）
        # 格式: {base_id}{count} 例如 A1001, B1002
        # base_id: A0100 → A100, B0100 → B100
        self._launch_count[agent_id] = self._launch_count.get(agent_id, 0) + 1
        base_id = agent_id[0] + agent_id[2:]  # A0100 → A100
        missile_id = f"{base_id}{self._launch_count[agent_id]}"
        
        # 🔥 我方使用R-27ER（半主动制导），敌方使用AIM-120C（主动制导）
        if env is not None:
            try:
                aircraft = env.agents.get(agent_id)
                target = env.agents.get(target_id)
                if aircraft and target and aircraft.is_alive and target.is_alive:
                    # 🔥 我方飞机(A*)使用R-27ER，敌方(B*)使用AIM-120C
                    if agent_id.startswith('A') and R27ER_AVAILABLE:
                        # R-27ER 半主动雷达制导（需要中制导）
                        missile = R27ERMissileSimulator.create(
                            parent=aircraft, target=target, uid=missile_id
                        )
                        log.info(
                            f"[R-27ER发射] missile={missile_id} shooter={agent_id} "
                            f"target={target_id} left={self._inventory[agent_id]-1}"
                        )
                    else:
                        # AIM-120C 主动雷达制导（射后不管）
                        from envs.JSBSim.core.simulatior import MissileSimulator
                        missile = MissileSimulator.create(
                            parent=aircraft, target=target, uid=missile_id
                        )
                        log.info(
                            f"[AIM-120C发射] missile={missile_id} shooter={agent_id} "
                            f"target={target_id} left={self._inventory[agent_id]-1}"
                        )
                    
                    env.add_temp_simulator(missile)
                    if not hasattr(env, 'missiles') or env.missiles is None:
                        env.missiles = {}
                    env.missiles[missile_id] = missile
                    self._real_missiles[missile_id] = missile

                    # 记录统一状态，便于上层协同交战/接力逻辑读取
                    distance_km = 100.0
                    if hasattr(missile, 'target_distance'):
                        try:
                            distance_km = float(getattr(missile, 'target_distance')) / 1000.0
                        except Exception:
                            pass
                    self._missiles[missile_id] = MissileStatus(
                        missile_id=missile_id,
                        state=MissileState.LAUNCHED,
                        target_id=target_id,
                        guide_agent_id=agent_id,
                        distance_to_target=distance_km,
                        time_of_flight=0.0
                    )
                    self._guidance[missile_id] = agent_id

                    # R-27ER 半主动制导：显式标记当前制导机，便于后续接力
                    if hasattr(missile, 'set_guidance_aircraft'):
                        try:
                            missile.set_guidance_aircraft(aircraft)
                        except Exception:
                            pass

                    self._inventory[agent_id] -= 1
                    self._sync_aircraft_inventory(aircraft, agent_id)
                    self._last_launch_time[agent_id] = current_time  # 🔥 记录发射时间
                    self._record_launch_event(
                        agent_id,
                        target_id,
                        missile_id,
                        type(missile).__name__,
                        "semi_active" if agent_id.startswith('A') else "active",
                        current_time,
                        distance_km,
                    )
                    return missile_id
            except Exception as e:
                log.warning(f"[导弹发射失败] real_launch_error={e} fallback=simulated")
        
        # 回退到模拟导弹
        self._missiles[missile_id] = MissileStatus(
            missile_id=missile_id,
            state=MissileState.LAUNCHED,
            target_id=target_id,
            guide_agent_id=agent_id,
            distance_to_target=100.0,
            time_of_flight=0.0
        )
        self._guidance[missile_id] = agent_id
        self._inventory[agent_id] -= 1
        if env is not None:
            aircraft = env.agents.get(agent_id)
            if aircraft is not None:
                self._sync_aircraft_inventory(aircraft, agent_id)
        self._last_launch_time[agent_id] = current_time
        self._record_launch_event(
            agent_id,
            target_id,
            missile_id,
            "SimulatedMissile",
            "semi_active" if agent_id.startswith('A') else "active",
            current_time,
            100.0,
        )
        
        log.info(
            f"[导弹发射-模拟] missile={missile_id} shooter={agent_id} "
            f"target={target_id} left={self._inventory[agent_id]}"
        )
        return missile_id
    
    def update(self, current_time: float, target_positions: Dict[str, Tuple[float, float]], env=None):
        """更新导弹状态
        
        Args:
            current_time: 当前时间
            target_positions: 目标位置 {target_id: (x, y)}
        """
        real_pool = {}
        if env is not None and hasattr(env, 'missiles') and env.missiles:
            real_pool.update(env.missiles)
        real_pool.update(self._real_missiles)
        real_ids = set(real_pool.keys())

        for mid, status in list(self._missiles.items()):
            if mid in real_ids:
                continue
            if status.state in (MissileState.HIT, MissileState.MISS):
                continue
            prev_state = status.state
            
            # 更新飞行时间
            status.time_of_flight += 0.2  # 假设0.2s/步
            
            # 简化：假设导弹以1km/s速度飞行
            status.distance_to_target = max(0, status.distance_to_target - 0.2)
            
            # 状态转换
            if status.distance_to_target < 1.0:
                # 命中判定（简化：80%命中率）
                import random
                if random.random() < 0.8 and not self._suppress_duplicate_hit(status, current_time):
                    status.state = MissileState.HIT
                    self._record_state_transition(status, prev_state, status.state)
                    log.info(f"💥 导弹命中: {mid} → {status.target_id}")
                else:
                    status.state = MissileState.MISS
                    self._record_state_transition(status, prev_state, status.state)
                    log.info(f"❌ 导弹脱靶: {mid}")
            elif status.distance_to_target < 10.0:
                status.state = MissileState.TERMINAL
                self._record_state_transition(status, prev_state, status.state)
            elif status.state == MissileState.LAUNCHED:
                status.state = MissileState.GUIDING

        def _resolve_missile_ids(missile_obj) -> Tuple[Optional[str], Optional[str]]:
            guide_id = None
            target_id = self._extract_frozen_target_id(missile_obj, env=env)
            try:
                guide_id = getattr(missile_obj, 'guide_agent_id', None)
            except Exception:
                guide_id = None

            try:
                guide_aircraft = getattr(missile_obj, 'guide_aircraft', None)
                if guide_aircraft is not None:
                    guide_id = (
                        getattr(guide_aircraft, 'real_id', None)
                        or getattr(guide_aircraft, 'uid', None)
                        or guide_id
                    )
            except Exception:
                pass

            try:
                target_aircraft = getattr(missile_obj, 'target_aircraft', None)
                if target_aircraft is not None and not target_id:
                    target_id = (
                        getattr(target_aircraft, 'real_id', None)
                        or getattr(target_aircraft, 'uid', None)
                        or target_id
                    )
            except Exception:
                pass

            try:
                target_id = target_id or getattr(missile_obj, 'target_id', None)
            except Exception:
                pass

            return self._resolve_real_agent_id(guide_id, env), self._resolve_real_agent_id(target_id, env)

        # 同步真实导弹状态（优先使用环境中的导弹对象）
        for mid, missile in real_pool.items():
            if not mid or missile is None:
                continue

            status = self._missiles.get(mid)
            if status is None:
                guide_id = self._guidance.get(mid)
                target_id = self._extract_frozen_target_id(missile, env=env)
                try:
                    target = getattr(missile, 'target_aircraft', None)
                    if target is not None and not target_id:
                        target_id = getattr(target, 'real_id', None) or getattr(target, 'uid', None)
                except Exception:
                    pass
                resolved_guide_id, resolved_target_id = _resolve_missile_ids(missile)
                status = MissileStatus(
                    missile_id=mid,
                    state=MissileState.LAUNCHED,
                    target_id=resolved_target_id or self._resolve_real_agent_id(target_id, env),
                    guide_agent_id=resolved_guide_id or self._resolve_real_agent_id(guide_id, env),
                    distance_to_target=100.0,
                    time_of_flight=0.0,
                )
                self._missiles[mid] = status

            prev_state = status.state

            try:
                status.time_of_flight = float(getattr(missile, '_t', status.time_of_flight))
            except Exception:
                pass

            try:
                if hasattr(missile, 'target_distance'):
                    status.distance_to_target = float(getattr(missile, 'target_distance')) / 1000.0
            except Exception:
                pass

            resolved_guide_id, resolved_target_id = _resolve_missile_ids(missile)
            cached_guide_id = self._resolve_real_agent_id(self._guidance.get(mid), env)
            if resolved_guide_id and cached_guide_id and str(resolved_guide_id) != str(cached_guide_id):
                if not hasattr(self, "_relay_revert_log_time"):
                    self._relay_revert_log_time = {}
                last_log_t = float(self._relay_revert_log_time.get(mid, -9999.0))
                if float(current_time) - last_log_t >= 4.0:
                    self._relay_revert_log_time[mid] = float(current_time)
                    log.info(
                        "[RELAY_REVERT] missile=%s actual=%s cached=%s status=%s",
                        mid,
                        resolved_guide_id,
                        cached_guide_id,
                        status.guide_agent_id,
                    )
            if resolved_guide_id:
                status.guide_agent_id = resolved_guide_id
                self._guidance[mid] = resolved_guide_id
            elif status.guide_agent_id is None:
                status.guide_agent_id = self._resolve_real_agent_id(self._guidance.get(mid), env)

            if resolved_target_id:
                status.target_id = resolved_target_id

            new_state = status.state
            if hasattr(missile, 'is_success') and getattr(missile, 'is_success', False):
                new_state = MissileState.HIT
            elif hasattr(missile, 'is_done') and getattr(missile, 'is_done', False):
                new_state = MissileState.MISS
            elif hasattr(missile, 'is_alive') and getattr(missile, 'is_alive', False):
                phase = getattr(missile, '_phase', None)
                terminal_phase = getattr(missile.__class__, 'TERMINAL_PHASE', None)
                if phase is not None and terminal_phase is not None and phase == terminal_phase:
                    new_state = MissileState.TERMINAL
                elif status.state == MissileState.LAUNCHED:
                    new_state = MissileState.GUIDING

            if new_state == MissileState.HIT and self._suppress_duplicate_hit(status, current_time):
                new_state = MissileState.MISS

            status.state = new_state
            if new_state != prev_state:
                self._record_state_transition(status, prev_state, new_state)
                if new_state == MissileState.HIT:
                    log.info(f"💥 导弹命中: {mid} → {status.target_id}")
                elif new_state == MissileState.MISS:
                    log.info(f"❌ 导弹脱靶: {mid}")
    
                if new_state == MissileState.MISS:
                    miss_reason = None
                    illumination_lost_s = None
                    phase_name = None
                    try:
                        if hasattr(missile, '_get_miss_reason'):
                            miss_reason = str(missile._get_miss_reason())
                    except Exception:
                        miss_reason = None
                    try:
                        illumination_lost_s = float(getattr(missile, '_illumination_lost_s'))
                    except Exception:
                        illumination_lost_s = None
                    try:
                        phase = getattr(missile, '_phase', None)
                        phase_name = {
                            getattr(missile.__class__, 'BOOST_PHASE', object()): 'BOOST',
                            getattr(missile.__class__, 'MIDCOURSE_PHASE', object()): 'MIDCOURSE',
                            getattr(missile.__class__, 'TERMINAL_PHASE', object()): 'TERMINAL',
                        }.get(phase)
                    except Exception:
                        phase_name = None
                    if miss_reason or illumination_lost_s is not None:
                        log.info(
                            "[MISSILE_END_DETAIL] missile=%s reason=%s phase=%s dist=%.1fkm lost=%.1fs guide=%s target=%s",
                            mid,
                            miss_reason or "unknown",
                            phase_name or "UNKNOWN",
                            float(status.distance_to_target),
                            float(illumination_lost_s or 0.0),
                            status.guide_agent_id,
                            status.target_id,
                        )

    def request_relay_guidance(self, missile_id: str, new_guide_id: str, env=None, aircraft=None) -> bool:
        """请求接力制导
        
        Args:
            missile_id: 导弹ID
            new_guide_id: 新制导机ID
        
        Returns:
            是否成功
        """
        if missile_id not in self._missiles:
            return False
        
        status = self._missiles[missile_id]
        if status.state not in (MissileState.GUIDING, MissileState.LAUNCHED, MissileState.TERMINAL):
            return False
        
        old_guide = status.guide_agent_id
        status.guide_agent_id = new_guide_id
        self._guidance[missile_id] = new_guide_id

        # 真实导弹对象同步制导源（R-27ER半主动照射链路）
        missile_obj = self._real_missiles.get(missile_id)
        if missile_obj is not None:
            setattr(missile_obj, 'guide_agent_id', new_guide_id)
        return True

    def bind_relay_aircraft(self, missile_id: str, aircraft) -> bool:
        """将接力后的实际飞机对象绑定到真实导弹制导源。"""
        missile_obj = self._real_missiles.get(missile_id)
        if missile_obj is None or aircraft is None:
            return False
        if hasattr(missile_obj, 'set_guidance_aircraft'):
            try:
                missile_obj.set_guidance_aircraft(aircraft)
                return True
            except Exception:
                return False
        return False
    
    def apply_relay_guidance(self, missile_id: str, new_guide_id: str, env=None, aircraft=None) -> bool:
        """Atomically switch relay guidance on both adapter state and the real missile object."""
        if missile_id not in self._missiles:
            return False

        status = self._missiles[missile_id]
        if status.state not in (MissileState.GUIDING, MissileState.LAUNCHED, MissileState.TERMINAL):
            return False

        expected_guide_id = self._resolve_real_agent_id(new_guide_id, env)
        old_guide = status.guide_agent_id
        old_guidance = self._guidance.get(missile_id)
        missile_obj = self._real_missiles.get(missile_id)
        old_obj_guide_id = getattr(missile_obj, 'guide_agent_id', None) if missile_obj is not None else None
        old_obj_guide_aircraft = getattr(missile_obj, 'guide_aircraft', None) if missile_obj is not None else None

        status.guide_agent_id = expected_guide_id
        self._guidance[missile_id] = expected_guide_id
        if missile_obj is None:
            return True

        resolved_aircraft = self._resolve_guidance_aircraft(new_guide_id, env=env, aircraft=aircraft)
        if resolved_aircraft is None:
            status.guide_agent_id = old_guide
            if old_guidance is None:
                self._guidance.pop(missile_id, None)
            else:
                self._guidance[missile_id] = old_guidance
            return False

        try:
            setattr(missile_obj, 'guide_agent_id', expected_guide_id)
            if hasattr(missile_obj, 'set_guidance_aircraft'):
                missile_obj.set_guidance_aircraft(resolved_aircraft)
            else:
                setattr(missile_obj, 'guide_aircraft', resolved_aircraft)
                setattr(missile_obj, 'guide_agent_id', getattr(resolved_aircraft, 'uid', expected_guide_id))
        except Exception:
            status.guide_agent_id = old_guide
            if old_guidance is None:
                self._guidance.pop(missile_id, None)
            else:
                self._guidance[missile_id] = old_guidance
            try:
                setattr(missile_obj, 'guide_agent_id', old_obj_guide_id)
                setattr(missile_obj, 'guide_aircraft', old_obj_guide_aircraft)
            except Exception:
                pass
            return False

        actual_guide_id = self._extract_missile_guide_id(missile_obj, env=env)
        if actual_guide_id:
            status.guide_agent_id = actual_guide_id
            self._guidance[missile_id] = actual_guide_id

        log.info(
            "[RELAY_APPLY] missile=%s expected=%s actual=%s obj=%s",
            missile_id,
            expected_guide_id,
            status.guide_agent_id,
            type(missile_obj).__name__,
        )

        return str(status.guide_agent_id or "") == str(expected_guide_id or "")

    def get_missile_status(self, agent_id: str) -> List[MissileStatus]:
        """获取指定飞机相关的导弹状态"""
        result = []
        # base_id: A0100 → A100
        base_id = agent_id[0] + agent_id[2:]
        for mid, status in self._missiles.items():
            if status.guide_agent_id == agent_id or mid.startswith(base_id):
                result.append(status)
        return result
    
    def get_inventory(self, agent_id: str) -> int:
        """获取剩余导弹数"""
        return self._inventory.get(agent_id, 0)

    def get_guided_missiles(
        self,
        guide_agent_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> List[MissileStatus]:
        result: List[MissileStatus] = []
        for status in self._missiles.values():
            if status.state not in (MissileState.LAUNCHED, MissileState.GUIDING, MissileState.TERMINAL):
                continue
            if guide_agent_id is not None and str(status.guide_agent_id or "") != str(guide_agent_id):
                continue
            if target_id is not None and str(status.target_id or "") != str(target_id):
                continue
            result.append(status)
        return result

    def has_active_guidance_commit(
        self,
        guide_agent_id: str,
        target_id: Optional[str] = None,
        terminal_only: bool = False,
    ) -> bool:
        for status in self.get_guided_missiles(guide_agent_id=guide_agent_id, target_id=target_id):
            if terminal_only and status.state != MissileState.TERMINAL:
                continue
            return True
        return False

    def get_active_missiles(self) -> List[MissileStatus]:
        """获取所有活跃导弹"""
        return [s for s in self._missiles.values() 
                if s.state in (MissileState.LAUNCHED, MissileState.GUIDING, MissileState.TERMINAL)]
    
    def needs_relay(self, missile_id: str, guide_agent_alive: bool,
                    guide_distance_to_target: float) -> bool:
        """判断是否需要接力制导
        
        Args:
            missile_id: 导弹ID
            guide_agent_alive: 当前制导机是否存活
            guide_distance_to_target: 制导机到目标距离
        
        Returns:
            是否需要接力
        """
        if missile_id not in self._missiles:
            return False
        
        status = self._missiles[missile_id]
        
        # 制导机被击落
        if not guide_agent_alive:
            return True

        mar_km = float(getattr(self.ranges, 'MAR', 35.0))
        relay_entry_km = max(mar_km + 10.0, 45.0)
        terminal_entry_km = max(mar_km + 14.0, 50.0)
        terminal_hold_km = max(12.0, min(15.0, mar_km - 20.0))
        terminal_guard_km = max(18.0, terminal_hold_km + 4.0)
        terminal_relay_range_km = max(mar_km + 4.0, 39.0)

        # 末段最后十几公里优先保持当前照射，避免快命中时仍频繁换照射机。
        if status.distance_to_target <= terminal_hold_km:
            return False

        # 末制导不再一进 TR 就默认请求接力；只有当前制导机几何明显变差时才换。
        if status.state == MissileState.TERMINAL:
            if status.distance_to_target <= terminal_guard_km:
                return guide_distance_to_target <= terminal_relay_range_km
            return guide_distance_to_target <= terminal_entry_km

        if status.distance_to_target <= 28.0:
            return guide_distance_to_target <= max(relay_entry_km - 4.0, 40.0)

        if status.distance_to_target <= 42.0 and guide_distance_to_target <= (relay_entry_km + 2.0):
            return True

        if status.time_of_flight >= 34.0 and guide_distance_to_target <= max(relay_entry_km + 8.0, 54.0):
            return True
        
        return False
