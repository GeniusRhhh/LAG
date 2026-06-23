"""
鎴樻湳鎵ц妯″潡
鍖呭惈鎵€鏈夋垬鏈墽琛屽嚱鏁帮細鎷栨洺灏勫嚮銆侀挸褰㈡敾鍔裤€佷笂涓嬪す鍑汇€佸墠鍚庢敾鍑汇€佸苟鎺掑皠鍑荤瓑
浠巘actical_task.py涓彁鍙栵紝鎻愰珮浠ｇ爜鍙淮鎶ゆ€?
"""
import logging
import numpy as np
from tactical_types import TacticalPhase
from envs.JSBSim.core.catalog import Catalog as c
from core.target_assignment import get_enemy_team, get_target_with_fallback
from formation_reset_manager import FormationResetManager
from nodes.mtr_prime_node import MTRPrimeNode
from tactical_utils import TacticalUtils
from tactical_executor_support_mixin import TacticalExecutorSupportMixin
import tactical_executor_refactor_helpers as _terfh
import tactical_executor_unified_second_attack as _tues
class TacticalExecutor(TacticalExecutorSupportMixin):
    def __init__(self, tactical_task):
        self.task = tactical_task
        self.evasion_states = {}
        self.turn_states = {}
        # 鏈哄姩鍐崇瓥灞傚彲閫氳繃璇ュ紑鍏抽檷浣?澧炲己瑙勯伩婵€杩涘害锛堥伩鍏嶇煭鏃堕棿澶ц浆寮鑷磋浆鍦堬級
        self.evasion_aggressive_mode = {}
        # 瑙勯伩鏈熼棿涔熻璺熻釜鈥滄渶灏忚窛绂烩€濓紝鍚﹀垯浼氬嚭鐜帮細瑙勯伩涓?40km浣嗚閬垮悗璺濈鎷夊ぇ鈫掓案涓嶈繑鑸?
        self.evasion_min_distances = {}
        # 瑙勯伩缁撴潫鍚庤Е鍙戠殑寮哄埗杩旇埅鏍囧織锛堢敤浜庝簩娆¤繘鏀绘湡闂达級
        self.force_rtb_after_evasion = {}
        # === SIDE_BY_SIDE 涓撶敤锛氶槦浼嶇骇杩旇埅涓庤閬挎湡闂存渶灏忚窛绂昏窡韪紙涓嶅奖鍝嶅叾瀹冩垬鏈級 ===
        self.sbs_second_attack_min_distance = {}
        self.sbs_evasion_min_distance = {}
        self.sbs_evasion_active = {}
        self.sbs_team_rtb = False
        self.sbs_team_rtb_reason = None

        # 初始化队形重置管理器
        self.formation_reset_manager = getattr(self.task, 'formation_reset_manager', None)

        # 初始化 MTR' 节点
        self.mtr_prime_node = MTRPrimeNode()

        # 初始化二次进攻协同状态
        self.second_attack_coordination = {}
        self.second_attack_tactic_selected = False
        self._heading_smoother = {}
        self.high_low_heading_locks = {}
        self.tactical_turn_states = {}
        self.pre_nlt_shape_states = {}

        # 日志频率控制，减少高频非关键输出
        self.last_log_times = {}  # 记录上次打印时间
        self.log_intervals = {
            'tactical_turn_protection': 10.0,  # TACTICAL_TURN 高度保护日志间隔10秒
            'high_low_attack_info': 15.0,      # HIGH_LOW_ATTACK 战术信息间隔15秒
            'front_back_info': 15.0,           # FRONT_BACK 战术信息间隔15秒
            'debug_heading': 20.0,             # 航向调试信息间隔20秒
            'missile_status': 5.0,             # 导弹状态信息间隔5秒
            'pre_nlt_shape': 12.0,
        }
    def _get_min_distance_to_alive_enemies(self, env, agent_id: str) -> tuple:
        return super()._get_min_distance_to_alive_enemies(env, agent_id)

    def _get_agent_phase(self, agent_id: str):
        return super()._get_agent_phase(agent_id)

    def should_log(self, log_key: str, current_time: float) -> bool:
        return super().should_log(log_key, current_time)

    def _is_second_attack(self, agent_id: str) -> bool:
        return super()._is_second_attack(agent_id)

    def _get_formation_agents(self, agent_id: str):
        return super()._get_formation_agents(agent_id)

    def _get_teammate_id(self, agent_id: str):
        return super()._get_teammate_id(agent_id)

    def _get_formation_role_by_position(self, env, agent_id: str) -> str:
        return super()._get_formation_role_by_position(env, agent_id)

    def _is_formation_lead(self, env, agent_id: str) -> bool:
        return super()._is_formation_lead(env, agent_id)

    def _promote_drag_shoot_post_skate(self, env, agent_id: str, current_time: float, is_lead: bool) -> None:
        return super()._promote_drag_shoot_post_skate(env, agent_id, current_time, is_lead)

    def _get_formation_label(self, agent_id: str) -> str:
        return super()._get_formation_label(agent_id)

    def _get_base_heading(self, env, agent_id: str) -> float:
        return super()._get_base_heading(env, agent_id)

    def _get_horizontal_course_heading(self, aircraft) -> float:
        return super()._get_horizontal_course_heading(aircraft)

    def _smooth_heading(self, key: str, target_heading: float, alpha: float = 0.3) -> float:
        return super()._smooth_heading(key, target_heading, alpha=alpha)

    def _get_formation_axis_heading(self, env, agent_id: str, alpha: float = 0.25) -> float:
        return super()._get_formation_axis_heading(env, agent_id, alpha=alpha)

    def on_formation_tactic_changed(self, formation_label: str, old_tactic: str, new_tactic: str):
        return super().on_formation_tactic_changed(formation_label, old_tactic, new_tactic)

    def _get_locked_high_low_heading(self, env, agent_id: str) -> float:
        return super()._get_locked_high_low_heading(env, agent_id)

    def _exit_second_attack_to_rtb(self, env, agent_id: str, reason: str) -> tuple:
        return super()._exit_second_attack_to_rtb(env, agent_id, reason)

    def _build_altitude_heading_action(self, env, agent_id: str, alt_cmd_value: float,
                                       target_heading: float, climb_vel_cmd: int = 4,
                                       descend_vel_cmd: int = 3) -> tuple:
        return super()._build_altitude_heading_action(
            env, agent_id, alt_cmd_value, target_heading, climb_vel_cmd=climb_vel_cmd, descend_vel_cmd=descend_vel_cmd
        )

    def _get_vertical_speed_mps(self, env, agent_id: str) -> float:
        return super()._get_vertical_speed_mps(env, agent_id)

    def _project_offsets_along_heading(self, reference_heading: float, origin_pos: tuple, target_pos: tuple) -> tuple:
        return super()._project_offsets_along_heading(reference_heading, origin_pos, target_pos)

    def execute_pre_nlt_formation_shaping(self, env, agent_id: str, selected_tactic: str = None) -> tuple:
        formation_agents = [
            aid for aid in self._get_formation_agents(agent_id)
            if aid in getattr(env, 'agents', {}) and getattr(env.agents[aid], 'is_alive', False)
        ]
        formation_label = self._get_formation_label(agent_id)
        leader_id = next((aid for aid in formation_agents if self._is_formation_lead(env, aid)), formation_agents[0] if formation_agents else agent_id)
        base_heading = self._smooth_heading(
            f'pre_nlt_base:{formation_label}',
            self._get_base_heading(env, leader_id),
            alpha=0.18
        )

        if len(formation_agents) < 2:
            return self.task._maintain_heading_precise(env, agent_id, base_heading)

        wingman_id = next((aid for aid in formation_agents if aid != leader_id), None)
        if wingman_id is None or leader_id not in env.agents or wingman_id not in env.agents:
            return self.task._maintain_heading_precise(env, agent_id, base_heading)

        leader = env.agents[leader_id]
        wingman = env.agents[wingman_id]
        leader_pos = leader.get_position()
        wingman_pos = wingman.get_position()
        longitudinal, lateral = self._project_offsets_along_heading(base_heading, leader_pos, wingman_pos)

        shape_state = self.pre_nlt_shape_states.get(formation_label)
        if shape_state is None or shape_state.get('wingman_id') != wingman_id:
            side_sign = 1.0 if lateral >= 0.0 else -1.0
            if abs(lateral) < 400.0:
                world_lateral = float(wingman_pos[1]) - float(leader_pos[1])
                side_sign = 1.0 if world_lateral >= 0.0 else -1.0
            shape_state = {
                'wingman_id': wingman_id,
                'side_sign': side_sign,
            }
            self.pre_nlt_shape_states[formation_label] = shape_state

        side_sign = float(shape_state.get('side_sign', 1.0))
        target_spacing = 4.0 * 1852.0
        min_spacing = 3.0 * 1852.0
        max_spacing = 5.0 * 1852.0

        if agent_id == leader_id:
            if env.current_step % 60 == 0:
                spacing_nm = abs(lateral) / 1852.0
                logging.info(
                    f"[PRE_NLT-LEADER] {leader_id} heading={base_heading:.1f}deg "
                    f"formation_spacing={spacing_nm:.1f}nm"
                )
            return self.task._maintain_heading_precise(env, agent_id, base_heading)

        heading_rad = np.deg2rad(base_heading)
        right = np.array([-np.sin(heading_rad), np.cos(heading_rad)])
        slot_xy = np.array([float(leader_pos[0]), float(leader_pos[1])]) + right * side_sign * target_spacing
        slot_heading_raw = self.task._heading_to_point(env, wingman_id, float(slot_xy[0]), float(slot_xy[1]))
        slot_heading_delta = self.task._normalize_angle_diff(slot_heading_raw - base_heading)
        heading_limit = 35.0 if (abs(longitudinal) > 2500.0 or abs(abs(lateral) - target_spacing) > 2000.0) else 20.0
        target_heading = (base_heading + float(np.clip(slot_heading_delta, -heading_limit, heading_limit))) % 360.0
        target_heading = self._smooth_heading(f'pre_nlt_slot:{formation_label}', target_heading, alpha=0.28)

        current_heading = float(env.agents[wingman_id].get_property_value(c.attitude_psi_deg))
        heading_error = abs(self.task._normalize_angle_diff(target_heading - current_heading))
        within_band = (
            min_spacing <= abs(lateral) <= max_spacing
            and abs(longitudinal) <= 1200.0
            and heading_error <= 8.0
        )
        if within_band:
            if self.should_log('pre_nlt_shape', env.current_step * env.time_interval):
                spacing_nm = abs(lateral) / 1852.0
                logging.info(
                    f"[PRE_NLT-WINGMAN] {wingman_id} ready spacing={spacing_nm:.1f}nm "
                    f"longitudinal={longitudinal/1000.0:.1f}km"
                )
            return self.task._maintain_heading_precise(env, wingman_id, base_heading)

        vel_cmd = self.task._get_dynamic_velocity_cmd(env, wingman_id) if hasattr(self.task, '_get_dynamic_velocity_cmd') else 3
        slot_error = float(np.hypot(longitudinal, lateral - side_sign * target_spacing))
        if longitudinal < -2500.0 or slot_error > 7000.0:
            vel_cmd = max(vel_cmd, 5)
        elif longitudinal > 2500.0 and abs(lateral) <= max_spacing + 1500.0:
            vel_cmd = min(vel_cmd, 3)

        alt_cmd = 7
        alt_diff = float(leader_pos[2]) - float(wingman_pos[2])
        if abs(alt_diff) > 400.0:
            alt_cmd = self.task._convert_altitude_to_index(float(np.clip(alt_diff, -300.0, 300.0)))

        hdg_cmd = self.task._get_heading_cmd(env, wingman_id, target_heading) if hasattr(self.task, '_get_heading_cmd') else 8
        if self.should_log('pre_nlt_shape', env.current_step * env.time_interval):
            spacing_nm = abs(lateral) / 1852.0
            logging.info(
                f"[PRE_NLT-WINGMAN] {wingman_id} converging spacing={spacing_nm:.1f}nm "
                f"longitudinal={longitudinal/1000.0:.1f}km target_heading={target_heading:.1f}deg"
            )
        return alt_cmd, hdg_cmd, vel_cmd

    def _build_front_back_follow_action(self, env, wingman_id: str, leader_id: str, current_time: float) -> tuple:
        leader_pool = getattr(env, '_jsbsims', {})
        leader = leader_pool.get(leader_id)
        if not leader or not leader.is_alive:
            return self.task._maintain_heading_precise(env, wingman_id, self._get_base_heading(env, wingman_id))

        leader_x, leader_y, _ = leader.get_position()
        trail_heading = self._smooth_heading(
            f'front_back_trail:{self._get_formation_label(wingman_id)}',
            self._get_horizontal_course_heading(leader),
            alpha=0.25
        )
        trail_heading_rad = np.deg2rad(trail_heading)
        back_dist = 10000.0
        target_x = leader_x - back_dist * np.sin(trail_heading_rad)
        target_y = leader_y - back_dist * np.cos(trail_heading_rad)

        wingman_x, wingman_y, _ = env.agents[wingman_id].get_position()
        slot_error = float(np.hypot(wingman_x - target_x, wingman_y - target_y))
        current_heading = float(env.agents[wingman_id].get_property_value(c.attitude_psi_deg))
        heading_error = abs(self.task._normalize_angle_diff(trail_heading - current_heading))

        if slot_error <= 2000.0 and heading_error <= 8.0:
            if env.current_step % 60 == 0:
                logging.info(f"[FRONT_BACK-僚机保持] {wingman_id} 保持一字型, 参考航向: {trail_heading:.1f}°")
            return self.task._maintain_heading_precise(env, wingman_id, trail_heading)

        target_heading = self.task._heading_to_point(env, wingman_id, target_x, target_y)
        if env.current_step % 60 == 0:
            logging.info(f"[FRONT_BACK-僚机跟随] {wingman_id} 飞向长机正后方, 目标航向: {target_heading:.1f}°")
        return self.task._maintain_heading_precise(env, wingman_id, target_heading)

    def _get_rtb_heading(self, env, agent_id: str) -> float:
        return super()._get_rtb_heading(env, agent_id)

    def _build_rtb_command_with_alt_guard(self, env, agent_id: str, target_heading: float = None) -> tuple:
        return super()._build_rtb_command_with_alt_guard(env, agent_id, target_heading=target_heading)

    
    def execute_drag_shoot(self, env, agent_id: str):
        return _terfh.execute_drag_shoot(self, env, agent_id)
    def execute_pincer_attack(self, env, agent_id: str):
        return _terfh.execute_pincer_attack(self, env, agent_id)
    def execute_high_low_attack(self, env, agent_id: str):
        return _terfh.execute_high_low_attack(self, env, agent_id)
    def execute_front_back(self, env, agent_id: str):
        return _terfh.execute_front_back(self, env, agent_id)
    def execute_side_by_side(self, env, agent_id: str):
        return _terfh.execute_side_by_side(self, env, agent_id)
    def execute_unified_second_attack(self, env, agent_id: str) -> tuple:
        return _tues.execute_unified_second_attack(self, env, agent_id)

    def _get_beam_state_store(self):
        state_manager = getattr(self.task, 'state_manager', None)
        beam_state = getattr(state_manager, 'beam_maneuver_state', None)
        if isinstance(beam_state, dict):
            return beam_state
        if not hasattr(self.task, 'beam_maneuver_state') or not isinstance(getattr(self.task, 'beam_maneuver_state', None), dict):
            self.task.beam_maneuver_state = {}
        return self.task.beam_maneuver_state

    def _clear_beam_state(self, agent_id: str) -> None:
        beam_state = self._get_beam_state_store()
        beam_state.pop(agent_id, None)

    def _clear_missile_evasion_session(self, agent_id: str) -> None:
        self.evasion_states.pop(agent_id, None)
        self.evasion_min_distances.pop(agent_id, None)
        self._clear_beam_state(agent_id)
        escape_sessions = getattr(self, '_bvr_escape_session', None)
        if isinstance(escape_sessions, dict):
            escape_sessions.pop(agent_id, None)
        if hasattr(self.task, 'tactical_evasion_direction_state'):
            self.task.tactical_evasion_direction_state.pop(agent_id, None)
        if hasattr(self.task, 'beam_direction_state'):
            self.task.beam_direction_state.pop(agent_id, None)
        session_dirs = getattr(self, '_missile_evasion_direction_session', None)
        if isinstance(session_dirs, dict):
            session_dirs.pop(agent_id, None)
        missile_track = getattr(self, '_missile_track', None)
        if isinstance(missile_track, dict):
            stale_track_keys = [key for key in missile_track.keys() if isinstance(key, tuple) and str(key[0]) == str(agent_id)]
            for key in stale_track_keys:
                missile_track.pop(key, None)

    def _collect_agent_identity_aliases(self, env, agent_id: str) -> set[str]:
        aliases = set()

        def _append(candidate):
            cid = str(candidate or "").strip()
            if cid:
                aliases.add(cid)

        _append(agent_id)
        reference_env = getattr(self.task, '_get_reference_env', lambda _env=None: env)(env)
        for env_obj in (
            env,
            getattr(env, '_original', None) if env is not None else None,
            reference_env,
            getattr(reference_env, '_original', None) if reference_env is not None else None,
            getattr(self.task, 'env', None),
        ):
            if env_obj is None:
                continue
            if hasattr(env_obj, 'resolve_real_agent_id'):
                try:
                    _append(env_obj.resolve_real_agent_id(agent_id))
                except Exception:
                    pass
            if hasattr(env_obj, 'resolve_virtual_agent_id'):
                try:
                    _append(env_obj.resolve_virtual_agent_id(agent_id))
                except Exception:
                    pass
        return aliases

    def _collect_missile_target_aliases(self, env_obj, missile_sim):
        frozen_ids = set()
        runtime_ids = set()

        def _append(target_set, candidate):
            cid = str(candidate or "").strip()
            if cid:
                target_set.add(cid)

        def _resolve_aliases(target_set):
            resolved_ids = list(target_set)
            for target_id in resolved_ids:
                if hasattr(env_obj, 'resolve_real_agent_id'):
                    try:
                        _append(target_set, env_obj.resolve_real_agent_id(target_id))
                    except Exception:
                        pass
                if hasattr(env_obj, 'resolve_virtual_agent_id'):
                    try:
                        _append(target_set, env_obj.resolve_virtual_agent_id(target_id))
                    except Exception:
                        pass

        for attr_name in (
            'launch_target_real_id',
            'launch_target_id',
            'launch_target_uid',
            'frozen_target_real_id',
            'frozen_target_id',
            'frozen_target_uid',
        ):
            try:
                _append(frozen_ids, getattr(missile_sim, attr_name, None))
            except Exception:
                pass
        _resolve_aliases(frozen_ids)

        for attr_name in (
            'target_id',
            'target_uid',
        ):
            try:
                _append(runtime_ids, getattr(missile_sim, attr_name, None))
            except Exception:
                pass

        try:
            target_aircraft = getattr(missile_sim, 'target_aircraft', None)
        except Exception:
            target_aircraft = None
        if target_aircraft is not None:
            for attr_name in ('uid', 'real_id', 'target_id'):
                try:
                    _append(runtime_ids, getattr(target_aircraft, attr_name, None))
                except Exception:
                    pass
        _resolve_aliases(runtime_ids)

        authoritative_ids = set(frozen_ids or runtime_ids)
        if frozen_ids and runtime_ids and not runtime_ids.issubset(frozen_ids):
            missile_id = str(getattr(missile_sim, 'uid', '') or getattr(missile_sim, 'missile_id', '') or '')
            conflict_key = (
                missile_id,
                tuple(sorted(frozen_ids)),
                tuple(sorted(runtime_ids)),
            )
            conflict_log = getattr(self, '_missile_target_conflict_log', None)
            if conflict_log is None:
                conflict_log = set()
                self._missile_target_conflict_log = conflict_log
            if conflict_key not in conflict_log:
                conflict_log.add(conflict_key)
                try:
                    self.task._log_key_event(
                        env_obj,
                        "",
                        "导弹目标冲突",
                        当前阶段="MISSILE_TARGET_RESOLVE",
                        进入函数="TacticalExecutor._collect_missile_target_aliases",
                        当前状态=(
                            f"missile={missile_id or '-'}, "
                            f"frozen={','.join(sorted(frozen_ids)) or '-'}, "
                            f"runtime={','.join(sorted(runtime_ids)) or '-'}"
                        ),
                        当前指令="frozen_target_authoritative",
                        退出条件="存在冻结目标则禁止runtime target_aircraft改写规避归属",
                        是否满足退出="是",
                    )
                except Exception:
                    pass
        return authoritative_ids, ("frozen" if frozen_ids else "runtime")

    def _iter_missile_simulators_for_agent(self, env, agent_id: str):
        enemy_prefix = 'B' if str(agent_id).startswith('A') else 'A'
        reference_env = getattr(self.task, '_get_reference_env', lambda _env=None: env)(env)
        env_chain = []
        for candidate in (
            env,
            getattr(env, '_original', None) if env is not None else None,
            reference_env,
            getattr(reference_env, '_original', None) if reference_env is not None else None,
            getattr(self.task, 'env', None),
        ):
            if candidate is None:
                continue
            if any(candidate is existing for existing in env_chain):
                continue
            env_chain.append(candidate)

        seen_missiles = set()
        agent_aliases = self._collect_agent_identity_aliases(env, agent_id)
        real_agent_id = next((alias for alias in agent_aliases if alias != str(agent_id)), None)

        for env_obj in env_chain:
            tempsims = getattr(env_obj, '_tempsims', None)
            if not isinstance(tempsims, dict):
                continue
            previous_filter = getattr(env_obj, '_missile_filter_real_id', None)
            if real_agent_id and hasattr(env_obj, 'resolve_real_agent_id'):
                env_obj._missile_filter_real_id = real_agent_id
            try:
                for missile_id, missile_sim in tempsims.items():
                    missile_id = str(missile_id)
                    if missile_id in seen_missiles or not missile_id.startswith(enemy_prefix):
                        continue
                    if not hasattr(missile_sim, 'get_position'):
                        continue
                    target_aliases, _target_source = self._collect_missile_target_aliases(env_obj, missile_sim)
                    if agent_aliases and target_aliases and not (agent_aliases & target_aliases):
                        continue
                    seen_missiles.add(missile_id)
                    yield missile_id, missile_sim
            finally:
                if real_agent_id and hasattr(env_obj, 'resolve_real_agent_id'):
                    env_obj._missile_filter_real_id = previous_filter

    def _resolve_missile_launcher(self, env, missile_sim):
        launcher_candidates = []

        def _append(candidate):
            cid = str(candidate or "").strip()
            if cid:
                launcher_candidates.append(cid)

        for attr_name in ('launcher_id', 'guide_agent_id', 'parent_uid'):
            try:
                _append(getattr(missile_sim, attr_name, None))
            except Exception:
                pass
        try:
            parent_aircraft = getattr(missile_sim, 'parent_aircraft', None)
        except Exception:
            parent_aircraft = None
        if parent_aircraft is not None:
            for attr_name in ('real_id', 'uid'):
                try:
                    _append(getattr(parent_aircraft, attr_name, None))
                except Exception:
                    pass

        reference_env = getattr(self.task, '_get_reference_env', lambda _env=None: env)(env)
        for launcher_id in launcher_candidates:
            for env_obj in (
                env,
                getattr(env, '_original', None) if env is not None else None,
                reference_env,
                getattr(reference_env, '_original', None) if reference_env is not None else None,
                getattr(self.task, 'env', None),
            ):
                if env_obj is None:
                    continue
                resolved_id = launcher_id
                if hasattr(env_obj, 'resolve_real_agent_id'):
                    try:
                        resolved_id = env_obj.resolve_real_agent_id(launcher_id) or resolved_id
                    except Exception:
                        resolved_id = launcher_id
                try:
                    aircraft = getattr(env_obj, 'agents', {}).get(resolved_id)
                except Exception:
                    aircraft = None
                if aircraft is not None and getattr(aircraft, 'is_alive', False):
                    return resolved_id, aircraft
        return None, None

    def _get_sim_entity(self, env, entity_id: str):
        entity_id = str(entity_id or "")
        if not entity_id:
            return None
        for attr_name in ("agents", "_jsbsims"):
            collection = getattr(env, attr_name, None)
            if isinstance(collection, dict):
                entity = collection.get(entity_id)
                if entity is not None:
                    return entity
        return None

    def _bearing_to_entity(self, env, agent_id: str, entity_id: str):
        aircraft = env.agents.get(agent_id)
        target = self._get_sim_entity(env, entity_id)
        if aircraft is None or target is None or not getattr(target, 'is_alive', True):
            return None
        try:
            own_pos = aircraft.get_position()
            target_pos = target.get_position()
            dx = float(target_pos[0]) - float(own_pos[0])
            dy = float(target_pos[1]) - float(own_pos[1])
            return float(np.rad2deg(np.arctan2(dy, dx)) % 360.0)
        except Exception:
            return None

    def _seed_beam_state(
        self,
        env,
        agent_id: str,
        target_heading: float,
        assigned_direction: str,
        start_time: float,
        state_token: str,
        execute_duration_s: float,
        total_duration_s: float,
    ) -> None:
        beam_state = self._get_beam_state_store()
        beam_state[agent_id] = {
            'start_time': float(start_time),
            'target_heading': float(target_heading) % 360.0,
            'assigned_direction': str(assigned_direction),
            'state_token': str(state_token),
            'execute_duration_s': float(max(1.0, execute_duration_s)),
            'total_duration_s': float(max(execute_duration_s, total_duration_s)),
        }

    def _prepare_tactical_evasion_beam(self, env, agent_id: str, state: dict, assigned_direction: str):
        reference_bearing = None
        reference_target_id = str(state.get('reference_target_id', '') or '')
        if reference_target_id:
            reference_bearing = self._bearing_to_entity(env, agent_id, reference_target_id)

        if reference_bearing is None:
            candidate_target_id = get_target_with_fallback(agent_id, env)
            if candidate_target_id:
                reference_bearing = self._bearing_to_entity(env, agent_id, candidate_target_id)
                if reference_bearing is not None:
                    state['reference_target_id'] = str(candidate_target_id)

        if reference_bearing is None:
            reference_bearing = self._get_enemy_bearing(env, agent_id)

        if reference_bearing is None:
            reference_bearing = state.get('reference_bearing')
        if reference_bearing is None:
            return False

        beam_angle_deg = float(state.get('beam_angle_deg', 85.0))
        target_heading = (
            (reference_bearing - beam_angle_deg) % 360.0
            if assigned_direction == 'left'
            else (reference_bearing + beam_angle_deg) % 360.0
        )
        state['reference_bearing'] = float(reference_bearing)
        state['beam_angle_deg'] = beam_angle_deg
        state_token = str(state.get('beam_state_token') or f"tactical_evasion:{agent_id}:{state.get('start', 0.0):.1f}")
        state['beam_state_token'] = state_token
        beam_state = self._get_beam_state_store()
        existing = beam_state.get(agent_id)
        if existing is None or str(existing.get('state_token', '')) != state_token:
            self._seed_beam_state(
                env,
                agent_id,
                target_heading=target_heading,
                assigned_direction=assigned_direction,
                start_time=float(state.get('start', 0.0)),
                state_token=state_token,
                execute_duration_s=min(2.5, max(1.8, float(state.get('ttl', 12.0)) * 0.18)),
                total_duration_s=min(4.0, max(3.0, float(state.get('ttl', 12.0)) * 0.30)),
            )
        return True

    def execute_tactical_evasion(self, env, agent_id: str) -> tuple:
        action = self._check_and_evade_missile(env, agent_id)
        if action is not None:
            return action
        self._clear_missile_evasion_session(agent_id)
        return None

    def _calculate_shortest_turn_to_north(self, current_heading: float) -> dict:
        return super()._calculate_shortest_turn_to_north(current_heading)

    def execute_tactical_turn(self, env, agent_id: str) -> tuple:
        return super().execute_tactical_turn(env, agent_id)

    def _heading_to_attack_wp_or_default(self, env, agent_id: str, default_heading: float) -> float:
        return super()._heading_to_attack_wp_or_default(env, agent_id, default_heading)

    def _execute_unified_pincer(self, env, agent_id: str, is_lead: bool) -> tuple:
        return _tues._execute_unified_pincer(self, env, agent_id, is_lead)

    def _execute_unified_high_low(self, env, agent_id: str, is_lead: bool) -> tuple:
        return _tues._execute_unified_high_low(self, env, agent_id, is_lead)

    def _execute_unified_side_by_side(self, env, agent_id: str, is_lead: bool) -> tuple:
        return _tues._execute_unified_side_by_side(self, env, agent_id, is_lead)

    def _execute_unified_front_back(self, env, agent_id: str, is_lead: bool) -> tuple:
        return _tues._execute_unified_front_back(self, env, agent_id, is_lead)

    def _execute_unified_drag_shoot(self, env, agent_id: str, is_lead: bool) -> tuple:
        return _tues._execute_unified_drag_shoot(self, env, agent_id, is_lead)

    def _get_enemy_bearing(self, env, agent_id: str) -> float:
        """Return bearing to the nearest alive enemy."""
        try:
            aircraft = env.agents[agent_id]
            current_pos = aircraft.get_position()
            min_dist = float('inf')
            enemy_bearing = 0.0

            enemy_prefix = 'B' if agent_id.startswith('A') else 'A'
            for eid, enemy in env.agents.items():
                if eid.startswith(enemy_prefix) and enemy.is_alive:
                    enemy_pos = enemy.get_position()
                    dist = np.linalg.norm(np.array(current_pos[:2]) - np.array(enemy_pos[:2]))
                    if dist < min_dist:
                        min_dist = dist
                        dx = enemy_pos[0] - current_pos[0]
                        dy = enemy_pos[1] - current_pos[1]
                        enemy_bearing = (np.rad2deg(np.arctan2(dy, dx))) % 360

            return enemy_bearing
        except Exception as e:
            logging.warning(f"Failed to get enemy bearing: {e}")
            return 0.0
    
    def _check_and_evade_missile(self, env, agent_id: str) -> tuple:
        current_step = getattr(env, 'current_step', -1)
        if not hasattr(self, '_evasion_result_cache'):
            self._evasion_result_cache = {}
        if not hasattr(self, '_evasion_call_counter'):
            self._evasion_call_counter = {}
        if getattr(self, '_evasion_cache_step', None) != current_step:
            self._evasion_cache_step = current_step
            self._evasion_result_cache.clear()
            self._evasion_call_counter.clear()

        cache_key = (agent_id, current_step)
        if cache_key in self._evasion_result_cache:
            return self._evasion_result_cache[cache_key]

        aircraft = env.agents.get(agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            self._evasion_result_cache[cache_key] = None
            return None

        current_pos = aircraft.get_position()
        current_time = current_step * float(getattr(env, 'time_interval', 1.0))
        formation_guidance_commit = False
        if agent_id.startswith('A') and hasattr(self.task, '_formation_has_active_guidance_commit'):
            try:
                formation_guidance_commit = bool(self.task._formation_has_active_guidance_commit(env, agent_id))
            except Exception:
                formation_guidance_commit = False
        nearest_enemy = None
        nearest_enemy_id = None
        nearest_enemy_distance = None
        entry_floor_m = None
        close_bvr_floor = False
        if agent_id.startswith('A'):
            try:
                nearest_enemy_id, nearest_enemy = self.task._get_nearest_alive_enemy_anywhere(env, agent_id)
            except Exception:
                nearest_enemy_id, nearest_enemy = None, None
            try:
                if nearest_enemy is not None and getattr(nearest_enemy, 'is_alive', False):
                    nearest_enemy_distance = float(self.task._calculate_distance_between(aircraft, nearest_enemy))
            except Exception:
                nearest_enemy_distance = None
            try:
                entry_floor_m = float(self.task._get_friendly_bvr_entry_floor_m(agent_id))
            except Exception:
                entry_floor_m = 40000.0
            close_bvr_floor = bool(
                nearest_enemy is not None
                and nearest_enemy_distance is not None
                and np.isfinite(nearest_enemy_distance)
                and nearest_enemy_distance <= entry_floor_m
            )
        try:
            current_heading = float(aircraft.get_property_value(c.attitude_psi_deg))
        except Exception:
            current_heading = float(np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad)))

        missile_threats = []
        missile_track = getattr(self, '_missile_track', {})
        for missile_id, missile_sim in self._iter_missile_simulators_for_agent(env, agent_id):
            try:
                missile_pos = missile_sim.get_position()
                distance = float(np.linalg.norm(np.array(current_pos) - np.array(missile_pos)))
                track_key = (agent_id, missile_id)
                prev = missile_track.get(track_key)
                closing_rate = 0.0
                tti = float('inf')
                missile_tof = float(getattr(missile_sim, '_t', 0.0) or 0.0)
                missile_phase = getattr(missile_sim, '_phase', None)
                terminal_phase = getattr(missile_sim.__class__, 'TERMINAL_PHASE', None)
                terminal_guidance = bool(
                    missile_phase is not None
                    and terminal_phase is not None
                    and missile_phase == terminal_phase
                )
                if prev is not None:
                    dt = current_time - float(prev.get('last_t', current_time))
                    if dt > 1e-3:
                        closing_rate = max(0.0, (float(prev.get('last_d', distance)) - distance) / dt)
                        if closing_rate > 1.0:
                            tti = distance / closing_rate
                missile_track[track_key] = {'last_d': distance, 'last_t': current_time}
                imminent_threat = bool(
                    terminal_guidance
                    or distance < 22000.0
                    or tti < 12.0
                )
                should_evade = bool(
                    imminent_threat
                    or distance < 38000.0
                    or tti < 28.0
                    or (distance < 52000.0 and (missile_tof >= 6.0 or closing_rate >= 220.0))
                    or (
                        distance < 62000.0
                        and np.isfinite(tti)
                        and tti < 48.0
                        and closing_rate >= 120.0
                    )
                )
                if should_evade:
                    dx = float(missile_pos[0]) - float(current_pos[0])
                    dy = float(missile_pos[1]) - float(current_pos[1])
                    bearing = float(np.rad2deg(np.arctan2(dy, dx)) % 360.0)
                    target_aliases, target_source = self._collect_missile_target_aliases(env, missile_sim)
                    launcher_id, launcher_aircraft = self._resolve_missile_launcher(env, missile_sim)
                    launcher_distance = None
                    if launcher_aircraft is not None:
                        try:
                            launcher_distance = float(self.task._calculate_distance_between(aircraft, launcher_aircraft))
                        except Exception:
                            launcher_distance = None
                    missile_threats.append((
                        0 if imminent_threat else 1,
                        0 if terminal_guidance else 1,
                        tti,
                        distance,
                        closing_rate,
                        missile_tof,
                        bearing,
                        missile_id,
                        terminal_guidance,
                        (
                            ",".join(sorted(target_aliases)) + f"|src={target_source}"
                            if target_aliases else f"-|src={target_source}"
                        ),
                        launcher_id,
                        launcher_aircraft,
                        launcher_distance,
                    ))
            except Exception:
                continue

        self._missile_track = missile_track
        if not missile_threats:
            self._clear_missile_evasion_session(agent_id)
            self._evasion_result_cache[cache_key] = None
            return None

        missile_threats.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        (
            _,
            _,
            closest_tti,
            closest_distance,
            closest_closing_rate,
            closest_missile_tof,
            closest_bearing,
            closest_missile_id,
            terminal_threat,
            closest_target_aliases,
            threat_launcher_id,
            threat_launcher_aircraft,
            threat_launcher_distance,
        ) = missile_threats[0]

        if not hasattr(self.task, 'tactical_evasion_direction_state'):
            self.task.tactical_evasion_direction_state = {}
        if not hasattr(self, '_missile_evasion_direction_session'):
            self._missile_evasion_direction_session = {}
        current_session = self._missile_evasion_direction_session.get(agent_id, {})
        if str(current_session.get('missile_id', '')) == str(closest_missile_id):
            assigned_direction = str(current_session.get('direction', 'right') or 'right')
        else:
            rel_bearing = ((closest_bearing - current_heading + 540.0) % 360.0) - 180.0
            assigned_direction = 'left' if rel_bearing >= 0.0 else 'right'
            self._missile_evasion_direction_session[agent_id] = {
                'missile_id': str(closest_missile_id),
                'direction': assigned_direction,
                'start_time': float(current_time),
            }
        self.task.tactical_evasion_direction_state[agent_id] = assigned_direction
        current_session = self._missile_evasion_direction_session.get(agent_id, {})
        missile_session_age_s = 0.0
        if str(current_session.get('missile_id', '')) == str(closest_missile_id):
            missile_session_age_s = max(0.0, float(current_time) - float(current_session.get('start_time', current_time)))

        if not hasattr(self.task, 'beam_direction_state'):
            self.task.beam_direction_state = {}
        self.task.beam_direction_state[agent_id] = assigned_direction

        try:
            from simulation.radar_manager import get_unified_radar_manager
            rwr_level = get_unified_radar_manager().get_rwr_threat_level(agent_id)
        except Exception:
            if closest_distance < 12000.0 or closest_tti < 6.0:
                rwr_level = 4
            elif closest_distance < 20000.0 or closest_tti < 12.0:
                rwr_level = 3
            else:
                rwr_level = 2

        effective_rwr_level = int(rwr_level)
        if terminal_threat or closest_distance < 18000.0 or closest_tti < 10.0:
            effective_rwr_level = max(effective_rwr_level, 4)
        elif closest_distance < 32000.0 or closest_tti < 20.0:
            effective_rwr_level = max(effective_rwr_level, 3)

        escape_arm_distance = float(
            getattr(self.task, '_get_friendly_bvr_escape_arming_m', lambda _aid: 60000.0)(agent_id)
        )
        escape_release_distance = float(
            getattr(self.task, '_get_friendly_bvr_escape_release_m', lambda _aid: 52000.0)(agent_id)
        )
        threat_reference_enemy = threat_launcher_aircraft if threat_launcher_aircraft is not None else nearest_enemy
        threat_reference_enemy_id = threat_launcher_id if threat_launcher_aircraft is not None else nearest_enemy_id
        threat_reference_enemy_distance = (
            threat_launcher_distance
            if threat_launcher_distance is not None and np.isfinite(threat_launcher_distance)
            else nearest_enemy_distance
        )
        escape_trigger_by_enemy = bool(
            threat_reference_enemy_distance is not None
            and np.isfinite(threat_reference_enemy_distance)
            and threat_reference_enemy_distance <= escape_arm_distance
        )
        escape_trigger_by_missile = bool(closest_distance <= escape_arm_distance)
        missile_escape_required = bool(
            terminal_threat
            or close_bvr_floor
            or escape_trigger_by_enemy
            or escape_trigger_by_missile
            or (
                effective_rwr_level >= 4
                and (
                    closest_distance <= max(45000.0, escape_release_distance)
                    or (np.isfinite(closest_tti) and closest_tti <= 32.0)
                    or closest_closing_rate >= 260.0
                )
            )
        )
        preserve_guidance = bool(
            formation_guidance_commit
            and not terminal_threat
            and closest_distance > 16000.0
            and (not np.isfinite(closest_tti) or closest_tti > 9.0)
            and not close_bvr_floor
            and not missile_escape_required
        )
        if not hasattr(self, '_bvr_escape_session'):
            self._bvr_escape_session = {}
        escape_session = self._bvr_escape_session.get(agent_id, {})
        stale_far_threat = bool(
            not terminal_threat
            and not close_bvr_floor
            and not missile_escape_required
            and effective_rwr_level >= 3
            and closest_distance >= max(68000.0, escape_arm_distance + 8000.0)
            and (
                threat_reference_enemy_distance is None
                or not np.isfinite(threat_reference_enemy_distance)
                or threat_reference_enemy_distance >= (escape_arm_distance + 35000.0)
            )
            and (not np.isfinite(closest_tti) or closest_tti >= 80.0)
            and closest_missile_tof >= 10.0
            and closest_closing_rate <= 320.0
            and missile_session_age_s >= 6.0
        )
        if stale_far_threat:
            if agent_id.startswith('A') and hasattr(self.task, '_log_key_event'):
                try:
                    phase_obj = self.task.state_manager.get_agent_phase(agent_id)
                except Exception:
                    phase_obj = None
                phase_str = phase_obj.value if hasattr(phase_obj, 'value') else phase_obj
                tti_text = f"{closest_tti:.1f}s" if np.isfinite(closest_tti) else "inf"
                self.task._log_key_event(
                    env,
                    agent_id,
                    "导弹规避释放",
                    current_time=current_time,
                    当前阶段=phase_str,
                    当前战术=getattr(self.task, 'selected_tactic', None),
                    进入函数="_check_and_evade_missile",
                    执行机动="RELEASE_STALE_FAR_THREAT",
                    当前状态=(
                        f"missile={closest_missile_id}, dist={closest_distance / 1000.0:.1f}km, "
                        f"tti={tti_text}, closing={closest_closing_rate:.1f}m/s, "
                        f"tof={closest_missile_tof:.1f}s, enemy_dist="
                        f"{(threat_reference_enemy_distance / 1000.0):.1f}km"
                        if threat_reference_enemy_distance is not None and np.isfinite(threat_reference_enemy_distance)
                        else (
                            f"missile={closest_missile_id}, dist={closest_distance / 1000.0:.1f}km, "
                            f"tti={tti_text}, closing={closest_closing_rate:.1f}m/s, "
                            f"tof={closest_missile_tof:.1f}s, enemy_dist=-"
                        )
                    ),
                    当前指令="clear_missile_evasion_session",
                    退出条件="far_stale_threat geometry satisfied",
                    是否满足退出="是",
                )
            self._clear_missile_evasion_session(agent_id)
            self._evasion_result_cache[cache_key] = None
            return None
        long_range_escape_active = bool(escape_session.get('active', False))
        if long_range_escape_active:
            long_range_escape_active = bool(
                str(escape_session.get('missile_id', '')) == str(closest_missile_id)
                and (
                    closest_distance <= (escape_arm_distance + 8000.0)
                    or (
                        threat_reference_enemy_distance is not None
                        and np.isfinite(threat_reference_enemy_distance)
                        and threat_reference_enemy_distance <= (escape_release_distance + 6000.0)
                    )
                    or terminal_threat
                    or effective_rwr_level >= 3
                )
            )
        elif missile_escape_required:
            long_range_escape_active = True
        elif closest_distance <= escape_release_distance:
            long_range_escape_active = True
        if long_range_escape_active:
            self._bvr_escape_session[agent_id] = {
                'active': True,
                'missile_id': str(closest_missile_id),
                'start_time': float(current_time),
            }
        else:
            self._bvr_escape_session.pop(agent_id, None)

        floor_override_action = None
        floor_override_active = False
        if (
            close_bvr_floor
            and threat_reference_enemy is not None
            and threat_reference_enemy_distance is not None
            and np.isfinite(threat_reference_enemy_distance)
            and hasattr(self.task, '_build_bvr_hard_escape_command')
        ):
            try:
                floor_override_action = self.task._build_bvr_hard_escape_command(
                    env,
                    agent_id,
                    threat_reference_enemy,
                    threat_reference_enemy_distance,
                    reason=f"missile_evasion_bvr_floor:{threat_reference_enemy_distance/1000.0:.1f}km",
                )
            except Exception:
                floor_override_action = None
            floor_override_active = floor_override_action is not None

        target_heading = None
        maneuver_name = "BEAM"
        floor_override_priority = bool(
            floor_override_active
            and (
                not np.isfinite(closest_tti)
                or closest_tti > 8.0
                or closest_distance > 6000.0
            )
        )
        if floor_override_priority:
            self._clear_beam_state(agent_id)
            action = floor_override_action
            maneuver_name = "BVR_STANDOFF_EVADE"
        elif long_range_escape_active and threat_reference_enemy is not None and threat_reference_enemy_distance is not None and np.isfinite(threat_reference_enemy_distance):
            self._clear_beam_state(agent_id)
            try:
                action = self.task._build_bvr_hard_escape_command(
                    env,
                    agent_id,
                    threat_reference_enemy,
                    threat_reference_enemy_distance,
                    reason=f"missile_evasion_long_range:{threat_reference_enemy_distance/1000.0:.1f}km",
                )
            except Exception:
                action = None
            if action is not None:
                maneuver_name = "BVR_HARD_ESCAPE_EVADE"
            else:
                try:
                    action = self.task._build_bvr_standoff_command(
                        env,
                        agent_id,
                        threat_reference_enemy,
                        threat_reference_enemy_distance,
                        reason=f"missile_evasion_long_range_fallback:{threat_reference_enemy_distance/1000.0:.1f}km",
                        allow_guidance_commit=True,
                    )
                except Exception:
                    action = None
                if action is not None:
                    maneuver_name = "BVR_PREEMPT_ESCAPE"
            if action is None and effective_rwr_level >= 4 and not preserve_guidance:
                action = self.task.maneuver_lib.execute_notch_back(env, agent_id, direction=assigned_direction)
                maneuver_name = "NOTCH_BACK"
            if action is None:
                floor_override_action = None
                floor_override_active = False
                long_range_escape_active = False
        elif effective_rwr_level >= 4 and not preserve_guidance:
            self._clear_beam_state(agent_id)
            action = self.task.maneuver_lib.execute_notch_back(env, agent_id, direction=assigned_direction)
            maneuver_name = "NOTCH_BACK"
        elif floor_override_active:
            self._clear_beam_state(agent_id)
            action = floor_override_action
            maneuver_name = "BVR_STANDOFF_EVADE"
        else:
            beam_angle_deg = 80.0 if preserve_guidance else 85.0
            target_heading = (
                (closest_bearing - beam_angle_deg) % 360.0
                if assigned_direction == 'left'
                else (closest_bearing + beam_angle_deg) % 360.0
            )
            session_start = float(
                self._missile_evasion_direction_session
                .get(agent_id, {})
                .get('start_time', current_time)
            )
            state_token = f"missile_evasion:{agent_id}:{closest_missile_id}:{session_start:.1f}"
            beam_state = self._get_beam_state_store()
            existing = beam_state.get(agent_id)
            if existing is None or str(existing.get('state_token', '')) != state_token:
                self._seed_beam_state(
                    env,
                    agent_id,
                    target_heading=target_heading,
                    assigned_direction=assigned_direction,
                    start_time=current_time,
                    state_token=state_token,
                    execute_duration_s=3.0 if preserve_guidance else 2.5,
                    total_duration_s=5.0 if preserve_guidance else 4.0,
                )
            action = self.task.maneuver_lib.execute_beam_maneuver(env, agent_id)

        if agent_id.startswith('A'):
            tti_text = f"{closest_tti:.1f}s" if np.isfinite(closest_tti) else "inf"
            if hasattr(self.task, '_log_key_event'):
                phase_obj = None
                try:
                    phase_obj = self.task.state_manager.get_agent_phase(agent_id)
                except Exception:
                    phase_obj = None
                phase_str = phase_obj.value if hasattr(phase_obj, 'value') else phase_obj
                self.task._log_key_event(
                    env,
                    agent_id,
                    "导弹规避推进",
                    current_time=current_time,
                    当前阶段=phase_str,
                    当前战术=getattr(self.task, 'selected_tactic', None),
                    进入函数="_check_and_evade_missile",
                    执行机动=maneuver_name,
                    当前状态=(
                        f"missile={closest_missile_id}, target={closest_target_aliases}, "
                        f"dist={closest_distance / 1000.0:.1f}km, tti={tti_text}, rwr={effective_rwr_level}, "
                        f"dir={assigned_direction}, enemy_dist={nearest_enemy_distance / 1000.0:.1f}km, "
                        f"floor={'Y' if close_bvr_floor else 'N'}, preserve={'Y' if preserve_guidance else 'N'}, "
                        f"escape={'Y' if long_range_escape_active else 'N'}, session_lock=Y"
                    ) if threat_reference_enemy_distance is not None and np.isfinite(threat_reference_enemy_distance) else (
                        f"missile={closest_missile_id}, target={closest_target_aliases}, "
                        f"dist={closest_distance / 1000.0:.1f}km, tti={tti_text}, rwr={effective_rwr_level}, "
                        f"dir={assigned_direction}, floor={'Y' if close_bvr_floor else 'N'}, "
                        f"preserve={'Y' if preserve_guidance else 'N'}, escape={'Y' if long_range_escape_active else 'N'}, "
                        f"threat_enemy={threat_reference_enemy_id or '-'}, session_lock=Y"
                    ),
                    目标点=(
                        f"beam_heading={target_heading:.1f}"
                        if maneuver_name == "BEAM" and target_heading is not None
                        else (f"nearest_enemy={threat_reference_enemy_id}" if maneuver_name in ("BVR_STANDOFF_EVADE", "BVR_HARD_ESCAPE_EVADE") else None)
                    ),
                    当前指令=str(action),
                    退出条件="threat cleared / tti released / session finalized",
                    是否满足退出="否",
                )
            elif current_step % 40 == 0:
                logging.info(
                    "[MISSILE_EVADE] %s missile=%s dist=%.1fkm tti=%s rwr=%d dir=%s",
                    agent_id,
                    closest_missile_id,
                    closest_distance / 1000.0,
                    tti_text,
                    effective_rwr_level,
                    assigned_direction,
                )

        if hasattr(self.task, 'is_agent_second_attack') and self.task.is_agent_second_attack(agent_id):
            if hasattr(self.task, '_set_agent_tactic'):
                self.task._set_agent_tactic(agent_id, 'UNIFIED_SECOND_ATTACK')
            else:
                self.task.selected_tactic = 'UNIFIED_SECOND_ATTACK'
        elif getattr(self.task, 'force_tactic', None):
            if hasattr(self.task, '_set_agent_tactic'):
                self.task._set_agent_tactic(agent_id, self.task.force_tactic)
            else:
                self.task.selected_tactic = self.task.force_tactic
        elif getattr(self.task, '_defense_prev_tactic', None):
            if hasattr(self.task, '_set_agent_tactic'):
                self.task._set_agent_tactic(agent_id, self.task._defense_prev_tactic)
            else:
                self.task.selected_tactic = self.task._defense_prev_tactic

        if self.evasion_min_distances.get(agent_id, float('inf')) < 40000.0:
            self.force_rtb_after_evasion[agent_id] = True

        self._evasion_result_cache[cache_key] = action
        return action
    def _execute_second_attack_drag_shoot(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
        return _terfh._execute_second_attack_drag_shoot(self, env, agent_id, is_lead, current_time)
    def _execute_second_attack_pincer(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
        return _terfh._execute_second_attack_pincer(self, env, agent_id, is_lead, current_time)
    def _execute_second_attack_high_low(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
        return _terfh._execute_second_attack_high_low(self, env, agent_id, is_lead, current_time)
    def _execute_second_attack_side_by_side(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
        return _terfh._execute_second_attack_side_by_side(self, env, agent_id, is_lead, current_time)
    def _execute_second_attack_front_back(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
        return _terfh._execute_second_attack_front_back(self, env, agent_id, is_lead, current_time)




