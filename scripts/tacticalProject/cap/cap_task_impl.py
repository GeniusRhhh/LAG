"""
CAP战术任务 - P1-4巡逻功能 + 状态机 + 感知层
整合坐标系统、FAOR、编队管理、巡逻状态机、CAP状态机
"""
import logging
import os
import json
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir

from .coordinate_system import CoordinateSystem, BattlefieldConfig
from .faor_manager import FAORManager, RiskZone
from .formation_manager import FormationManager, PatrolPhase
from .patrol_state_machine import PatrolStateMachine, PatrolBox, PatrolState
from .cap_state_machine import CAPStateMachine, CAPState, StateContext
from .control_ranges import ControlRanges, DEFAULT_RANGES
from .picture import Picture, Track, FusedTrack, TrackSource, MockAwacsDataSource, TrackFusion
from .tactics import CooperativeDetection, DetectionMode, CooperativeEngagement
from .tactics import FormationGuidance, VelocityCoordination
from .cap_radar import CAPRadarManager, RadarMode
from .mission_evaluator import MissionEvaluator, MissionResult
from .tactic_selector import TacticSelector, TacticType, TacticAssignment
from . import cap_picture_helpers as _cph
from . import cap_tactic_helpers as _ctah
try:
    from . import cap_task_refactor_helpers as _caprfh
except ImportError:
    import cap_task_refactor_helpers as _caprfh
try:
    from . import cap_guidance_helpers as _capgh
except ImportError:
    import cap_guidance_helpers as _capgh
try:
    from . import cap_picture_helpers as _capph
except ImportError:
    import cap_picture_helpers as _capph
try:
    from . import cap_tactic_helpers as _captach
except ImportError:
    import cap_tactic_helpers as _captach
try:
    from . import cap_engagement_helpers as _capegh
except ImportError:
    import cap_engagement_helpers as _capegh
try:
    from . import cap_tracking_helpers as _captrk
except ImportError:
    import cap_tracking_helpers as _captrk
try:
    from . import cap_rootcause_logging as _caproot
except ImportError:
    import cap_rootcause_logging as _caproot
try:
    from . import cap_task_init_helpers as _capinit
except ImportError:
    import cap_task_init_helpers as _capinit
try:
    from . import cap_task_step_helpers as _capstep
except ImportError:
    import cap_task_step_helpers as _capstep
try:
    from . import cap_lowlevel_helpers as _caplow
except ImportError:
    import cap_lowlevel_helpers as _caplow
from .cap_task_runtime_mixin import CAPTaskRuntimeMixin

# 🔥 导入雷达管理系统
try:
    from simulation.radar_manager import UnifiedRadarManager
    UNIFIED_RADAR_AVAILABLE = True
except ImportError:
    UNIFIED_RADAR_AVAILABLE = False
    UnifiedRadarManager = None

# 🔥 集成现有战术系统
import sys
import os
tactical_project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if tactical_project_path not in sys.path:
    sys.path.insert(0, tactical_project_path)

# 🔥 导入适配器模块
from .adapters import EnemyAIAdapter, MissileAdapter, IntentAdapter, TacticExecutorAdapter

try:
    from unified_enemy_tactical_ai import UnifiedEnemyTacticalAI
    ENEMY_AI_AVAILABLE = True
except ImportError:
    ENEMY_AI_AVAILABLE = False
    UnifiedEnemyTacticalAI = None

try:
    from missile_manager import MissileManager
    from tactical_state_manager import TacticalStateManager
    MISSILE_MANAGER_AVAILABLE = True
except ImportError:
    MISSILE_MANAGER_AVAILABLE = False
    MissileManager = None
    TacticalStateManager = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


_CAP_DEBUG_PRINT = os.environ.get('CAP_DEBUG_PRINT') == '1'
_CAP_VERIFY_TABLE = os.environ.get('CAP_VERIFY_TABLE', '0').strip().lower() not in ('0', 'false', 'no', 'off')
_CAP_CONTROL_DEBUG = os.environ.get('CAP_CONTROL_DEBUG', '0').strip().lower() in ('1', 'true', 'yes', 'on')


def _cap_print(*args, **kwargs):
    if _CAP_DEBUG_PRINT:
        print(*args, **kwargs)


def _cap_table_print(*args, **kwargs):
    # 验证用“真值/预警机/融合对比表”默认开启；如需彻底静默可设 CAP_VERIFY_TABLE=0
    if _CAP_VERIFY_TABLE or _CAP_DEBUG_PRINT:
        print(*args, **kwargs)


def _actor_uses_mlp_actlayer(state_dict) -> bool:
    return any(str(key).startswith('act.mlp.') for key in state_dict.keys())


class CAPTask(CAPTaskRuntimeMixin, MultipleCombatTask):
    """CAP战术任务 - 4机跑马圈巡逻"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        _capinit.initialize_cap_task(self)

    def _get_alive_enemy_ids(self, env) -> set:
        return {
            str(aid)
            for aid, aircraft in getattr(env, "agents", {}).items()
            if str(aid).startswith("B") and getattr(aircraft, "is_alive", False)
        }

    def _get_alive_awacs_tracks(self, env) -> dict:
        awacs = getattr(self, "awacs", None)
        if awacs is None:
            return {}
        try:
            tracks = awacs.get_tracks() or {}
        except Exception:
            return {}
        alive_enemy_ids = self._get_alive_enemy_ids(env)
        if not alive_enemy_ids:
            return {}
        return {
            str(tid): track
            for tid, track in tracks.items()
            if str(tid) in alive_enemy_ids
        }

    def _clear_awacs_tracks(self, env=None, current_time: float = None) -> None:
        awacs = getattr(self, "awacs", None)
        if awacs is None:
            return

        if current_time is None:
            try:
                current_time = float(getattr(env, "current_step", 0)) * float(getattr(env, "time_interval", 0.2))
            except Exception:
                current_time = 0.0

        friendly_positions = []
        if env is not None:
            for aid, aircraft in getattr(env, "agents", {}).items():
                if str(aid).startswith("A") and getattr(aircraft, "is_alive", False):
                    try:
                        pos = self._get_battlefield_pos(env, aid)
                        _, _, alt_m = aircraft.get_geodetic()
                        friendly_positions.append(np.array([pos[0], pos[1], alt_m * 0.001]))
                    except Exception:
                        continue

        try:
            if hasattr(awacs, "set_ground_truth"):
                awacs.set_ground_truth([])
            if hasattr(awacs, "update"):
                awacs.update(float(current_time), friendly_positions=friendly_positions)
            elif hasattr(awacs, "_tracks"):
                awacs._tracks = {}
        except Exception:
            if hasattr(awacs, "_tracks"):
                awacs._tracks = {}

        self._awacs_updated_this_step = False
        self._awacs_last_stamp = None
        self._last_awacs_info_time = None

    def _clear_enemy_contact_state(self, env, clear_picture: bool = False, reason: str = "") -> set:
        alive_enemy_ids = self._get_alive_enemy_ids(env)

        picture = getattr(self, "picture", None)
        if picture is not None:
            for tid in list(getattr(picture, "tracks", {}).keys()):
                tid = str(tid)
                if tid.startswith("B") and (clear_picture or tid not in alive_enemy_ids):
                    picture.remove_track(tid)

        radar = getattr(self, "cap_radar", None)
        if radar is not None:
            for aid, tracks in list(getattr(radar, "_radar_tracks", {}).items()):
                for tid in list(tracks.keys()):
                    tid = str(tid)
                    if tid.startswith("B") and tid not in alive_enemy_ids:
                        del tracks[tid]
                if str(aid).startswith("A"):
                    radar.lock_target(aid, None)

        if not alive_enemy_ids and hasattr(self, "track_fusion"):
            try:
                self.track_fusion.clear()
            except Exception:
                pass

        coop_engagement = getattr(self, "coop_engagement", None)
        if coop_engagement is not None:
            coop_engagement.assignments = {
                aid: asgn
                for aid, asgn in getattr(coop_engagement, "assignments", {}).items()
                if getattr(asgn, "target_id", None) in alive_enemy_ids
            }
            coop_engagement.tracking_status = {
                tid: status
                for tid, status in getattr(coop_engagement, "tracking_status", {}).items()
                if tid in alive_enemy_ids
            }
            coop_engagement._last_assignment_time = {
                tid: ts
                for tid, ts in getattr(coop_engagement, "_last_assignment_time", {}).items()
                if tid in alive_enemy_ids
            }
            if not alive_enemy_ids:
                coop_engagement.guidance_status.clear()

        coop_detection = getattr(self, "coop_detection", None)
        if coop_detection is not None:
            for attr in ("_last_track_time", "_last_known_targets", "_imm_estimators", "_target_covariances"):
                cache = getattr(coop_detection, attr, None)
                if isinstance(cache, dict):
                    setattr(
                        coop_detection,
                        attr,
                        {tid: value for tid, value in cache.items() if tid in alive_enemy_ids},
                    )
            if not alive_enemy_ids:
                coop_detection._target_lost_time = None

        for attr in ("_stable_tracking_elapsed", "_stable_tracking_last_ok", "_stable_tracking_trackers"):
            cache = getattr(self, attr, None)
            if isinstance(cache, dict):
                setattr(self, attr, {tid: value for tid, value in cache.items() if tid in alive_enemy_ids})
        ready_targets = getattr(self, "_stable_tracking_ready_announced", None)
        if isinstance(ready_targets, set):
            self._stable_tracking_ready_announced = {tid for tid in ready_targets if tid in alive_enemy_ids}

        if not alive_enemy_ids:
            self._clear_awacs_tracks(env)
            self._cooperative_tracking_active = False
            if hasattr(self, "_track_event_last"):
                self._track_event_last = {}
            if hasattr(self, "_tactic_assignments_by_agent"):
                self._tactic_assignments_by_agent = {}
            for attr in (
                "_last_tactic_assignments_by_pair",
                "_pair_target_memory",
                "_pair_target_lock_until_step",
                "_agent_template_target",
                "_agent_template_node_progress",
                "_pair_template_target",
                "_pair_template_node_progress",
                "_last_template_node_by_agent",
                "_second_attack_window_announced",
            ):
                cache = getattr(self, attr, None)
                if hasattr(cache, "clear"):
                    cache.clear()
            last_reason = getattr(self, "_last_enemy_contact_clear_reason", None)
            if reason and last_reason != reason:
                self._last_enemy_contact_clear_reason = reason
                log.info("[CAP_CONTACT_CLEAR] reason=%s cleared_all_enemy_contacts", reason)
        else:
            self._last_enemy_contact_clear_reason = None

        return alive_enemy_ids

    def _load_config_params(self):
        """从配置读取参数"""
        # 默认值
        self.a0100_lon = 120.6757
        self.a0100_lat = 60.0
        self.a0100_heading = 0.0
        self.wingman_y = 100.0
        self.lead_x_left = 75.0
        self.lead_x_right = 125.0
        self.wingman_x_left = 25.0
        self.wingman_x_right = 175.0

        # 拦截阶段目标信息来源：'auto' | 'awacs' | 'picture'
        # - auto: 有AWACS用AWACS，否则用picture融合航迹
        # - awacs: 仅用AWACS（更贴近上交版0–R_radar边界）
        # - picture: 仅用融合航迹（用于工程回退/对比）
        self.intercept_target_source = 'auto'

        # 实验对比模式：'proposed' | 'baseline'
        self.experiment_mode = 'proposed'

        # 复现：固定AWACS随机序列（可选）
        self.awacs_seed = None
        
        # 从aircraft_configs读取A0100位置
        if hasattr(self.config, 'aircraft_configs'):
            ac = self.config.aircraft_configs
            if 'A0100' in ac:
                init = ac['A0100'].get('init_state', {})
                if isinstance(init, dict):
                    self.a0100_lon = init.get('ic_long_gc_deg', self.a0100_lon)
                    self.a0100_lat = init.get('ic_lat_geod_deg', self.a0100_lat)
                    self.a0100_heading = init.get('ic_psi_true_deg', self.a0100_heading)

        # 读取拦截目标信息来源配置（兼容dict/对象两种）
        try:
            if isinstance(self.config, dict):
                self.intercept_target_source = self.config.get('cap_intercept_target_source', self.intercept_target_source)
                self.experiment_mode = self.config.get('cap_experiment_mode', self.experiment_mode)
                self.awacs_seed = self.config.get('cap_awacs_seed', self.awacs_seed)
                if self.awacs_seed is None:
                    self.awacs_seed = self.config.get('cap_rng_seed', self.awacs_seed)
            else:
                self.intercept_target_source = getattr(self.config, 'cap_intercept_target_source', self.intercept_target_source)
                self.experiment_mode = getattr(self.config, 'cap_experiment_mode', self.experiment_mode)
                self.awacs_seed = getattr(self.config, 'cap_awacs_seed', self.awacs_seed)
                if self.awacs_seed is None:
                    self.awacs_seed = getattr(self.config, 'cap_rng_seed', self.awacs_seed)
        except Exception:
            pass

        # 规范化
        try:
            self.experiment_mode = str(self.experiment_mode).strip().lower()
        except Exception:
            self.experiment_mode = 'proposed'
        if self.experiment_mode not in ('proposed', 'baseline'):
            self.experiment_mode = 'proposed'
        try:
            if self.awacs_seed is not None:
                self.awacs_seed = int(self.awacs_seed)
        except Exception:
            self.awacs_seed = None

    def _init_coordinate_system(self):
        """初始化坐标系统"""
        self.coord_sys = CoordinateSystem(BattlefieldConfig(
            a0100_lon=self.a0100_lon,
            a0100_lat=self.a0100_lat,
            a0100_heading=self.a0100_heading,
            a0100_x=self.lead_x_left,
            a0100_y=0.0
        ))
        
    def _init_faor(self):
        """初始化FAOR管理"""
        self.faor = FAORManager(width=200.0, length=300.0)

    def _init_formation(self):
        """初始化编队管理"""
        self.formation = FormationManager(
            wingman_y_offset=self.wingman_y,
            lead_x=(self.lead_x_left, self.lead_x_right),
            wingman_x=(self.wingman_x_left, self.wingman_x_right)
        )

    def _init_patrol_machines(self):
        """初始化巡逻状态机"""
        self.patrol_machines: Dict[str, PatrolStateMachine] = {}
        wy = self.wingman_y
        
        # 左编队巡逻区域: x从僚机位置到长机位置
        left_box = PatrolBox(x_min=self.wingman_x_left, x_max=self.lead_x_left, y_min=0, y_max=wy)
        # 右编队巡逻区域
        right_box = PatrolBox(x_min=self.lead_x_right, x_max=self.wingman_x_right, y_min=0, y_max=wy)
        
        # 长机初始热段(朝北)，僚机初始冷段(朝南)
        self.patrol_machines['A0100'] = PatrolStateMachine(left_box, initial_hot=True)
        self.patrol_machines['A0200'] = PatrolStateMachine(left_box, initial_hot=False)
        self.patrol_machines['A0300'] = PatrolStateMachine(right_box, initial_hot=True)
        self.patrol_machines['A0400'] = PatrolStateMachine(right_box, initial_hot=False)

    def _rebuild_patrol_machines_from_env(self, env) -> bool:
        """从reset后的真实位置重建巡逻状态机。

        注意：严格参考原始 SimplePatrolTask：
        - 使用 NED 坐标（east/north）转换为 km 作为状态机输入
        - 状态机输出的 0/90/180/270 视为“地球航向(真北=0)”
        """
        formations = {
            'left': ('A0100', 'A0200'),
            'right': ('A0300', 'A0400'),
        }

        machines: Dict[str, PatrolStateMachine] = {}
        built_any = False

        def _xy_km(aid: str) -> Tuple[float, float] | None:
            ac = env.agents.get(aid)
            if ac is None or not getattr(ac, 'is_alive', False):
                return None
            pos = ac.get_position()  # NED: (north, east, down)
            x_km = float(pos[1]) / 1000.0
            y_km = float(pos[0]) / 1000.0
            return x_km, y_km

        for _, (lead_id, wing_id) in formations.items():
            if lead_id not in env.agents or wing_id not in env.agents:
                continue
            if not env.agents[lead_id].is_alive or not env.agents[wing_id].is_alive:
                continue

            lead_xy = _xy_km(lead_id)
            wing_xy = _xy_km(wing_id)
            if lead_xy is None or wing_xy is None:
                continue

            x_vals = [float(lead_xy[0]), float(wing_xy[0])]
            y_vals = [float(lead_xy[1]), float(wing_xy[1])]
            box = PatrolBox(
                x_min=min(x_vals),
                x_max=max(x_vals),
                y_min=min(y_vals),
                y_max=max(y_vals),
            )

            machines[lead_id] = PatrolStateMachine(box, initial_hot=True)
            machines[wing_id] = PatrolStateMachine(box, initial_hot=False)
            built_any = True

            log.info(
                "[INIT] [CAP Init] box(%s/%s) x=[%.1f,%.1f]km y=[%.1f,%.1f]km (NED)",
                lead_id,
                wing_id,
                box.x_min,
                box.x_max,
                box.y_min,
                box.y_max,
            )

        if built_any:
            self.patrol_machines = machines
        return built_any

    def _load_baseline_models(self):
        """加载底层控制模型"""
        root = get_root_dir()
        f16_path = os.path.join(root, 'model', 'baseline_model.pt')
        f16_native_path = os.getenv(
            "CAP_ENEMY_F16_NATIVE_PATH",
            os.path.join(tactical_project_path, 'models', 'f16_cap_lowlevel_native.pt'),
        )
        f16_native_meta_path = os.getenv(
            "CAP_ENEMY_F16_NATIVE_META_PATH",
            os.path.join(tactical_project_path, 'models', 'f16_cap_lowlevel_native.json'),
        )
        su27_path = r"D:\Pycharm\LAG\lqyLAG\scripts\results\SingleControl\1\heading_su27\ppo\su27_baseline_v1\run40\actor_990.pt"
        
        self.f16_model = None
        self.f16_cap_model = None
        self.f16_cap_model_path = None
        self.f16_cap_command_semantics = None
        self.su27_model = None
        
        if os.path.exists(f16_path):
            try:
                f16_state_dict = torch.load(f16_path, map_location='cpu', weights_only=True)
                self.f16_model = BaselineActor(
                    input_dim=12,
                    use_mlp_actlayer=_actor_uses_mlp_actlayer(f16_state_dict),
                )
                self.f16_model.load_state_dict(f16_state_dict)
                self.f16_model.eval()
                self.f16_model_path = f16_path
                log.info(f"   F16模型路径: {f16_path}")
                log.info("   F16模型加载成功")
            except Exception as e:
                log.warning(f"   F16模型加载失败: {e}")

        if os.path.exists(f16_native_path):
            try:
                f16_native_state_dict = torch.load(f16_native_path, map_location='cpu', weights_only=True)
                self.f16_cap_model = BaselineActor(
                    input_dim=12,
                    use_mlp_actlayer=_actor_uses_mlp_actlayer(f16_native_state_dict),
                )
                self.f16_cap_model.load_state_dict(f16_native_state_dict)
                self.f16_cap_model.eval()
                self.f16_cap_model_path = f16_native_path
                self.f16_cap_command_semantics = 'cap_native_15x17x7'
                if os.path.exists(f16_native_meta_path):
                    try:
                        with open(f16_native_meta_path, 'r', encoding='utf-8') as meta_file:
                            metadata = json.load(meta_file)
                        self.f16_cap_command_semantics = str(
                            metadata.get('command_semantics', self.f16_cap_command_semantics)
                        )
                    except Exception as meta_exc:
                        log.warning(f"   F16 CAP Native元数据读取失败: {meta_exc}")
                semantics_override = str(os.getenv("CAP_ENEMY_F16_NATIVE_COMMAND_SEMANTICS", "")).strip()
                if semantics_override:
                    self.f16_cap_command_semantics = semantics_override
                log.info(f"   F16 CAP Native模型路径: {f16_native_path}")
                log.info(f"   F16 CAP Native语义: {self.f16_cap_command_semantics}")
                log.info("   F16 CAP Native模型加载成功")
            except Exception as e:
                log.warning(f"   F16 CAP Native模型加载失败: {e}")
        
        if os.path.exists(su27_path):
            try:
                self.su27_model = BaselineActor(input_dim=12, use_mlp_actlayer=True)
                sd = torch.load(su27_path, map_location='cpu', weights_only=True)
                if not any(k.startswith('act.mlp') for k in sd.keys()):
                    self.su27_model = BaselineActor(input_dim=12, use_mlp_actlayer=False)
                self.su27_model.load_state_dict(sd)
                self.su27_model.eval()
                log.info("   SU27模型加载成功")
            except Exception as e:
                log.warning(f"   SU27模型加载失败: {e}")

        self.friend_lowlevel_type = os.getenv("FRIEND_BASELINE_MODEL", "SU27").upper()
        self.enemy_lowlevel_type = os.getenv("ENEMY_BASELINE_MODEL", "F16").upper()
        self.enemy_use_f16_native = os.getenv("CAP_ENEMY_F16_NATIVE_ENABLED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.enemy_lowlevel_semantics = (
            self.f16_cap_command_semantics
            if self.enemy_lowlevel_type == 'F16' and self.enemy_use_f16_native and self.f16_cap_model is not None
            else 'legacy_3x5x3'
        )
        os.environ["CAP_ENEMY_F16_COMMAND_SEMANTICS"] = self.enemy_lowlevel_semantics
        if self.enemy_lowlevel_type == 'F16' and self.enemy_use_f16_native and self.f16_cap_model is not None:
            log.info("   敌方F16将使用CAP Native低层模型")
        elif self.enemy_lowlevel_type == 'F16' and self.f16_cap_model is not None and not self.enemy_use_f16_native:
            log.info("   敌方F16默认使用legacy baseline，禁用CAP Native低层模型")
        elif self.enemy_lowlevel_type == 'F16':
            log.info("   敌方F16仅使用baseline_model.pt，已禁用CAP Native低层模型")

    def _init_action_arrays(self):
        """初始化动作数组
        
        🔥 V3优化：增加航向控制密度，支持小角度调整（5°, 10°）
        关键：保持hdg_cmd=8为保持航向（向后兼容）
        """
        self.norm_alt = np.array([-1500,-1000,-750,-500,-300,-150,-50,0,50,150,300,500,750,1000,1500]) / 1000.0
        # 在保持hdg_cmd=8=0的前提下，使用标准的17维战术动作空间
        self.norm_hdg = np.array([
            -np.pi, -2*np.pi/3, -np.pi/2, -5*np.pi/12, -np.pi/3, -np.pi/4, -np.pi/6, 
            -np.pi/12,  # -15°
            0,  # 保持航向（hdg_cmd=8）
            np.pi/12,   # +15°
            np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2, 2*np.pi/3, np.pi
        ])
        self.norm_vel = np.array([-150,-100,-50,0,50,100,150]) / 100.0

    def _get_bridge_tactic_info(self, agent_id: str) -> Dict[str, str]:
        bridge = getattr(self, 'tactical_bridge', None)
        if not bridge or not getattr(bridge, 'initialized', False):
            return {}
        try:
            info = bridge.get_agent_tactic_info(agent_id) or {}
            return info if isinstance(info, dict) else {}
        except Exception:
            return {}

    def _get_bridge_tactic_name(self, agent_id: str) -> str:
        return str(self._get_bridge_tactic_info(agent_id).get('tactic', '') or '')

    def _get_bridge_phase_name(self, agent_id: str) -> str:
        return str(self._get_bridge_tactic_info(agent_id).get('phase', '') or '')

    def _is_friendly_rootcause_return(self, agent_id: str) -> bool:
        if not agent_id.startswith('A'):
            return False
        info = self._get_bridge_tactic_info(agent_id)
        tactic = str(info.get('tactic', '') or '').upper()
        phase = str(info.get('phase', '') or '').upper()
        return tactic == 'TACTICAL_TURN' or phase in ('TR_DOR', 'DOR_DR', 'DR_MAR', 'BEYOND_MAR')

    def build_return_command_with_alt_guard(
        self,
        env,
        agent_id: str,
        target_heading: float,
    ) -> Tuple[int, int, int]:
        current_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg))
        # get_position()[2] is the local NED/down-axis position, not the aircraft's
        # true MSL altitude. Using it here corrupts the enemy RTB altitude bands.
        current_alt = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
        current_speed = float(np.linalg.norm(env.agents[agent_id].get_velocity()))
        current_vc = float(env.agents[agent_id].get_property_value(c.velocities_vc_mps))
        vertical_speed = -float(env.agents[agent_id].get_property_value(c.velocities_v_down_mps))
        heading_diff = ((float(target_heading) - current_heading + 180.0) % 360.0) - 180.0
        command_speed = current_vc if agent_id.startswith('B') else current_speed

        if current_alt < 1800.0:
            heading_diff = float(np.clip(heading_diff, -12.0, 12.0))
        elif current_alt < 3000.0:
            heading_diff = float(np.clip(heading_diff, -20.0, 20.0))
        elif current_alt < 4500.0:
            heading_diff = float(np.clip(heading_diff, -30.0, 30.0))
        elif current_alt < 5500.0:
            heading_diff = float(np.clip(heading_diff, -45.0, 45.0))

        if current_alt < 1800.0 or command_speed < 170.0 or vertical_speed < -15.0:
            heading_diff = 0.0
        elif current_alt < 3000.0 and (command_speed < 210.0 or vertical_speed < -8.0):
            heading_diff = float(np.clip(heading_diff, -8.0, 8.0))
        elif current_alt < 4500.0 and (command_speed < 230.0 or vertical_speed < -5.0):
            heading_diff = float(np.clip(heading_diff, -15.0, 15.0))

        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.deg2rad(heading_diff))))

        if agent_id.startswith('B') and current_alt >= 5500.0:
            if current_vc < 120.0 or vertical_speed < -30.0:
                cmd = (10, 8, 6)
            elif current_vc < 140.0 or vertical_speed < -20.0:
                cmd = (9, 8, 5)
            elif current_vc < 160.0 or vertical_speed < -12.0:
                cmd = (7, 8, 5)
            elif current_vc < 190.0:
                cmd = (7, hdg_cmd, 4)
            else:
                cmd = (7, hdg_cmd, 3)
        elif current_alt < 1200.0:
            if command_speed < 190.0 or vertical_speed < -8.0:
                cmd = (12, 8, 6)
            elif command_speed < 170.0:
                cmd = (9, hdg_cmd, 6)
            elif command_speed < 190.0:
                cmd = (10, hdg_cmd, 6)
            elif vertical_speed < -8.0:
                cmd = (12, hdg_cmd, 5)
            else:
                cmd = (11, hdg_cmd, 5)
        elif current_alt < 2500.0:
            if command_speed < 170.0 or vertical_speed < -12.0:
                cmd = (12, 8, 6)
            elif command_speed < 190.0:
                cmd = (10, hdg_cmd, 6)
            elif command_speed < 220.0 or vertical_speed < -6.0:
                cmd = (11, hdg_cmd, 5)
            else:
                cmd = (10, hdg_cmd, 4)
        elif current_alt < 4000.0:
            if command_speed < 210.0:
                cmd = (10, hdg_cmd, 6)
            elif vertical_speed < -4.0:
                cmd = (10, hdg_cmd, 5)
            else:
                cmd = (9, hdg_cmd, 4)
        elif current_alt < 5500.0:
            if command_speed < 220.0:
                cmd = (9, hdg_cmd, 5)
            elif vertical_speed < -3.0:
                cmd = (9, hdg_cmd, 4)
            else:
                cmd = (7, hdg_cmd, 4)
        elif command_speed < 220.0:
            cmd = (7, hdg_cmd, 5)
        else:
            cmd = (7, hdg_cmd, 3)

        root_trace = os.environ.get('CAP_ROOTCAUSE_TRACE', '').strip().lower() in ('1', 'true', 'yes', 'on')
        current_step = int(getattr(env, 'current_step', self.step_count))
        if root_trace and current_step % 10 == 0:
            if agent_id.startswith('A') and self._is_friendly_rootcause_return(agent_id):
                phase_name = self._get_bridge_phase_name(agent_id) or 'UNKNOWN'
                log.warning(
                    f"🧩 [友机根因记录-返航高层] {agent_id} phase={phase_name} "
                    f"target_hdg={float(target_heading):.1f}° hdg_diff={heading_diff:.1f}° "
                    f"cmd=({cmd[0]},{cmd[1]},{cmd[2]}) alt={current_alt:.0f}m tas={current_speed:.0f}m/s vc={current_vc:.0f}m/s "
                    f"v_up={vertical_speed:.1f}m/s"
                )
            elif agent_id.startswith('B'):
                log.warning(
                    f"🧩 [根因链-返航对齐] {agent_id} target_hdg={float(target_heading):.1f}deg "
                    f"hdg_diff={heading_diff:.1f}deg cmd=({cmd[0]},{cmd[1]},{cmd[2]}) "
                    f"alt={current_alt:.0f}m tas={current_speed:.0f}m/s vc={current_vc:.0f}m/s "
                    f"v_up={vertical_speed:.1f}m/s"
                )

        return cmd

    def _reset_lowlevel_rnn_state(self, agent_id: str):
        self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)

    def get_lowlevel_rnn_norm(self, agent_id: str) -> float:
        state = self._inner_rnn_states.get(agent_id)
        if state is None:
            return 0.0
        try:
            return float(np.linalg.norm(state))
        except Exception:
            return 0.0

    def reset_enemy_lowlevel_rnn_state(self, agent_id: str) -> bool:
        if not agent_id.startswith('B'):
            return False
        self._reset_lowlevel_rnn_state(agent_id)
        return True

    def _rewrite_friendly_recovery_command(
        self,
        env,
        agent_id: str,
        alt_cmd: int,
        hdg_cmd: int,
        spd_cmd: int,
    ) -> Tuple[int, int, int]:
        if not agent_id.startswith('A'):
            return alt_cmd, hdg_cmd, spd_cmd

        try:
            aircraft = env.agents[agent_id]
            current_alt = float(aircraft.get_property_value(c.position_h_sl_m))
            tas_mps = float(aircraft.get_property_value(c.velocities_vc_mps))
            v_up_mps = -float(aircraft.get_property_value(c.velocities_v_down_mps))
            pitch_deg = np.degrees(float(aircraft.get_property_value(c.attitude_theta_rad)))
            roll_deg = np.degrees(float(aircraft.get_property_value(c.attitude_phi_rad)))
            tactic_name = self._get_bridge_tactic_name(agent_id)

            recovery_state = self._friendly_recovery_state.setdefault(
                agent_id,
                {'active': False, 'last_reset_step': -9999, 'last_reset_time': -1e9},
            )

            sensitive_tactic = tactic_name in (
                'TACTICAL_TURN',
                'FORMATION_RESET',
                'UNIFIED_SECOND_ATTACK',
                'ADAPTIVE_ATTACK',
            )
            severe_bank = abs(roll_deg) > 120.0
            severe_sink = v_up_mps < -35.0
            low_energy = (
                tas_mps < 190.0                              # 近失速：无条件介入
                or (tas_mps < 220.0 and v_up_mps < -8.0)   # 中低速+下沉
                or (tas_mps < 260.0 and v_up_mps < -15.0)  # 早期预警：中速段大下沉（原来盲区）
            )

            recover_required = (
                (current_alt < 4500.0 and (low_energy or v_up_mps < -15.0 or abs(roll_deg) > 75.0 or pitch_deg < -12.0))
                or (current_alt < 2800.0 and (tas_mps < 220.0 or v_up_mps < -8.0 or abs(roll_deg) > 55.0))
                or current_alt < 1200.0
                or (sensitive_tactic and current_alt < 6500.0 and (tas_mps < 210.0 or v_up_mps < -6.0))
            )
            recover_hold = recovery_state.get('active', False) and (
                current_alt < 4800.0
                or tas_mps < 240.0
                or v_up_mps < -2.0
                or abs(roll_deg) > 45.0
                or pitch_deg < -5.0
            )

            if not (recover_required or recover_hold):
                recovery_state['active'] = False
                return alt_cmd, hdg_cmd, spd_cmd

            entered_recovery = not recovery_state.get('active', False)
            recovery_state['active'] = True

            sim_step = int(getattr(env, 'current_step', self.step_count))
            sim_dt = float(getattr(env, 'time_interval', 0.2))
            current_time_s = sim_step * sim_dt
            reset_cooldown_s = float(os.getenv('CAP_RECOVERY_RNN_RESET_COOLDOWN', '2.0'))
            can_reset_by_step = (self.step_count - int(recovery_state.get('last_reset_step', -9999))) >= 8
            can_reset_by_time = (current_time_s - float(recovery_state.get('last_reset_time', -1e9))) >= reset_cooldown_s
            reset_required = (
                entered_recovery
                or severe_bank
                or severe_sink
                or current_alt < 1200.0
                or (current_alt < 2500.0 and v_up_mps < -25.0)
            )
            if reset_required and can_reset_by_step and can_reset_by_time:
                self._reset_lowlevel_rnn_state(agent_id)
                recovery_state['last_reset_step'] = self.step_count
                recovery_state['last_reset_time'] = current_time_s
                log.warning(
                    f"🔁 [友机恢复重置-{agent_id}] tactic={tactic_name or 'UNKNOWN'} "
                    f"alt={current_alt:.0f}m tas={tas_mps:.0f}m/s v_up={v_up_mps:.1f}m/s "
                    f"pitch={pitch_deg:.1f}° roll={roll_deg:.1f}° -> reset SU27 RNN"
                )

            desired_turn_deg = float(np.degrees(self.norm_hdg[int(np.clip(hdg_cmd, 0, len(self.norm_hdg) - 1))]))
            if current_alt < 1800.0 or severe_bank or severe_sink or tas_mps < 175.0:
                safe_turn_deg = 0.0
            elif current_alt < 3200.0 or abs(roll_deg) > 75.0 or v_up_mps < -15.0:
                safe_turn_deg = float(np.clip(desired_turn_deg, -8.0, 8.0))
            else:
                safe_turn_deg = float(np.clip(desired_turn_deg, -15.0, 15.0))

            safe_hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.deg2rad(safe_turn_deg))))
            safe_alt_cmd = alt_cmd
            if current_alt < 1800.0 or v_up_mps < -25.0 or pitch_deg < -18.0:
                safe_alt_cmd = max(safe_alt_cmd, 12)
            elif current_alt < 3200.0 or v_up_mps < -12.0 or pitch_deg < -8.0:
                safe_alt_cmd = max(safe_alt_cmd, 11)
            elif current_alt < 4800.0 or v_up_mps < -4.0:
                safe_alt_cmd = max(safe_alt_cmd, 10)

            # 速度档位按风险分级：濒危恢复允许 5/6，常态保持保守档位。
            critical_recovery = (current_alt < 2000.0) or (tas_mps < 160.0) or (v_up_mps < -18.0)
            recovery_needed = (current_alt < 3500.0) or (tas_mps < 190.0) or (v_up_mps < -10.0)
            if critical_recovery:
                target_spd_cmd = 6
            elif recovery_needed:
                target_spd_cmd = 5
            elif tas_mps < 230.0 or v_up_mps < -4.0:
                target_spd_cmd = 4
            elif tas_mps > 320.0 and v_up_mps > 2.0:
                target_spd_cmd = 2
            else:
                target_spd_cmd = 3

            max_spd_cmd = 6 if recovery_needed else 4
            safe_spd_cmd = int(np.clip(max(spd_cmd, target_spd_cmd), 2, max_spd_cmd))

            if entered_recovery or self.step_count % 10 == 0:
                log.warning(
                    f"🛡️ [友机离散恢复-{agent_id}] tactic={tactic_name or 'UNKNOWN'} "
                    f"alt={current_alt:.0f}m tas={tas_mps:.0f}m/s v_up={v_up_mps:.1f}m/s "
                    f"pitch={pitch_deg:.1f}° roll={roll_deg:.1f}° "
                    f"cmd=({alt_cmd},{hdg_cmd},{spd_cmd})->({safe_alt_cmd},{safe_hdg_cmd},{safe_spd_cmd})"
                )

            return safe_alt_cmd, safe_hdg_cmd, safe_spd_cmd
        except Exception:
            return alt_cmd, hdg_cmd, spd_cmd

    def _apply_friendly_safety_override(self, env, agent_id: str, norm_act: np.ndarray) -> np.ndarray:
        if not agent_id.startswith('A'):
            return norm_act

        try:
            aircraft = env.agents[agent_id]
            current_alt = float(aircraft.get_property_value(c.position_h_sl_m))
            tas_mps = float(aircraft.get_property_value(c.velocities_vc_mps))
            v_up_mps = -float(aircraft.get_property_value(c.velocities_v_down_mps))
            pitch_deg = np.degrees(float(aircraft.get_property_value(c.attitude_theta_rad)))
            roll_deg = np.degrees(float(aircraft.get_property_value(c.attitude_phi_rad)))

            terminal_danger = current_alt < 220.0 and (v_up_mps < -25.0 or pitch_deg < -20.0)
            if not terminal_danger:
                return norm_act

            safe_act = np.array(norm_act, dtype=np.float32, copy=True)
            safe_act[3] = 1.0
            safe_act[2] = 0.0
            safe_act[0] = float(np.clip(-roll_deg / 140.0, -0.35, 0.35))
            safe_act[1] = max(float(safe_act[1]), 0.75 if tas_mps >= 150.0 else 0.55)
            safe_act = np.clip(safe_act, [-1, -1, -1, 0], [1, 1, 1, 1]).astype(np.float32)

            if self.step_count % 15 == 0:
                log.warning(
                    f"🛟 [友机末级保护-{agent_id}] "
                    f"alt={current_alt:.0f}m tas={tas_mps:.0f}m/s v_up={v_up_mps:.1f}m/s "
                    f"pitch={pitch_deg:.1f}° roll={roll_deg:.1f}° -> act={safe_act}"
                )

            return safe_act
        except Exception:
            return norm_act

    def normalize_action(self, env, agent_id: str, action: np.ndarray) -> np.ndarray:
        """Delegate to extracted helper to keep CAPTask focused on orchestration."""
        return _caprfh.normalize_action(self, env, agent_id, action)

    def reset(self, env):
        """重置任务状态"""
        self._inner_rnn_states = {aid: np.zeros((1, 1, 128), dtype=np.float32) for aid in env.agents}
        self._native_residual_target_state = {}
        self._friendly_recovery_state = {}
        self._friendly_sim_recreate_state = {}
        self._enemy_envelope_state = {}
        self._enemy_energy_state = {}
        self._enemy_hard_recovery_state = {}
        self._enemy_rnn_reset_state = {}
        self._enemy_sim_recreate_state = {}
        self.step_count = 0
        self.aircraft_states = {}

        self._safety_diag_last_print_step = {}
        self._safety_diag_last_level = {}

        ret = super().reset(env)

        # 关键：episode级重置（确保验证/对比可复现、避免跨episode状态泄漏）
        try:
            # 重置AWACS随机源与内部缓存
            self.awacs = MockAwacsDataSource(seed=getattr(self, 'awacs_seed', None))
        except Exception:
            pass

        try:
            # 重置协同探测内部IMM/缓存，避免上一局协方差/last_track_time影响本局
            self.coop_detection = CooperativeDetection(faor_width=200.0, faor_length=300.0)
        except Exception:
            pass

        try:
            # 重置交战分配、雷达管理器等缓存
            self.coop_engagement = CooperativeEngagement()
            self.cap_radar = CAPRadarManager()
            self.formation_guidance = FormationGuidance(formation_pairs=[
                ('A0100', 'A0200'),
                ('A0300', 'A0400'),
            ])
            self.velocity_coordination = VelocityCoordination(formation_pairs=[
                ('A0100', 'A0200'),
                ('A0300', 'A0400'),
            ])
        except Exception:
            pass

        # reset后用真实位置重建巡逻矩形（若失败则保留配置初始化的默认矩形）
        self._rebuild_patrol_machines_from_env(env)

        # 与原始巡逻任务一致：不强行覆盖 state（由 PatrolStateMachine(initial_hot=...) 决定）
        for aid, pm in self.patrol_machines.items():
            self._last_patrol_state[aid] = pm.state
        
        # 重置编队状态
        self.formation.cycle_count = 0
        
        # 🔥 重置CAP状态机和感知层
        self.cap_state_machine = CAPStateMachine(DEFAULT_RANGES)
        self.picture = Picture(faor_length=300.0)
        self.track_fusion.clear()
        
        if hasattr(self, 'tactical_bridge') and self.tactical_bridge:
            self.tactical_bridge.reset(env)
        if hasattr(self, 'enemy_adapter') and self.enemy_adapter:
            try:
                self.enemy_adapter.reset(env)
            except Exception as exc:
                log.warning(f"[CAP Reset] enemy_adapter.reset failed: {exc}")
        if hasattr(self, '_enemy_safe_teacher_stats'):
            self._enemy_safe_teacher_stats = {}
        # self._enemy_bridge_takeover_pairs = set()
            
        self._tactic_assignments_by_agent = {}
        self._last_tactic_assignment = None
        self._last_tactic_assignments_by_pair.clear()
        self._pair_target_memory.clear()
        self._pair_target_lock_until_step.clear()
        self._agent_template_target.clear()
        self._agent_template_node_progress.clear()
        self._pair_template_target.clear()
        self._pair_template_node_progress.clear()
        self._last_template_node_by_agent.clear()
        self._second_attack_window_announced.clear()
        self.mission_start_time = env.current_step * 0.2  # 假设0.2s/步
        for aid in self.formation.current_phase:
            is_lead = self.formation.is_lead(aid)
            self.formation.current_phase[aid] = PatrolPhase.HOT if is_lead else PatrolPhase.COLD

        return ret

    def _get_battlefield_pos(self, env, agent_id: str) -> Tuple[float, float]:
        """获取战场相对坐标"""
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 0, 0
        geo = ac.get_geodetic()
        return self.coord_sys.geodetic_to_battlefield(geo[0], geo[1])

    def _get_heading(self, env, agent_id: str) -> float:
        """获取当前航向（地球坐标系，度）"""
        ac = env.agents.get(agent_id)
        if not ac:
            return 0
        return np.degrees(ac.get_property_value(c.attitude_heading_true_rad)) % 360

    def get_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3

        if agent_id == 'B0100' and os.environ.get('CAP_B0100_DEEP_TRACE', '1').strip().lower() in ('1', 'true', 'yes', 'on'):
            try:
                current_alt = float(ac.get_property_value(c.position_h_sl_m))
                current_vc = float(ac.get_property_value(c.velocities_vc_mps))
                current_hdg = float(np.degrees(ac.get_property_value(c.attitude_heading_true_rad))) % 360.0
                current_pos = ac.get_position()
                g_load = float(ac.get_property_value(c.accelerations_n_pilot_z_norm)) if hasattr(c, 'accelerations_n_pilot_z_norm') else 0.0
                log.warning(
                    "[B0100][T+%07.1fs][get_action][entry][state] cap_state=%s alt=%.3fm vc=%.3fmps hdg=%.3fdeg pos=%s g=%.3f",
                    float(self.step_count * 0.2),
                    getattr(self.cap_state_machine, 'state', 'UNKNOWN'),
                    current_alt,
                    current_vc,
                    current_hdg,
                    np.array2string(np.asarray(current_pos), precision=6, separator=",", threshold=np.inf, max_line_width=100000),
                    g_load,
                )
            except Exception as exc:
                log.warning("[B0100][T+%07.1fs][get_action][entry][state] failed=%s", float(self.step_count * 0.2), exc)

        if agent_id.startswith('B'):
            if os.environ.get('CAP_ENEMY_SIMPLE', '0').strip().lower() in ('1', 'true', 'yes', 'on'):
                return self._get_enemy_action_simple(env, agent_id)
            return self._get_enemy_action(env, agent_id)

        cap_state = self.cap_state_machine.state
        if cap_state == CAPState.PATROL:
            return self._get_patrol_action(env, agent_id)
        if cap_state == CAPState.INTERCEPT:
            return self._get_intercept_action(env, agent_id)
        if cap_state == CAPState.ENGAGE:
            return self._get_engage_action(env, agent_id)
        if cap_state == CAPState.EVADE:
            return self._get_engage_action(env, agent_id)
        if cap_state == CAPState.RTB:
            return self._get_rtb_action(env, agent_id)
        return self._get_patrol_action(env, agent_id)

    def _get_enemy_action_simple(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _capact._get_enemy_action_simple(self, env, agent_id)
    
    def _get_intercept_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        """拦截动作：完整7层算法架构（算法0）
        
        按照《协同探测算法报告》实现完整流程：
        层1-信息层 → 层2-估计层(IMM-EKF) → 层3-预测层(sigma_man) → 
        层4-区域层(R_target) → 层5-规划层(编队引导) → 层6-协调层(速度) → 
        层7-执行层(动态调整)
        """
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3
        
        # 获取当前位置和航向
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)
        current_time = self.step_count * 0.2

        # ========== Baseline：不使用协同算法（直飞敌方中心 + 固定速度） ==========
        if getattr(self, 'experiment_mode', 'proposed') == 'baseline':
            awacs_tracks = self.awacs.get_tracks() if hasattr(self, 'awacs') and self.awacs is not None else {}
            pts = []
            for t in awacs_tracks.values():
                pos = getattr(t, 'position', None)
                if pos is not None and len(pos) >= 2:
                    pts.append((float(pos[0]), float(pos[1])))
            if not pts:
                for tr in self._get_picture_threats(current_time=current_time, purpose="search"):
                    if hasattr(tr, 'x') and hasattr(tr, 'y'):
                        pts.append((float(tr.x), float(tr.y)))
            if not pts:
                return self._get_patrol_action(env, agent_id)

            ex = float(np.mean([p[0] for p in pts]))
            ey = float(np.mean([p[1] for p in pts]))
            target_hdg = float(np.degrees(np.arctan2(ex - x, ey - y)) % 360)

            heading_diff = target_hdg - current_hdg
            if heading_diff > 180:
                heading_diff -= 360
            elif heading_diff < -180:
                heading_diff += 360

            if abs(heading_diff) < 5.0:
                hdg_cmd = 8  # 保持
            else:
                distances = np.abs(self.norm_hdg - np.radians(heading_diff))
                hdg_cmd = int(np.argmin(distances))

            alt_cmd = 7  # 保持高度
            spd_cmd = 3  # 保持速度
            return alt_cmd, hdg_cmd, spd_cmd
        
        # ========== 层1：信息层 - 预警机信息质量评估 ==========
        # 从AWACS获取航迹（通过awacs对象）
        awacs_tracks = {}
        if hasattr(self, 'awacs') and self.awacs is not None:
            awacs_tracks = self.awacs.get_tracks()
        has_awacs_info = bool(awacs_tracks)

        # 与CAP上下文保持一致：AWACS短暂掉线采用grace window，避免同一步出现
        # “状态栏=预警机在线，但7层链路认为无预警信息”从而切换数据源与触发不必要抖动。
        if not has_awacs_info:
            grace_s = float(getattr(self, '_awacs_grace_seconds', 15.0))
            last_t = getattr(self, '_last_awacs_info_time', None)
            if last_t is not None and (current_time - float(last_t)) <= grace_s:
                has_awacs_info = True
        
        # 获取所有目标位置（按配置决定来源）
        source_mode = getattr(self, 'intercept_target_source', 'auto')
        use_awacs_tracks = (source_mode in ('auto', 'awacs')) and has_awacs_info
        if source_mode == 'picture':
            use_awacs_tracks = False

        target_positions_raw = {}
        target_measure_times = {}
        if use_awacs_tracks:
            for tid, t in awacs_tracks.items():
                pos = getattr(t, 'position', None)
                if pos is None or len(pos) < 2:
                    continue
                target_positions_raw[tid] = (float(pos[0]), float(pos[1]))
                ts = getattr(t, 'timestamp', None)
                target_measure_times[tid] = float(ts) if ts is not None else current_time
        else:
            # 若无AWACS，尝试用雷达航迹更新时间戳驱动“是否新量测”
            radar_update_times = {}
            try:
                for _, tracks in self.cap_radar._radar_tracks.items():
                    for tid, trk in tracks.items():
                        last_upd = float(getattr(trk, 'last_update', 0.0))
                        if last_upd <= 0:
                            continue
                        radar_update_times[tid] = max(radar_update_times.get(tid, 0.0), last_upd)
            except Exception:
                radar_update_times = {}

            for track in self._get_picture_threats(current_time=current_time, purpose="search"):
                if hasattr(track, 'x') and hasattr(track, 'y'):
                    target_positions_raw[track.track_id] = (float(track.x), float(track.y))
                    # 关键：若无雷达last_update，则优先使用picture航迹自己的timestamp，
                    # 避免把“旧航迹”误当成每步都有新量测导致IMM反复update。
                    ts_pic = getattr(track, 'timestamp', None)
                    if ts_pic is not None:
                        target_measure_times[track.track_id] = float(ts_pic)
                    else:
                        target_measure_times[track.track_id] = radar_update_times.get(track.track_id, current_time)
        
        if not target_positions_raw:
            # 无目标信息，回退到巡逻
            return self._get_patrol_action(env, agent_id)
        
        # 目标丢失检测（覆盖“掉出当前输入列表”的目标）
        target_lost = self.coop_detection.any_target_lost(current_time)
        
        # ========== 层2：估计层 - IMM-EKF状态估计（算法1.1）==========
        target_positions_estimated = {}
        for tid, pos in target_positions_raw.items():
            meas_time = float(target_measure_times.get(tid, current_time))
            last_meas = self.coop_detection.get_last_track_time(tid)

            if last_meas is None or meas_time > last_meas + 1e-6:
                # 有新量测：执行IMM更新（时间戳用量测时间）
                est_pos, P = self.coop_detection.update_target_state_imm(
                    target_id=tid,
                    position=pos,
                    current_time=meas_time
                )
            else:
                # 无新量测：只做预测输出，不刷新last_track_time
                est_pos, P = self.coop_detection.predict_target_state(tid, current_time)

            target_positions_estimated[tid] = est_pos
        
        # ========== 层3：预测层 - 机动不确定性传播（算法2.5）==========
        sigma_man, r_targets = self.coop_detection.propagate_sigma_man(
            current_time=current_time,
            gamma=0.99,  # 99%置信度
            target_ids=list(target_positions_estimated.keys()),
        )
        
        # ========== 层4：区域层 - 分布范围与R_target计算（公式2.4）==========
        sigma_source_km = 2.5 if use_awacs_tracks else 0.5
        sigma_enemy, R_target = self.coop_detection.compute_distribution_range(
            target_positions=target_positions_estimated,
            current_time=current_time,
            sigma_awacs=sigma_source_km,
            k_sigma=3.0       # 3倍标准差覆盖
        )

        # 工程增强：可选上限钳制，避免极端不确定性把R_target放大到数百公里
        # 默认不启用；如需启用可设置环境变量 CAP_R_TARGET_MAX_KM=240
        try:
            raw_cap = os.environ.get('CAP_R_TARGET_MAX_KM', '').strip()
            if raw_cap:
                cap_km = float(raw_cap)
                if cap_km > 0:
                    R_target = min(float(R_target), cap_km)
        except Exception:
            pass
        
        # 估计敌方编队中心（使用IMM估计后的位置）
        enemy_center, _ = self.coop_detection.estimate_enemy_center(target_positions_estimated)
        
        # ========== 调试输出：层1-4（每30秒输出一次，仅A0100）==========
        if _CAP_DEBUG_PRINT and self.cap_state_machine.state == CAPState.INTERCEPT and agent_id == 'A0100' and self.step_count % 150 == 0:
            log.info("=" * 80)
            log.info(f"[算法0] 协同探测7层架构 | 时间={current_time:.1f}s 步数={self.step_count}")
            log.info(f"[层1-信息] 预警信息={'有' if has_awacs_info else '无'} | "
                     f"目标数={len(target_positions_raw)} | 目标丢失={'是' if target_lost else '否'}")
            log.info(f"[层2-估计] IMM-EKF已更新 {len(target_positions_estimated)} 个目标状态")
            for tid, pos in list(target_positions_estimated.items())[:3]:  # 只显示前3个
                r = r_targets.get(tid, 0)
                log.info(f"  目标{tid}: 位置=({pos[0]:.1f},{pos[1]:.1f})km 置信半径={r:.2f}km")
            log.info(f"[层3-预测] sigma_man={sigma_man:.2f}km (机动不确定性)")
            log.info(f"[层4-区域] sigma_enemy={sigma_enemy:.2f}km | R_target={R_target:.2f}km")
            log.info(f"  敌方中心=({enemy_center[0]:.1f},{enemy_center[1]:.1f})km")
        
        # 收集所有我方飞机位置和速度
        fighter_positions = {}
        fighter_speeds = {}
        current_headings = {}
        for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
            if aid in env.agents and env.agents[aid].is_alive:
                fighter_positions[aid] = self._get_battlefield_pos(env, aid)
                current_headings[aid] = self._get_heading(env, aid)
                try:
                    speed_mps = env.agents[aid].get_property_value(c.velocities_vc_mps)
                    fighter_speeds[aid] = float(speed_mps)
                except:
                    fighter_speeds[aid] = 250.0
        
        my_current_speed = fighter_speeds.get(agent_id, 250.0)
        
        # ========== 层5：规划层 - 编队引导（算法2.6）==========
        # 使用完整链路计算的R_target
        guidance = self.formation_guidance.compute_guidance(
            enemy_center=enemy_center,
            confidence_radius=R_target,  # ← 使用完整算法链计算的R_target
            fighter_positions=fighter_positions
        )
        
        # 获取当前飞机的引导目标
        my_guidance = guidance.get(agent_id)
        spd_cmd = 3  # 默认：保持速度
        target_hdg = current_hdg  # 默认保持当前航向
        
        if my_guidance:
            target_hdg_raw = my_guidance.target_heading
            
            # ========== 层6：协调层 - 速度协调（算法2.7）==========
            target_positions_for_speed = {
                fid: g.target_point for fid, g in guidance.items()
            }
            # 🔥 V2：传递任务阶段给速度协调算法
            current_phase = self.cap_state_machine.state.value if hasattr(self.cap_state_machine, 'state') else 'INTERCEPT'
            speed_commands = self.velocity_coordination.compute_coordinated_speeds(
                fighter_positions=fighter_positions,
                fighter_speeds=fighter_speeds,
                target_positions=target_positions_for_speed,
                mission_phase=current_phase  # 传递任务阶段
            )
            
            # 获取速度命令
            cmd = self.velocity_coordination.get_command(agent_id)
            target_speed_raw = my_current_speed
            if cmd:
                target_speed_raw = cmd.target_speed
                speed_diff = target_speed_raw - my_current_speed
                spd_cmd = self._speed_error_to_spd_cmd(speed_diff)
            
            # ========== 层7：执行层 - 动态路径调整（算法2.10）==========
            # 构造目标航向和速度字典
            target_headings = {fid: g.target_heading for fid, g in guidance.items()}
            target_speeds = {}
            for fid in fighter_speeds:
                cmd_fid = self.velocity_coordination.get_command(fid)
                target_speeds[fid] = cmd_fid.target_speed if cmd_fid else fighter_speeds[fid]
            
            # 事件检测
            # AWACS更新：由_update_awacs_data()统一计算，保证所有飞机同一步一致
            event_awacs_update = bool(getattr(self, '_awacs_updated_this_step', False)) and has_awacs_info

            event_maneuver = sigma_man > 5.0  # 机动不确定性过大
            event_awacs_lost = not has_awacs_info
            event_target_lost_flag = target_lost
            
            # 调用动态路径调整（含平滑）
            new_headings, new_speeds, action_taken = self.coop_detection.dynamic_path_adjustment(
                current_headings=current_headings,
                current_speeds=fighter_speeds,
                target_headings=target_headings,
                target_speeds=target_speeds,
                event_awacs_update=event_awacs_update,
                event_maneuver_detected=event_maneuver,
                event_awacs_lost=event_awacs_lost,
                event_target_lost=event_target_lost_flag,
                alpha=0.7  # 平滑系数
            )
            
            # 使用平滑后的航向和速度
            target_hdg = new_headings.get(agent_id, target_hdg_raw)
            target_speed_smoothed = new_speeds.get(agent_id, target_speed_raw)
            
            # 重新计算速度命令（基于平滑后的速度）
            speed_diff_smoothed = target_speed_smoothed - my_current_speed
            spd_cmd = self._speed_error_to_spd_cmd(speed_diff_smoothed)
            
            # ========== 调试输出：层5-7（每30秒输出一次，仅A0100）==========
            if _CAP_DEBUG_PRINT and self.cap_state_machine.state == CAPState.INTERCEPT and agent_id == 'A0100' and self.step_count % 150 == 0:
                log.info(f"[层5-规划] 编队引导（使用R_target={R_target:.2f}km）")
                for fid, g in guidance.items():
                    log.info(f"  {fid}: 目标点=({g.target_point[0]:.1f},{g.target_point[1]:.1f}) "
                             f"航向={g.target_heading:.1f}° 距离={g.distance_to_target:.1f}km")
                log.info(f"[层6-协调V2] 速度协调 | 阶段={current_phase}")
                for fid in fighter_speeds:
                    cmd_fid = self.velocity_coordination.get_command(fid)
                    if cmd_fid:
                        log.info(f"  {fid}: 当前={fighter_speeds[fid]:.0f}m/s → "
                                 f"目标={cmd_fid.target_speed:.0f}m/s (ETA={cmd_fid.eta:.1f}s)")
                # 显示阶段约束
                constraints = self.velocity_coordination._get_phase_constraints()
                log.info(f"  阶段约束: v_min={constraints['v_min']:.0f} v_nominal={constraints['v_nominal']:.0f} v_max={constraints['v_max']:.0f} m/s")
                log.info(f"[层7-执行] 动态路径调整 | 动作={action_taken}")
                log.info(f"  事件: AWACS更新={'是' if event_awacs_update else '否'} | "
                         f"机动检测={'是' if event_maneuver else '否'} | "
                         f"AWACS丢失={'是' if event_awacs_lost else '否'} | "
                         f"目标丢失={'是' if event_target_lost_flag else '否'}")
                log.info(f"  {agent_id}: 当前航向={current_hdg:.1f}° → 原始目标={target_hdg_raw:.1f}° → 平滑后={target_hdg:.1f}° | "
                         f"速度 {my_current_speed:.0f}→{target_speed_smoothed:.0f}m/s")
                log.info(
                    f"  {agent_id}: speed_error={speed_diff_smoothed:+.1f}m/s -> spd_cmd={spd_cmd} "
                    f"(norm_vel={self.norm_vel[spd_cmd]:.3f})"
                )

                # 可选：输出僚机规划/平滑后的航向（用于定位“绕路/回正慢”）
                try:
                    if os.environ.get('CAP_DEBUG_WINGMEN', '').strip().lower() in ('1', 'true', 'yes', 'y', 'on'):
                        for wid in ('A0200', 'A0400'):
                            if wid in current_headings and wid in target_headings:
                                hdg_cur = float(current_headings[wid])
                                hdg_raw = float(target_headings[wid])
                                hdg_new = float(new_headings.get(wid, hdg_raw))
                                log.info(f"  {wid}: 当前航向={hdg_cur:.1f}° → 原始目标={hdg_raw:.1f}° → 平滑后={hdg_new:.1f}°")
                except Exception:
                    pass
                log.info("=" * 80)
        else:
            # 无引导信息，直接指向敌方中心
            dx = enemy_center[0] - x
            dy = enemy_center[1] - y
            target_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        
        # 计算航向命令
        diff = target_hdg - current_hdg
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        
        # 修复V3：航向命令计算 - 支持小角度调整，保持hdg_cmd=8为保持航向
        diff_rad = np.radians(diff)
        
        if abs(diff) < 5.0:
            hdg_cmd = 8  # 保持当前航向（norm_hdg[8]=0）
        else:
            hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - diff_rad)))
        
        # 🔥 诊断：打印航向命令详情（每30秒，仅A0100）
        if _CAP_DEBUG_PRINT and self.cap_state_machine.state == CAPState.INTERCEPT and agent_id == 'A0100' and self.step_count % 150 == 0:
            log.info(f"[航向命令V3] 当前:{current_hdg:.1f}° → 目标:{target_hdg:.1f}° | "
                     f"差值:{diff:.1f}° ({diff_rad:.3f}rad) | "
                     f"命令:hdg_cmd={hdg_cmd} (norm_hdg[{hdg_cmd}]={self.norm_hdg[hdg_cmd]:.3f}rad={np.degrees(self.norm_hdg[hdg_cmd]):.1f}°)")
        
        return 7, hdg_cmd, spd_cmd
    
    def _get_engage_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        """交战动作：直接向威胁方向飞行，处理导弹发射
        
        简化逻辑：不使用复杂战术执行器，直接计算航向
        关键节点: LR(100km)发射, TR(80km)接力, LR'(42km)二次发射
        
        🔥 修复：使用协同交战分配结果，而非简单选择最近威胁
        """
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3
        
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)
        current_time = self.step_count * 0.2
        current_step = int(getattr(env, 'current_step', self.step_count))
        suppress_store = getattr(self, '_defensive_engagement_suppressed_until', None)
        suppressed_until_step = -1
        defensive_engagement_suppressed = False
        if isinstance(suppress_store, dict):
            try:
                suppressed_until_step = int(suppress_store.get(agent_id, -1))
            except Exception:
                suppressed_until_step = -1
            defensive_engagement_suppressed = suppressed_until_step > current_step
        
        assign = self._get_pair_tactic_assignment(agent_id)

        target_threat = None
        if assign and getattr(assign, 'target_id', None):
            for threat in self._get_picture_threats(current_time=current_time, purpose="decision"):
                if threat.track_id == assign.target_id:
                    target_threat = threat
                    break

        if target_threat is None:
            target_threat = self._get_nearest_visible_pair_threat(agent_id, x, y)

        if target_threat is None and hasattr(self, 'coop_engagement'):
            assignment = self.coop_engagement.get_assignment(agent_id)
            if assignment:
                for threat in self._get_picture_threats(current_time=current_time, purpose="decision"):
                    if threat.track_id == assignment.target_id:
                        target_threat = threat
                        break

        if target_threat is None:
            target_threat = self._get_picture_nearest_threat(np.array([x, y, 8.0]), current_time=current_time, purpose="decision")

        maneuver_action = None
        should_shoot_with_tactic = True
        alive_enemy_exists = any(
            str(other_id).startswith('B' if str(agent_id).startswith('A') else 'A')
            and getattr(other_aircraft, 'is_alive', False)
            for other_id, other_aircraft in getattr(env, 'agents', {}).items()
        )
        bridge_tactic_name = str(self._get_bridge_tactic_name(agent_id) or '').upper()
        block_bridge_tactical_turn = (
            agent_id.startswith('A')
            and alive_enemy_exists
            and bridge_tactic_name == 'TACTICAL_TURN'
        )
        if assign and (assign.shooter_id == agent_id or assign.support_id == agent_id):
            try:
                if block_bridge_tactical_turn:
                    if self.step_count % 25 == 0:
                        log.warning(f"🛑 [CAP返航拦截] {agent_id} bridge tactic=TACTICAL_TURN 但敌机仍存活，改为直接接敌")
                elif hasattr(self, 'tactical_bridge') and self.tactical_bridge and assign.tactic in (TacticType.T_DS, TacticType.T_PA, TacticType.T_HL, TacticType.T_SBS, TacticType.T_FB, TacticType.T_TE):
                    # 真正执行旧战术逻辑（不再强行覆盖底层战术意图，让底层算法自主选择）
                    
                    # 真正执行旧战术逻辑
                    maneuver_action = self.tactical_bridge.get_action(env, agent_id, self)
                    
                    if assign.tactic == TacticType.T_TE:
                        should_shoot_with_tactic = False
                else:
                    # 备用回退逻辑
                    if assign.tactic == TacticType.T_DS: maneuver_action = self._execute_drag_shoot(env, agent_id)
                    elif assign.tactic == TacticType.T_PA: maneuver_action = self._execute_pincer_attack(env, agent_id)
                    elif assign.tactic == TacticType.T_HL: maneuver_action = self._execute_high_low_attack(env, agent_id)
                    elif assign.tactic == TacticType.T_SBS: maneuver_action = self._execute_side_by_side(env, agent_id)
                    elif assign.tactic == TacticType.T_FB: maneuver_action = self._execute_front_back(env, agent_id)
                    elif assign.tactic == TacticType.T_TE:
                        maneuver_action = self._get_evade_action(env, agent_id)
                        should_shoot_with_tactic = False
                    elif assign.tactic == TacticType.T_TT:
                        maneuver_action = self._execute_tactical_turn(env, agent_id)
            except Exception as e:
                log.error(f"[战术方法异常] {agent_id} {assign.tactic.value}: {e}")

        if target_threat is None:
            if maneuver_action is not None:
                return maneuver_action
            return self._get_intercept_action(env, agent_id)
        
        distance = np.sqrt((target_threat.x - x)**2 + (target_threat.y - y)**2)
        launch_target_threat = target_threat
        launch_distance = float(distance)
        try:
            candidate_threats = self._get_picture_threats(current_time=current_time, purpose="decision")
        except Exception:
            candidate_threats = []
        if candidate_threats and not defensive_engagement_suppressed:
            best_candidate = None
            best_distance = float(launch_distance)
            best_score = float("-inf")
            current_target_id = getattr(target_threat, "track_id", None)
            for candidate in candidate_threats:
                candidate_id = getattr(candidate, "track_id", None)
                if not candidate_id:
                    continue
                candidate_distance = float(np.sqrt((candidate.x - x) ** 2 + (candidate.y - y) ** 2))
                stable_ready = self._has_stable_dual_tracking(candidate_id)
                first_window = (DEFAULT_RANGES.TR - 6.0) <= candidate_distance <= (DEFAULT_RANGES.LR + 8.0)
                second_window = DEFAULT_RANGES.MAR < candidate_distance <= (DEFAULT_RANGES.LR_PRIME + 18.0)
                if not (stable_ready or first_window or second_window):
                    continue
                candidate_y = float(getattr(candidate, "y", 300.0))
                zone_bonus = 40.0 if 0.0 <= candidate_y < 200.0 else 10.0 if 0.0 <= candidate_y <= 300.0 else 0.0
                window_bonus = 140.0 if first_window else 80.0 if second_window else 0.0
                stable_bonus = 60.0 if stable_ready else 0.0
                same_bonus = 12.0 if candidate_id == current_target_id else 0.0
                reference_distance = 88.0 if first_window else 44.0
                score = window_bonus + stable_bonus + zone_bonus + same_bonus - abs(candidate_distance - reference_distance)
                if score > best_score:
                    best_candidate = candidate
                    best_distance = candidate_distance
                    best_score = score
            current_in_window = (DEFAULT_RANGES.TR - 2.0) <= launch_distance <= (DEFAULT_RANGES.LR + 4.0)
            best_in_window = (DEFAULT_RANGES.TR - 6.0) <= best_distance <= (DEFAULT_RANGES.LR + 8.0)
            if (
                best_candidate is not None
                and (
                    (not current_in_window and best_in_window)
                    or (
                        getattr(best_candidate, "track_id", None) != current_target_id
                        and best_score >= 40.0
                    )
                )
            ):
                launch_target_threat = best_candidate
                launch_distance = float(best_distance)
                if self.step_count % 25 == 0:
                    log.info(
                        f"[协同发射目标] {agent_id} maneuver={current_target_id}({distance:.1f}km) "
                        f"launch={launch_target_threat.track_id}({launch_distance:.1f}km)"
                    )
        is_second_attack = DEFAULT_RANGES.is_second_attack_zone(launch_distance)
        # 允许对子内射手和支援机都进入发射判定，避免长期只有单机发射。
        can_shoot = (
            assign is None
            or agent_id == getattr(assign, 'shooter_id', None)
            or agent_id == getattr(assign, 'support_id', None)
        ) and should_shoot_with_tactic and not defensive_engagement_suppressed

        if is_second_attack and not defensive_engagement_suppressed:
            window_key = (agent_id, target_threat.track_id)
            if window_key not in self._second_attack_window_announced:
                self._second_attack_window_announced.add(window_key)
                log.info(
                    f"🔁 [二次进攻窗口] {agent_id}->{launch_target_threat.track_id} "
                    f"距离{launch_distance:.1f}km | MTR'={DEFAULT_RANGES.MTR_PRIME:.0f} LR'={DEFAULT_RANGES.LR_PRIME:.0f} TR'={DEFAULT_RANGES.TR_PRIME:.0f}"
                )
        elif defensive_engagement_suppressed and self.step_count % 40 == 0:
            log.info(
                "🛡️ [CAP_ENGAGE_SUPPRESS] %s target=%s until=%.1fs",
                agent_id,
                getattr(launch_target_threat, "track_id", "-"),
                max(0.0, (suppressed_until_step - current_step) * float(getattr(env, "time_interval", 0.2))),
            )
        
        # 导弹发射检查（要求发射前稳定双机跟踪）
        can_launch = self._has_stable_dual_tracking(launch_target_threat.track_id)
        gate_mode = "dual_track"
        adaptive_attack = False
        semantic_second_attack = False
        if hasattr(self, '_get_agent_tactic'):
            try:
                adaptive_attack = self._get_agent_tactic(agent_id) == 'ADAPTIVE_ATTACK'
            except Exception:
                adaptive_attack = False
        if hasattr(self, 'is_agent_second_attack'):
            try:
                semantic_second_attack = bool(self.is_agent_second_attack(agent_id))
            except Exception:
                semantic_second_attack = False
        if (
            (not defensive_engagement_suppressed)
            and (not can_launch)
            and (is_second_attack or semantic_second_attack or adaptive_attack or launch_distance <= 45.0)
        ):
            single_ship_ready = launch_distance <= 65.0
            if single_ship_ready:
                can_launch = True
                gate_mode = "single_ship_override"
        should_launch_req = False
        if can_shoot:
            should_launch_req = self.missile_adapter.should_launch(
                agent_id,
                launch_target_threat.track_id,
                launch_distance,
                (is_second_attack or semantic_second_attack or adaptive_attack or launch_distance <= 45.0),
                current_time,
                env,
            )

        if can_shoot and should_launch_req and can_launch:
            trackers = self._stable_tracking_trackers.get(launch_target_threat.track_id, [])
            hold = float(self._stable_tracking_elapsed.get(launch_target_threat.track_id, 0.0))
            missile_id = self.missile_adapter.execute_launch(agent_id, launch_target_threat.track_id, current_time, env)
            if missile_id:
                self._guidance_verify['prelaunch_gate_pass_count'] += 1
                pass_by_target = self._guidance_verify['prelaunch_gate_pass_by_target']
                pass_by_target[launch_target_threat.track_id] = int(pass_by_target.get(launch_target_threat.track_id, 0)) + 1
                first_pass = self._guidance_verify['prelaunch_gate_first_pass_time_by_target']
                if launch_target_threat.track_id not in first_pass:
                    first_pass[launch_target_threat.track_id] = float(current_time)
                log.info(f"[导弹发射] [{agent_id}] -> {launch_target_threat.track_id} @ {launch_distance:.0f}km")
                log.info(
                    f"[发射门禁通过] shooter={agent_id} target={launch_target_threat.track_id} "
                    f"stable={hold:.1f}s trackers={trackers} missile={missile_id} mode={gate_mode}"
                )
        elif can_shoot and should_launch_req and not can_launch:
            self._guidance_verify['prelaunch_gate_block_count'] += 1
            block_by_target = self._guidance_verify['prelaunch_gate_block_by_target']
            block_by_target[launch_target_threat.track_id] = int(block_by_target.get(launch_target_threat.track_id, 0)) + 1
            if self.step_count % 50 == 0:
                hold = self._stable_tracking_elapsed.get(launch_target_threat.track_id, 0.0)
                trackers = self._stable_tracking_trackers.get(launch_target_threat.track_id, [])
                log.info(f"⏳ [{agent_id}] 发射保持: {launch_target_threat.track_id} 稳定跟踪{hold:.1f}s/{self._stable_tracking_window_s:.0f}s")
                log.info(
                    f"[发射门禁阻断] shooter={agent_id} target={launch_target_threat.track_id} "
                    f"stable={hold:.1f}s trackers={trackers} mode={gate_mode}"
                )
        
        if maneuver_action is not None:
            return maneuver_action
        
        # 直接计算目标航向（指向威胁）
        dx = target_threat.x - x
        dy = target_threat.y - y
        target_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        
        # 航向差计算
        diff = target_hdg - current_hdg
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        
        if abs(diff) < 5:
            return 7, 8, 3
        
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        return 7, hdg_cmd, 3
    
    def _execute_drag_shoot(self, env, agent_id: str) -> Tuple[int, int, int]:
        """拖曳射击：沿用旧 run_simulation 的节点化动作序列。"""
        # 🔥 诊断10：战术方法入口（每60步打印一次）
        if self.step_count % 60 == 0:
            log.info(f"[战术方法-拖曳射击] {agent_id} 进入_execute_drag_shoot()")
        
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3

        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)

        assign = self.coop_engagement.get_assignment(agent_id) if hasattr(self, 'coop_engagement') else None
        if hasattr(self, '_tactic_assignments_by_agent') and agent_id in self._tactic_assignments_by_agent:
            assign = self._tactic_assignments_by_agent.get(agent_id)

        target_threat = None
        target_id = getattr(assign, 'target_id', None)
        if target_id:
            for threat in self._get_picture_threats(purpose="decision"):
                if threat.track_id == target_id:
                    target_threat = threat
                    break
        if target_threat is None:
            target_threat = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
        if target_threat is None:
            return self._get_intercept_action(env, agent_id)

        dx = target_threat.x - x
        dy = target_threat.y - y
        distance = float(np.sqrt(dx * dx + dy * dy))
        base_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        node = DEFAULT_RANGES.get_current_node(distance, getattr(self, '_locked_compression', None))
        node = self._lock_template_node(agent_id, target_threat.track_id, node)

        is_shooter = bool(assign and getattr(assign, 'shooter_id', None) == agent_id)
        side_sign = -1.0 if agent_id in ('A0100', 'A0200') else 1.0

        # 默认参数
        target_hdg = base_hdg
        alt_cmd = 7
        spd_cmd = 3

        # 节点驱动机动（严格参考旧链路：长机诱敌、僚机前出、TR后脱离）
        if node in ('BEYOND_NLT', 'NLT_MELD'):
            target_hdg = base_hdg if is_shooter else (base_hdg - side_sign * 35.0) % 360
            spd_cmd = 4 if is_shooter else 3
        elif node == 'MELD_MTR':
            target_hdg = base_hdg
            spd_cmd = 4 if is_shooter else 3
        elif node == 'MTR_LR':
            target_hdg = base_hdg
            spd_cmd = 3
        elif node == 'LR_TR':
            # 旧版里该段主要“稳定照射+保持直飞”，避免过度偏航
            target_hdg = (base_hdg + (side_sign * 8.0 if is_shooter else 0.0)) % 360
            spd_cmd = 3
        elif node in ('TR_DOR', 'DOR_DR'):
            # 旧版 ShortSkate 脱离效果：射手外摆更大，支援机较小
            target_hdg = (base_hdg + side_sign * (70.0 if is_shooter else 42.0)) % 360
            alt_cmd = 7
            spd_cmd = 5 if is_shooter else 4
        elif node == 'DR_MAR':
            # 二次进攻段：按 MTR'→LR'→TR' 子节点分段机动
            second_node = self._get_second_attack_subnode(distance, getattr(self, '_locked_compression', None))
            if second_node == 'MTRP_ENTRY':
                target_hdg = (base_hdg + (0.0 if is_shooter else -side_sign * 28.0)) % 360
                spd_cmd = 4 if is_shooter else 3
            elif second_node == 'MTRP_LRP':
                target_hdg = base_hdg
                spd_cmd = 4
            elif second_node == 'LRP_TRP':
                target_hdg = (base_hdg + side_sign * (40.0 if is_shooter else 22.0)) % 360
                alt_cmd = 7
                spd_cmd = 5 if is_shooter else 4
            else:  # TRP_MAR
                target_hdg = (base_hdg + 180.0 + side_sign * (20.0 if is_shooter else 8.0)) % 360
                alt_cmd = 7
                spd_cmd = 5
        else:  # BELOW_MAR
            target_hdg = (base_hdg + 180.0 + side_sign * 30.0) % 360
            alt_cmd = 7
            spd_cmd = 6

        marker = f"drag_shoot/{node}"
        last_node = self._last_template_node_by_agent.get(agent_id)
        if last_node != marker:
            self._last_template_node_by_agent[agent_id] = marker
            role = '射手' if is_shooter else '支援'
            log.info(f"[模板机动] drag_shoot {agent_id}({role}) 节点:{node} 距离:{distance:.1f}km")

        diff = target_hdg - current_hdg
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        hdg_cmd = 8 if abs(diff) < 4 else int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        return alt_cmd, hdg_cmd, spd_cmd
    
    def _execute_pincer_attack(self, env, agent_id: str) -> Tuple[int, int, int]:
        """钳形攻势：沿用旧版节点逻辑（先展开，再回正，再脱离）。"""
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3
        
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)
        
        assign = self._tactic_assignments_by_agent.get(agent_id) if hasattr(self, '_tactic_assignments_by_agent') else None
        target_threat = None
        target_id = getattr(assign, 'target_id', None)
        if target_id:
            for threat in self._get_picture_threats(purpose="decision"):
                if threat.track_id == target_id:
                    target_threat = threat
                    break
        if target_threat is None:
            target_threat = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
        if not target_threat:
            return self._get_intercept_action(env, agent_id)

        dx = target_threat.x - x
        dy = target_threat.y - y
        distance = float(np.sqrt(dx * dx + dy * dy))
        base_hdg = np.degrees(np.arctan2(dx, dy)) % 360

        side_sign = -1.0 if agent_id in ('A0100', 'A0200') else 1.0
        node = DEFAULT_RANGES.get_current_node(distance, getattr(self, '_locked_compression', None))
        node = self._lock_template_node(agent_id, target_threat.track_id, node)

        # 动态偏转幅度：仍以目标方位为基准，但偏转大小随距离分层变化，
        # 避免固定角度在不同几何关系下过度/不足。
        far_ratio = float(np.clip((distance - DEFAULT_RANGES.MTR) / max(DEFAULT_RANGES.NLT - DEFAULT_RANGES.MTR, 1e-6), 0.0, 1.0))
        near_ratio = float(np.clip((DEFAULT_RANGES.DOR - distance) / max(DEFAULT_RANGES.DOR - DEFAULT_RANGES.TR, 1e-6), 0.0, 1.0))
        expand_offset = 24.0 + 16.0 * far_ratio      # 24~40°
        egress_offset = 48.0 + 18.0 * near_ratio     # 48~66°
        recompress_offset = 24.0 + 14.0 * near_ratio # 24~38°

        alt_cmd = 7
        spd_cmd = 3
        if node in ('BEYOND_NLT', 'NLT_MELD'):
            target_hdg = (base_hdg - side_sign * expand_offset) % 360
            spd_cmd = 4
        elif node == 'MELD_MTR':
            # 旧版MELD前段继续展开、后段回正
            target_hdg = (base_hdg - side_sign * expand_offset) % 360 if distance > 90.0 else base_hdg
            spd_cmd = 4 if distance > 90.0 else 3
        elif node in ('MTR_LR', 'LR_TR'):
            target_hdg = base_hdg
        elif node in ('TR_DOR', 'DOR_DR'):
            target_hdg = (base_hdg + side_sign * egress_offset) % 360
            alt_cmd = 7
            spd_cmd = 5
        elif node == 'DR_MAR':
            target_hdg = (base_hdg + side_sign * recompress_offset) % 360
            spd_cmd = 4
        else:
            target_hdg = (base_hdg + 180.0 + side_sign * 20.0) % 360
            alt_cmd = 7
            spd_cmd = 6

        marker = f"pincer/{node}"
        if self._last_template_node_by_agent.get(agent_id) != marker:
            self._last_template_node_by_agent[agent_id] = marker
            log.info(f"[模板机动] pincer {agent_id} 节点:{node} 距离:{distance:.1f}km")
        
        diff = target_hdg - current_hdg
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        
        if abs(diff) < 5:
            return alt_cmd, 8, spd_cmd
        
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        return alt_cmd, hdg_cmd, spd_cmd
    
    def _execute_front_back(self, env, agent_id: str) -> Tuple[int, int, int]:
        """前后攻击：长机直压、僚机保持后置，节点化脱离。"""
        is_lead = self.formation.is_lead(agent_id)
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)

        assign = self._tactic_assignments_by_agent.get(agent_id) if hasattr(self, '_tactic_assignments_by_agent') else None
        target_threat = None
        target_id = getattr(assign, 'target_id', None)
        if target_id:
            for threat in self._get_picture_threats(purpose="decision"):
                if threat.track_id == target_id:
                    target_threat = threat
                    break
        if target_threat is None:
            target_threat = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
        if target_threat is None:
            return self._get_intercept_action(env, agent_id)

        dx = target_threat.x - x
        dy = target_threat.y - y
        distance = float(np.sqrt(dx * dx + dy * dy))
        base_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        node = DEFAULT_RANGES.get_current_node(distance, getattr(self, '_locked_compression', None))
        node = self._lock_template_node(agent_id, target_threat.track_id, node)
        alt_cmd, spd_cmd = 7, 3
        target_hdg = base_hdg
        
        if is_lead:
            if node in ('TR_DOR', 'DOR_DR'):
                target_hdg = (base_hdg + 55.0) % 360
                alt_cmd, spd_cmd = 6, 5
            elif node in ('DR_MAR', 'BELOW_MAR'):
                target_hdg = (base_hdg + 180.0) % 360
                alt_cmd, spd_cmd = 6, 5
        else:
            ac = env.agents.get(agent_id)
            if not ac or not ac.is_alive:
                return 7, 8, 3
            
            # 找到对应长机
            lead_id = 'A0100' if agent_id == 'A0200' else 'A0300'
            lead_ac = env.agents.get(lead_id)
            if not lead_ac or not lead_ac.is_alive:
                return self._get_intercept_action(env, agent_id)
            
            # 跟随长机但保持后方位置
            lead_x, lead_y = self._get_battlefield_pos(env, lead_id)
            my_x, my_y = self._get_battlefield_pos(env, agent_id)
            current_hdg = self._get_heading(env, agent_id)
            
            # 僚机保持后方10km（旧版一字队形思想）
            target_x = lead_x
            target_y = lead_y - 10  # 后方10km
            
            dx = target_x - my_x
            dy = target_y - my_y
            target_hdg = np.degrees(np.arctan2(dx, dy)) % 360
            
            diff = target_hdg - current_hdg
            if diff > 180: diff -= 360
            elif diff < -180: diff += 360
            
            if node in ('TR_DOR', 'DOR_DR'):
                # 后机与长机反向脱离，形成前后分离
                target_hdg = (base_hdg - 55.0) % 360
                alt_cmd, spd_cmd = 7, 4
            if node in ('DR_MAR', 'BELOW_MAR'):
                target_hdg = (base_hdg + 170.0) % 360
                alt_cmd, spd_cmd = 6, 5

        marker = f"front_back/{node}"
        if self._last_template_node_by_agent.get(agent_id) != marker:
            self._last_template_node_by_agent[agent_id] = marker
            role = '前机' if is_lead else '后机'
            log.info(f"[模板机动] front_back {agent_id}({role}) 节点:{node} 距离:{distance:.1f}km")

        diff = target_hdg - current_hdg
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        hdg_cmd = 8 if abs(diff) < 5 else int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        return alt_cmd, hdg_cmd, spd_cmd

    def _get_direct_engage_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        """交战期直接追踪目标（不触发协同探测7层流程）。"""
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3

        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)

        target_threat = None
        assignment = self.coop_engagement.get_assignment(agent_id) if hasattr(self, 'coop_engagement') else None
        if assignment:
            for threat in self._get_picture_threats(purpose="decision"):
                if threat.track_id == assignment.target_id:
                    target_threat = threat
                    break
        if target_threat is None:
            target_threat = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
        if target_threat is None:
            return 7, 8, 3

        target_hdg = np.degrees(np.arctan2(target_threat.x - x, target_threat.y - y)) % 360
        diff = target_hdg - current_hdg
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        if abs(diff) < 5:
            return 7, 8, 3

        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        return 7, hdg_cmd, 4

    def _execute_high_low_attack(self, env, agent_id: str) -> Tuple[int, int, int]:
        """上下夹击：沿用旧版“长机平飞、僚机爬升高位”的节点策略。"""
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)
        assign = self._tactic_assignments_by_agent.get(agent_id) if hasattr(self, '_tactic_assignments_by_agent') else None
        target_threat = None
        target_id = getattr(assign, 'target_id', None)
        if target_id:
            for threat in self._get_picture_threats(purpose="decision"):
                if threat.track_id == target_id:
                    target_threat = threat
                    break
        if target_threat is None:
            target_threat = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
        if not target_threat:
            return self._get_intercept_action(env, agent_id)

        dx = target_threat.x - x
        dy = target_threat.y - y
        distance = float(np.sqrt(dx * dx + dy * dy))
        node = DEFAULT_RANGES.get_current_node(distance, getattr(self, '_locked_compression', None))
        node = self._lock_template_node(agent_id, target_threat.track_id, node)
        target_hdg = np.degrees(np.arctan2(dx, dy)) % 360
        diff = target_hdg - current_hdg
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        hdg_cmd = 8 if abs(diff) < 5 else int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        is_lead = self.formation.is_lead(agent_id)
        # 旧版：长机不主动下降，僚机在前段持续爬升，后段保持高位
        if is_lead:
            alt_cmd = 7 if node not in ('TR_DOR', 'DOR_DR', 'DR_MAR', 'BELOW_MAR') else 6
            spd_cmd = 3
        else:
            alt_cmd = 10 if node in ('BEYOND_NLT', 'NLT_MELD', 'MELD_MTR') else 8
            spd_cmd = 4 if node in ('BEYOND_NLT', 'NLT_MELD', 'MELD_MTR') else 3

        marker = f"high_low/{node}"
        if self._last_template_node_by_agent.get(agent_id) != marker:
            self._last_template_node_by_agent[agent_id] = marker
            role = '长机平飞' if is_lead else '僚机高位'
            log.info(f"[模板机动] high_low {agent_id}({role}) 节点:{node} 距离:{distance:.1f}km")
        return alt_cmd, hdg_cmd, spd_cmd

    def _execute_side_by_side(self, env, agent_id: str) -> Tuple[int, int, int]:
        """并列射击：双机并排推进，TR后对称脱离。"""
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)
        assign = self._tactic_assignments_by_agent.get(agent_id) if hasattr(self, '_tactic_assignments_by_agent') else None
        target_threat = None
        target_id = getattr(assign, 'target_id', None)
        if target_id:
            for threat in self._get_picture_threats(purpose="decision"):
                if threat.track_id == target_id:
                    target_threat = threat
                    break
        if target_threat is None:
            target_threat = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
        if not target_threat:
            return self._get_intercept_action(env, agent_id)

        base_hdg = np.degrees(np.arctan2(target_threat.x - x, target_threat.y - y)) % 360
        distance = float(np.sqrt((target_threat.x - x) ** 2 + (target_threat.y - y) ** 2))
        node = DEFAULT_RANGES.get_current_node(distance, getattr(self, '_locked_compression', None))
        node = self._lock_template_node(agent_id, target_threat.track_id, node)
        is_lead = self.formation.is_lead(agent_id)
        side_sign = -1.0 if agent_id in ('A0100', 'A0200') else 1.0

        if node in ('TR_DOR', 'DOR_DR'):
            target_hdg = (base_hdg + side_sign * 50.0) % 360
            alt_cmd, spd_cmd = 6, 5
        elif node in ('DR_MAR', 'BELOW_MAR'):
            target_hdg = (base_hdg + 180.0 + side_sign * 15.0) % 360
            alt_cmd, spd_cmd = 6, 5
        else:
            target_hdg = (base_hdg + (-8 if is_lead else 8)) % 360
            alt_cmd, spd_cmd = 7, 3

        diff = target_hdg - current_hdg
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        hdg_cmd = 8 if abs(diff) < 5 else int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        marker = f"side_by_side/{node}"
        if self._last_template_node_by_agent.get(agent_id) != marker:
            self._last_template_node_by_agent[agent_id] = marker
            log.info(f"[模板机动] side_by_side {agent_id} 节点:{node} 距离:{distance:.1f}km")
        return alt_cmd, hdg_cmd, spd_cmd
    
    def _execute_tactical_turn(self, env, agent_id: str) -> Tuple[int, int, int]:
        """战术转弯：快速转向脱离"""
        if agent_id.startswith('A'):
            alive_enemy_exists = any(
                str(other_id).startswith('B') and getattr(other_aircraft, 'is_alive', False)
                for other_id, other_aircraft in getattr(env, 'agents', {}).items()
            )
            if alive_enemy_exists:
                x, y = self._get_battlefield_pos(env, agent_id)
                threat = self._get_nearest_visible_pair_threat(agent_id, x, y)
                if threat is None:
                    threat = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
                if threat is not None:
                    target_hdg = np.degrees(np.arctan2(threat.x - x, threat.y - y)) % 360
                    if self.step_count % 25 == 0:
                        log.warning(f"🛑 [CAP TACTICAL_TURN拦截] {agent_id} 敌机仍存活，改为受限前出接敌 heading={target_hdg:.1f}°")
                    return self.build_return_command_with_alt_guard(env, agent_id, target_hdg)

        current_hdg = self._get_heading(env, agent_id)
        # 转向180°（掉头）
        target_hdg = (current_hdg + 180) % 360
        
        diff = target_hdg - current_hdg
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        
        if abs(diff) < 10:
            return 7, 8, 3
        
        # 使用最大转向率
        hdg_cmd = 0 if diff < 0 else 16  # -π 或 +π
        # 掉头硬转弯阶段给轻微下降意图，避免低层模型在大坡度时抬头爬升。
        return 7, hdg_cmd, 3
    
    def _get_evade_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        """规避动作：支持三种规避机动
        
        - 三九Beam机动：垂直于威胁方向
        - Notch机动：利用地杂波
        - 俯冲脱离：快速下降脱离
        """
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3
        
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)
        
        # 获取最近威胁
        nearest = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
        if not nearest:
            return 7, 8, 3
        
        distance = np.sqrt((nearest.x - x)**2 + (nearest.y - y)**2)
        
        # 根据距离选择规避方式
        if distance < DEFAULT_RANGES.MAR:
            # 极近距：俯冲脱离
            return self._execute_dive_escape(env, agent_id)
        elif distance < DEFAULT_RANGES.DR:
            # 近距：Notch机动
            return self._execute_notch_maneuver(env, agent_id, nearest)
        else:
            # 中距：Beam机动
            return self._execute_beam_maneuver(env, agent_id, nearest)
    
    def _execute_beam_maneuver(self, env, agent_id: str, threat) -> Tuple[int, int, int]:
        """三九Beam机动：垂直于威胁方向"""
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)
        
        # 计算威胁方向
        dx = threat.x - x
        dy = threat.y - y
        threat_bearing = np.degrees(np.arctan2(dx, dy)) % 360
        
        # Beam机动：垂直于威胁方向（±90°）
        beam_left = (threat_bearing - 90) % 360
        beam_right = (threat_bearing + 90) % 360
        
        # 选择转向角度较小的
        diff_left = beam_left - current_hdg
        if diff_left > 180: diff_left -= 360
        elif diff_left < -180: diff_left += 360
        
        diff_right = beam_right - current_hdg
        if diff_right > 180: diff_right -= 360
        elif diff_right < -180: diff_right += 360
        
        diff = diff_left if abs(diff_left) < abs(diff_right) else diff_right
        
        if abs(diff) < 10:
            return 7, 8, 3
        
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        return 7, hdg_cmd, 3
    
    def _execute_notch_maneuver(self, env, agent_id: str, threat) -> Tuple[int, int, int]:
        """Notch机动：利用地杂波，保持在威胁雷达的多普勒盲区"""
        x, y = self._get_battlefield_pos(env, agent_id)
        current_hdg = self._get_heading(env, agent_id)
        
        # 计算威胁方向
        dx = threat.x - x
        dy = threat.y - y
        threat_bearing = np.degrees(np.arctan2(dx, dy)) % 360
        
        # Notch：保持在威胁的侧方（约85-95°）
        notch_angle = 87  # 略小于90°以保持在多普勒盲区
        notch_left = (threat_bearing - notch_angle) % 360
        notch_right = (threat_bearing + notch_angle) % 360
        
        # 选择转向角度较小的
        diff_left = notch_left - current_hdg
        if diff_left > 180: diff_left -= 360
        elif diff_left < -180: diff_left += 360
        
        diff_right = notch_right - current_hdg
        if diff_right > 180: diff_right -= 360
        elif diff_right < -180: diff_right += 360
        
        diff = diff_left if abs(diff_left) < abs(diff_right) else diff_right
        
        if abs(diff) < 10:
            return 7, 8, 3
        
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        # Notch时同时下降高度
        return 7, hdg_cmd, 3
    
    def _execute_dive_escape(self, env, agent_id: str) -> Tuple[int, int, int]:
        """俯冲脱离：快速下降并转向脱离"""
        current_hdg = self._get_heading(env, agent_id)
        
        # 转向180°（掉头）
        target_hdg = (current_hdg + 180) % 360
        
        diff = target_hdg - current_hdg
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        
        # 使用最大转向率和最大下降率
        hdg_cmd = 0 if diff < 0 else 16  # -π 或 +π
        return 7, hdg_cmd, 3
    
    def _get_rtb_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        """返航动作：向南飞行"""
        current_hdg = self._get_heading(env, agent_id)
        target_hdg = 180.0  # 向南
        
        diff = target_hdg - current_hdg
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        
        if abs(diff) < 5:
            if agent_id == 'B0100' and os.environ.get('CAP_B0100_DEEP_TRACE', '1').strip().lower() in ('1', 'true', 'yes', 'on'):
                try:
                    ac = env.agents[agent_id]
                    log.warning(
                        "[B0100][T+%07.1fs][_get_rtb_action][before_return][decision] current_hdg=%.3f target_hdg=%.3f diff=%.3f action=(7,8,3) alt=%.3fm vc=%.3fmps pos=%s",
                        float(self.step_count * 0.2),
                        current_hdg,
                        target_hdg,
                        diff,
                        float(ac.get_property_value(c.position_h_sl_m)),
                        float(ac.get_property_value(c.velocities_vc_mps)),
                        np.array2string(np.asarray(ac.get_position()), precision=6, separator=",", threshold=np.inf, max_line_width=100000),
                    )
                except Exception as exc:
                    log.warning("[B0100][T+%07.1fs][_get_rtb_action][before_return][decision] failed=%s", float(self.step_count * 0.2), exc)
            return 7, 8, 3
        
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))
        if agent_id == 'B0100' and os.environ.get('CAP_B0100_DEEP_TRACE', '1').strip().lower() in ('1', 'true', 'yes', 'on'):
            try:
                ac = env.agents[agent_id]
                log.warning(
                    "[B0100][T+%07.1fs][_get_rtb_action][before_return][decision] current_hdg=%.3f target_hdg=%.3f diff=%.3f hdg_cmd=%d action=(7,%d,3) alt=%.3fm vc=%.3fmps pos=%s",
                    float(self.step_count * 0.2),
                    current_hdg,
                    target_hdg,
                    diff,
                    hdg_cmd,
                    hdg_cmd,
                    float(ac.get_property_value(c.position_h_sl_m)),
                    float(ac.get_property_value(c.velocities_vc_mps)),
                    np.array2string(np.asarray(ac.get_position()), precision=6, separator=",", threshold=np.inf, max_line_width=100000),
                )
            except Exception as exc:
                log.warning("[B0100][T+%07.1fs][_get_rtb_action][before_return][decision] failed=%s", float(self.step_count * 0.2), exc)
        return 7, hdg_cmd, 3

    def _get_enemy_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        """敌方动作：使用EnemyAIAdapter提供智能行为
        
        敌方活动范围限制在275km x 400km内
        """
        ac = env.agents.get(agent_id)
        if not ac or not ac.is_alive:
            return 7, 8, 3
        
        current_time = self.step_count * 0.2
        
        # 计算到最近我方的距离
        my_pos = self._get_battlefield_pos(env, agent_id)
        min_dist = float('inf')
        for aid in env.agents:
            if aid.startswith('A') and env.agents[aid].is_alive:
                pos = self._get_battlefield_pos(env, aid)
                dist = np.sqrt((my_pos[0]-pos[0])**2 + (my_pos[1]-pos[1])**2)
                min_dist = min(min_dist, dist)

        # 敌方桥接我方 CAPTask 逻辑已整体注释保留，默认不再进入该分支：
        # use_enemy_bridge = os.getenv("CAP_ENEMY_FRIENDLY_BRIDGE_ENABLED", "1").strip().lower() in (
        #     "1", "true", "yes", "on"
        # )
        # if use_enemy_bridge and hasattr(self, 'tactical_bridge') and self.tactical_bridge:
        #     pair_id = 'enemy_group_1' if agent_id in ('B0100', 'B0200') else 'enemy_group_2'
        #     bridge_takeover_distance_km = float(os.getenv('CAP_ENEMY_BRIDGE_TAKEOVER_DISTANCE_KM', '180'))
        #     bridge_takeover_time_s = float(os.getenv('CAP_ENEMY_BRIDGE_TAKEOVER_TIME_S', '90'))
        #     if (
        #         pair_id in self._enemy_bridge_takeover_pairs
        #         or (current_time >= bridge_takeover_time_s and min_dist <= bridge_takeover_distance_km)
        #     ):
        #         self._enemy_bridge_takeover_pairs.add(pair_id)
        #         try:
        #             bridge_action = self.tactical_bridge.get_enemy_action(env, agent_id, self)
        #             if bridge_action and len(bridge_action) >= 3:
        #                 return bridge_action
        #         except Exception as exc:
        #             log.warning(f"[EnemyBridge] fallback to EnemyAIAdapter for {agent_id}: {exc}")

        cmd = self.enemy_adapter.get_enemy_action(env, agent_id, current_time, min_dist, task=self)
        if agent_id == 'B0100' and os.environ.get('CAP_B0100_DEEP_TRACE', '1').strip().lower() in ('1', 'true', 'yes', 'on'):
            try:
                log.warning(
                    "[B0100][T+%07.1fs][_get_enemy_action][output][decision] cmd=%s min_dist=%.3f current_time=%.1f cap_state=%s",
                    float(current_time),
                    tuple(int(x) for x in cmd),
                    float(min_dist),
                    float(current_time),
                    getattr(self.cap_state_machine, 'state', 'UNKNOWN'),
                )
            except Exception as exc:
                log.warning("[B0100][T+%07.1fs][_get_enemy_action][output][decision] failed=%s", float(current_time), exc)
        return cmd

    def _get_patrol_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        """我方巡逻动作：严格参考 SimplePatrolTask（避免转圈）。"""
        ac = env.agents.get(agent_id)
        if ac is None or not ac.is_alive:
            return 7, 8, 3

        # NED: (north, east, down) -> (x=east_km, y=north_km)
        pos = ac.get_position()
        x_km = float(pos[1]) / 1000.0
        y_km = float(pos[0]) / 1000.0

        heading_deg = float(np.degrees(ac.get_property_values([c.attitude_heading_true_rad])[0])) % 360.0

        pm = self.patrol_machines.get(agent_id)
        if pm is None:
            return 7, 8, 3

        old_state = pm.state
        patrol_state, target_heading = pm.update(x_km, y_km)
        target_heading = float(target_heading) % 360.0

        if patrol_state != old_state:
            log.info(
                "[状态切换] [%s] %s -> %s @ (%.1f, %.1f)",
                agent_id,
                old_state.value,
                patrol_state.value,
                x_km,
                y_km,
            )

        # 初始化/读取动作层状态
        if agent_id not in self.aircraft_states:
            self.aircraft_states[agent_id] = {
                'target_heading': target_heading,
                'force_left_turn': False,
                'patrol_state': str(patrol_state),
            }

        st = self.aircraft_states[agent_id]
        prev_target_heading = float(st.get('target_heading', target_heading)) % 360.0
        prev_patrol_state = st.get('patrol_state', 'UNKNOWN')
        st['patrol_state'] = str(patrol_state)
        st['target_heading'] = target_heading

        # 仅在“真正触发转向点”发生时，开启强制左转（只针对 0<->180 翻转）
        if target_heading != prev_target_heading or st.get('patrol_state') != prev_patrol_state:
            delta = abs(target_heading - prev_target_heading)
            delta = min(delta, 360.0 - delta)
            st['force_left_turn'] = (abs(delta - 180.0) < 1e-6)

        # shortest_diff 用于接近判定
        shortest_diff = target_heading - heading_deg
        if shortest_diff > 180:
            shortest_diff -= 360
        elif shortest_diff < -180:
            shortest_diff += 360

        heading_diff = shortest_diff

        # 在 180 翻转的转向事件期间强制左转（避免偶发右转 -> 转圈）
        if st.get('force_left_turn', False):
            if abs(shortest_diff) < 5.0:
                st['force_left_turn'] = False
            else:
                left_amount = (heading_deg - target_heading) % 360.0
                forced_left = -left_amount
                if forced_left < -180.0:
                    forced_left = -180.0
                if forced_left > 0.0:
                    forced_left = -abs(forced_left)
                heading_diff = forced_left

        # 将航向差映射到命令索引（对齐 SimplePatrolTask 逻辑）
        if agent_id in ['A0200', 'A0400'] and abs(heading_diff) >= 175.0:
            hdg_cmd = 0  # -pi
        else:
            heading_diff_rad = np.radians(heading_diff)
            distances = np.abs(self.norm_hdg - heading_diff_rad)
            hdg_cmd = int(np.argmin(distances))
            if abs(shortest_diff) < 5.0:
                hdg_cmd = 8

        return 7, hdg_cmd, 3

    def _speed_error_to_spd_cmd(self, speed_error_mps: float) -> int:
        return _caplow._speed_error_to_spd_cmd(self, speed_error_mps)

    def _select_lowlevel_model_for_agent(self, agent_id: str):
        return _caplow._select_lowlevel_model_for_agent(self, agent_id)

    def _lowlevel_control(self, env, agent_id: str, alt: int, hdg: int, vel: int) -> np.ndarray:
        """Delegate to extracted helper to keep CAPTask focused on orchestration."""
        return _caprfh._lowlevel_control(self, env, agent_id, alt, hdg, vel)

