"""CAPTask lifecycle helpers for initialization."""

import logging
import os

import torch

try:
    from .cap_state_machine import CAPStateMachine, CAPState
    from .control_ranges import DEFAULT_RANGES
    from .picture import Picture, MockAwacsDataSource, TrackFusion
    from .tactics import CooperativeDetection, CooperativeEngagement
    from .tactics import FormationGuidance, VelocityCoordination
    from .cap_radar import CAPRadarManager
    from .mission_evaluator import MissionEvaluator
    from .tactic_selector import TacticSelector, TacticAssignment
    from .patrol_state_machine import PatrolState
    from .adapters import EnemyAIAdapter, MissileAdapter, IntentAdapter, TacticExecutorAdapter
    from .enemy_safe_teacher import EnemySafeTeacher
    from .tactics.tactical_bridge import TacticalBridge
except ImportError:
    from cap.cap_state_machine import CAPStateMachine, CAPState
    from cap.control_ranges import DEFAULT_RANGES
    from cap.picture import Picture, MockAwacsDataSource, TrackFusion
    from cap.tactics import CooperativeDetection, CooperativeEngagement
    from cap.tactics import FormationGuidance, VelocityCoordination
    from cap.cap_radar import CAPRadarManager
    from cap.mission_evaluator import MissionEvaluator
    from cap.tactic_selector import TacticSelector, TacticAssignment
    from cap.patrol_state_machine import PatrolState
    from cap.adapters import EnemyAIAdapter, MissileAdapter, IntentAdapter, TacticExecutorAdapter
    from cap.enemy_safe_teacher import EnemySafeTeacher
    from cap.tactics.tactical_bridge import TacticalBridge

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

log = logging.getLogger(__name__)


def initialize_cap_task(self):
    self.step_count = 0
    self._inner_rnn_states = {}
    self._friendly_recovery_state = {}
    # self._enemy_bridge_takeover_pairs = set()

    self._safety_diag_last_print_step = {}
    self._safety_diag_last_level = {}

    self.aircraft_states = {}

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    self._last_patrol_state = {}

    self._load_config_params()

    self._init_coordinate_system()
    self._init_faor()
    self._init_formation()
    self._init_formation()
    self._init_patrol_machines()

    self.coop_detection = CooperativeDetection()
    self.coop_engagement = CooperativeEngagement()

    self.mission_evaluator = MissionEvaluator()
    self.tactic_selector = TacticSelector()

    self._load_baseline_models()
    self._init_action_arrays()
    self._native_residual_target_state = {}

    self.cap_state_machine = CAPStateMachine(DEFAULT_RANGES)
    self.picture = Picture(faor_length=300.0)
    self.awacs = MockAwacsDataSource(seed=getattr(self, 'awacs_seed', None))
    self.track_fusion = TrackFusion()
    self.mission_start_time = 0.0

    self.coop_detection = CooperativeDetection(faor_width=200.0, faor_length=300.0)

    self.formation_guidance = FormationGuidance(formation_pairs=[
        ('A0100', 'A0200'),
        ('A0300', 'A0400'),
    ])
    self.velocity_coordination = VelocityCoordination(formation_pairs=[
        ('A0100', 'A0200'),
        ('A0300', 'A0400'),
    ])

    self.cap_radar = CAPRadarManager()

    if ENEMY_AI_AVAILABLE:
        self.enemy_ai = UnifiedEnemyTacticalAI()
        log.info("   [OK] Enemy Tactical AI Integrated")
    else:
        self.enemy_ai = None
        log.warning("   [WARN] Enemy AI not found, using simple behavior")

    if MISSILE_MANAGER_AVAILABLE:
        self.state_manager = TacticalStateManager()
        self.missile_manager = MissileManager(self.state_manager)
        log.info("   [OK] Missile Manager Integrated")
    else:
        self.state_manager = None
        self.missile_manager = None
        log.warning("   [WARN] Missile Manager not found, functionality limited")

    self.enemy_adapter = EnemyAIAdapter()
    self.enemy_rule_lowlevel = None
    self.enemy_rule_lowlevel_enabled = False
    self.enemy_sim_recreate_enabled = os.getenv("CAP_ENEMY_SIM_RECREATE_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    try:
        self.enemy_sim_recreate_max_count = max(0, int(os.getenv("CAP_ENEMY_SIM_RECREATE_MAX_COUNT", "8")))
    except Exception:
        self.enemy_sim_recreate_max_count = 8
    self._enemy_sim_recreate_state = {}
    self.friendly_sim_recreate_enabled = os.getenv("CAP_FRIENDLY_SIM_RECREATE_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    try:
        self.friendly_sim_recreate_max_count = max(0, int(os.getenv("CAP_FRIENDLY_SIM_RECREATE_MAX_COUNT", "8")))
    except Exception:
        self.friendly_sim_recreate_max_count = 8
    self._friendly_sim_recreate_state = {}
    self.enemy_safe_teacher = EnemySafeTeacher()
    safe_teacher_requested = os.getenv("CAP_ENEMY_SAFE_TEACHER_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    self.enemy_safe_teacher_enabled = bool(safe_teacher_requested and not self.enemy_rule_lowlevel_enabled)
    self._enemy_safe_teacher_stats = {}
    self._enemy_safe_teacher_skip_agents = set()
    self._enemy_lowlevel_action_overrides = {}
    self.missile_adapter = MissileAdapter(DEFAULT_RANGES)
    self.intent_adapter = IntentAdapter()
    self.tactic_executor = TacticExecutorAdapter(DEFAULT_RANGES)
    self.tactical_bridge = TacticalBridge()
    self.tactical_bridge.initialize_tactical_tasks(self.config, None, cap_task_instance=self)

    self.missile_adapter.init_inventory(['A0100', 'A0200', 'A0300', 'A0400'], missiles_per_agent=4)

    self._cooperative_tracking_active = False
    self._tracking_targets = {}
    self._tactic_assignments_by_agent = {}
    self._last_tactic_assignments_by_pair = {}

    self._stable_tracking_window_s = 8.0
    self._stable_tracking_min_quality = 0.45
    self._stable_tracking_grace_s = 2.5
    self._stable_tracking_elapsed = {}
    self._stable_tracking_last_ok = {}
    self._stable_tracking_trackers = {}
    self._stable_tracking_ready_announced = set()
    self._defensive_engagement_suppressed_until = {}

    self._initial_nodes_logged = False
    self._last_template_node_by_agent = {}
    self._second_attack_window_announced = set()

    self._pair_target_memory = {}
    self._pair_target_lock_until_step = {}
    self._agent_template_target = {}
    self._agent_template_node_progress = {}
    self._pair_template_target = {}
    self._pair_template_node_progress = {}

    self._relay_guidance_active = False
    self._relayed_missiles = set()
    self._relay_last_time = {}

    self._guidance_verify = {
        'prelaunch_gate_pass_count': 0,
        'prelaunch_gate_block_count': 0,
        'prelaunch_gate_pass_by_target': {},
        'prelaunch_gate_block_by_target': {},
        'prelaunch_gate_first_pass_time_by_target': {},
        'relay_attempt_count': 0,
        'relay_success_count': 0,
        'relay_success_by_reason': {},
        'relay_fail_count': 0,
        'active_guided_missile_samples': 0,
        'active_guided_missile_peak': 0,
    }
    self._external_gate_block_last_time = {}
    self._last_awacs_data_log_time = -9999.0
    self._last_awacs_data_log_signature = None
    self._last_awacs_empty_log_time = -9999.0
    self._last_awacs_empty_log_signature = None
    self._last_awacs_context_log_time = -9999.0
    self._last_awacs_context_log_signature = None
    self._last_coop_detect_log_time = -9999.0
    self._last_coop_detect_signature = None
    self._last_coop_detect_story_time = -9999.0
    self._last_coop_detect_story_signature = None
    self._last_detect_search_log_time = -9999.0
    self._last_detect_search_log_signature = None
    self._last_coop_track_log_time = -9999.0
    self._last_coop_track_signature = None
    self._relay_event_last_time = {}
    self._relay_event_last_key = {}
    self._track_event_last = {}
    self._target_first_awacs_time = {}
    self._target_first_fcr_time = {}
    self._target_first_ready_time = {}
    self._zone_transition_last = {}
    self._zone_transition_history = {}
    self._picture_decision_track_max_age_s = float(os.getenv("CAP_PICTURE_DECISION_TRACK_MAX_AGE_S", "6.0"))
    self._picture_decision_include_recent_lost = os.getenv(
        "CAP_PICTURE_DECISION_INCLUDE_RECENT_LOST",
        "1",
    ).strip().lower() in ("1", "true", "yes", "on")
    self._picture_decision_lost_grace_s = float(os.getenv("CAP_PICTURE_DECISION_LOST_GRACE_S", "2.5"))
    self._picture_search_track_max_age_s = float(os.getenv("CAP_PICTURE_SEARCH_TRACK_MAX_AGE_S", "20.0"))
    self._picture_search_lost_grace_s = float(os.getenv("CAP_PICTURE_SEARCH_LOST_GRACE_S", "20.0"))
    self._picture_awacs_partner_max_age_s = float(os.getenv("CAP_PICTURE_AWACS_PARTNER_MAX_AGE_S", "5.0"))
    self._picture_awacs_only_max_age_s = float(
        os.getenv(
            "CAP_PICTURE_AWACS_ONLY_MAX_AGE_S",
            str(max(self._picture_search_track_max_age_s, 20.0)),
        )
    )
    self._picture_radar_track_max_age_s = float(
        os.getenv("CAP_PICTURE_RADAR_TRACK_MAX_AGE_S", "1.0")
    )
    self._picture_fusion_fresh_window_s = float(os.getenv("CAP_PICTURE_FUSION_FRESH_WINDOW_S", "5.0"))
    self._picture_diag_distance_delta_km = float(os.getenv("CAP_PICTURE_DIAG_DISTANCE_DELTA_KM", "18.0"))
    self._last_picture_diag_log_time = -9999.0
    self._last_picture_diag_log_signature = None
    self._last_picture_freshness_log_time = -9999.0
    self._last_picture_freshness_log_signature = None
    self._last_tactic_diag_log_time = -9999.0
    self._last_tactic_diag_log_signature = None
    if hasattr(self, "track_fusion") and self.track_fusion is not None:
        self.track_fusion.awacs_partner_max_age_s = self._picture_awacs_partner_max_age_s
        self.track_fusion.awacs_only_max_age_s = self._picture_awacs_only_max_age_s
        self.track_fusion.fresh_track_window_s = self._picture_fusion_fresh_window_s

    self._initial_fuel_lbs = {}

    log.info("   [OK] Adapters Initialized")
    log.info("   [INFO] Enemy Rule Lowlevel: DISABLED")
    log.info("   [INFO] Friendly Simulator Recreate: %s", "ENABLED" if self.friendly_sim_recreate_enabled else "DISABLED")
    log.info("   [INFO] Enemy Simulator Recreate: %s", "ENABLED" if self.enemy_sim_recreate_enabled else "DISABLED")
    log.info(
        "   [INFO] Enemy Wave Controller: %s | Direct RTB Hack: %s",
        "ENABLED" if os.getenv("ENEMY_DISABLE_WAVE_MODE", "1") == "0" else "DISABLED",
        "ENABLED" if os.getenv("CAP_ENEMY_DIRECT_RTB_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on") else "DISABLED",
    )
    log.info("   [INFO] Enemy Safe Teacher: %s", "ENABLED" if self.enemy_safe_teacher_enabled else "DISABLED")
    log.info(
        "   [INFO] Picture Filters: decision(age<=%.1fs,lost<=%.1fs,recent=%s) | search(age<=%.1fs,lost<=%.1fs)",
        self._picture_decision_track_max_age_s,
        self._picture_decision_lost_grace_s,
        "Y" if self._picture_decision_include_recent_lost else "N",
        self._picture_search_track_max_age_s,
        self._picture_search_lost_grace_s,
    )
    log.info(
        "   [INFO] Picture Fusion: awacs_partner<=%.1fs | awacs_only<=%.1fs | fresh_window<=%.1fs",
        self._picture_awacs_partner_max_age_s,
        self._picture_awacs_only_max_age_s,
        self._picture_fusion_fresh_window_s,
    )
    log.info("=" * 50)
    log.info("=" * 50)
    log.info("[INIT] CAPTask Init Complete - Racing Patrol v1.0")
    log.info(f"   坐标系: A0100航向={self.a0100_heading}°")
    log.info(f"   编队: 僚机Y偏移={self.wingman_y}km")
    log.info(f"   FAOR: {self.faor.width}x{self.faor.length}km")
    try:
        bx, by = self.faor.get_bullseye()
        log.info(f"   靶眼(bullseye): ({bx:.1f}, {by:.1f})")
    except Exception:
        if hasattr(self.faor, 'bullseye'):
            bx, by = self.faor.bullseye
            log.info(f"   靶眼(bullseye): ({bx:.1f}, {by:.1f})")
    log.info("=" * 50)
