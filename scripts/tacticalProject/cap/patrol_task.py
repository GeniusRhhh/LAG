"""
基础巡逻任务 - 完全独立的实现
只实现基础的跑马道巡逻功能，与之前的战术系统完全剥离
"""
import logging
import os
import json
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from gymnasium import spaces
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
try:
    from ..action_codec import (
        NORM_ALTITUDE_3,
        NORM_HEADING_5,
        NORM_VELOCITY_3,
        ALT_DESCEND,
        ALT_HOLD,
        ALT_CLIMB,
        HDG_HOLD,
        VEL_DECEL,
        VEL_HOLD,
        VEL_ACCEL,
        heading_cmd_from_delta_deg,
        heading_cmd_from_delta_rad,
    )
except ImportError:
    from action_codec import (
        NORM_ALTITUDE_3,
        NORM_HEADING_5,
        NORM_VELOCITY_3,
        ALT_DESCEND,
        ALT_HOLD,
        ALT_CLIMB,
        HDG_HOLD,
        VEL_DECEL,
        VEL_HOLD,
        VEL_ACCEL,
        heading_cmd_from_delta_deg,
        heading_cmd_from_delta_rad,
    )

# 巡逻状态机（提供“热/冷段 + 横向过渡段(东/西)”能力）
try:
    # 当以包形式导入（cap.patrol_task）
    from .patrol_state_machine import PatrolStateMachine, PatrolBox, PatrolState
except Exception:
    # 当脚本以单文件形式导入（from patrol_task import ...）
    from cap.patrol_state_machine import PatrolStateMachine, PatrolBox, PatrolState


class SimplePatrolTask(MultipleCombatTask):
    """
    简单巡逻任务类
    实现4机在x=0-100之间的循环往复巡逻
    """
    
    def __init__(self, config):
        # 调用父类初始化
        super().__init__(config)
        self.config = config
        
        # 巡逻参数
        self.patrol_x_min = 0.0      # 巡逻区域x最小值（km）
        self.patrol_x_max = 100.0    # 巡逻区域x最大值（km）
        
        # 飞机状态记录（用于判断是否需要转向）
        self.aircraft_states = {}  # agent_id -> {'x': float, 'heading': float, 'target_heading': float}

        # 巡逻状态机（每架友机一台；敌机不使用）
        self.patrol_machines: Dict[str, PatrolStateMachine] = {}
        
        # 动作映射：航向命令索引 -> 航向角度（度）
        # 航向命令索引范围：0-40，对应-180到180度
        self.heading_cmd_to_deg = {}
        for i in range(41):
            self.heading_cmd_to_deg[i] = -180 + i * 9  # 每步9度
        
        # 🔥 初始化底层模型（参考tactical_task.py）
        self._load_baseline_model()
        self.friend_lowlevel_type = os.getenv("FRIEND_BASELINE_MODEL", "SU27").upper()
        self.enemy_lowlevel_type = os.getenv("ENEMY_BASELINE_MODEL", "F16").upper()
        self.enemy_use_f16_native = os.getenv("CAP_ENEMY_F16_NATIVE_ENABLED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        
        # 初始化动作归一化数组（参考tactical_task.py）
        self.norm_delta_altitude = NORM_ALTITUDE_3.copy()
        
        self.norm_delta_heading = NORM_HEADING_5.copy()
        self.norm_delta_velocity = NORM_VELOCITY_3.copy()
        self.action_space = spaces.MultiDiscrete([3, 5, 3])
        
        # 初始化RNN状态字典
        self._inner_rnn_states = {}
        
        # 🔥 修复：添加state_manager占位符（避免AttributeError）
        self.state_manager = None
        
        # 🔥 从配置文件读取初始位置（用于巡逻逻辑）
        self._load_initial_positions()

        # 日志开关与节流（默认：关闭敌方详细调试；我方巡逻日志降频）
        self.debug_enemy = os.getenv("CAP_DEBUG_ENEMY", "0") == "1"
        self.debug_friend = os.getenv("CAP_DEBUG_FRIEND", "0") == "1"
        self.patrol_log_interval_steps = int(os.getenv("CAP_PATROL_LOG_INTERVAL_STEPS", "300"))
        patrol_log_agents_env = os.getenv("CAP_PATROL_LOG_AGENTS", "A0100").strip()
        self.patrol_log_agents = {s.strip() for s in patrol_log_agents_env.split(",") if s.strip()}
        
        logging.info("✅ [巡逻任务] 简单巡逻任务初始化完成")
        logging.info(f"   巡逻区域: x=[{self.patrol_x_min}, {self.patrol_x_max}] km")
        logging.info(f"   低层模型: 我方={self.friend_lowlevel_type}, 敌方={self.enemy_lowlevel_type}")

    def reset(self, env):
        """重置任务状态。

        关键：重置 RNN 状态与保持目标，否则多次 episode 会带来不可控漂移。
        """
        try:
            self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128), dtype=np.float32) for agent_id in env.agents.keys()}
        except Exception:
            self._inner_rnn_states = {}
        self.aircraft_states = {}
        self.patrol_machines = {}
        self.step_count = 0
        ret = super().reset(env)
        # reset后初始化巡逻状态机：以当前初始NED位置自动推导巡逻矩形
        self._init_patrol_machines(env)
        return ret
    
    def _load_initial_positions(self):
        """从配置文件读取初始位置（经纬度），用于巡逻与诊断"""
        self.initial_latitudes = {}   # agent_id -> 初始纬度
        self.initial_longitudes = {}  # agent_id -> 初始经度
        
        try:
            # 从config中读取aircraft_configs（可能是字典或对象）
            if hasattr(self.config, 'aircraft_configs'):
                aircraft_configs = self.config.aircraft_configs
                for agent_id, config in aircraft_configs.items():
                    # 处理字典格式或对象格式
                    if isinstance(config, dict):
                        init_state = config.get('init_state', {})
                        lat = init_state.get('ic_lat_geod_deg', None)
                    else:
                        # 对象格式
                        if hasattr(config, 'init_state'):
                            init_state = config.init_state
                            if isinstance(init_state, dict):
                                lat = init_state.get('ic_lat_geod_deg', None)
                            elif hasattr(init_state, 'ic_lat_geod_deg'):
                                lat = init_state.ic_lat_geod_deg
                            else:
                                lat = None
                        else:
                            lat = None
                    
                    if lat is not None:
                        self.initial_latitudes[agent_id] = lat
                        logging.info(f"   {agent_id} 初始纬度: {lat:.4f}°")

                    # 经度同理
                    if isinstance(config, dict):
                        init_state = config.get('init_state', {})
                        lon = init_state.get('ic_long_gc_deg', None)
                    else:
                        if hasattr(config, 'init_state'):
                            init_state = config.init_state
                            if isinstance(init_state, dict):
                                lon = init_state.get('ic_long_gc_deg', None)
                            elif hasattr(init_state, 'ic_long_gc_deg'):
                                lon = init_state.ic_long_gc_deg
                            else:
                                lon = None
                        else:
                            lon = None
                    if lon is not None:
                        self.initial_longitudes[agent_id] = lon
                        logging.info(f"   {agent_id} 初始经度: {lon:.4f}°")
        except Exception as e:
            logging.warning(f"⚠️ 读取初始位置失败: {e}，将使用默认值")
            import traceback
            logging.debug(f"Traceback: {traceback.format_exc()}")
            # 默认值（如果读取失败）
            self.initial_latitudes = {
                'A0100': 60.0,
                'A0300': 60.0,
                'A0200': 60.9009,
                'A0400': 60.9009,
            }
            self.initial_longitudes = {
                'A0100': 120.6757,
                'A0200': 120.2252,
                'A0300': 121.5766,
                'A0400': 121.1261,
            }

    def _init_patrol_machines(self, env):
        """初始化巡逻状态机：从reset后的NED位置推导巡逻矩形（km）。

        目标：最小改动实现“横向过渡段(90/270)”而不引入额外坐标系统依赖。
        """
        self.patrol_machines = {}

        def _xy_km(aid: str) -> Optional[Tuple[float, float]]:
            ac = env.agents.get(aid)
            if ac is None or not getattr(ac, 'is_alive', False):
                return None
            pos = ac.get_position()  # NED: (north, east, down)
            x_km = float(pos[1]) / 1000.0  # east
            y_km = float(pos[0]) / 1000.0  # north
            return x_km, y_km

        formations = {
            'left': ('A0100', 'A0200'),
            'right': ('A0300', 'A0400'),
        }

        for _, (lead_id, wing_id) in formations.items():
            lead_xy = _xy_km(lead_id)
            wing_xy = _xy_km(wing_id)
            if lead_xy is None or wing_xy is None:
                continue

            x_vals = [lead_xy[0], wing_xy[0]]
            y_vals = [lead_xy[1], wing_xy[1]]
            box = PatrolBox(
                x_min=min(x_vals),
                x_max=max(x_vals),
                y_min=min(y_vals),
                y_max=max(y_vals),
            )

            # 长机初始热段(朝北)，僚机初始冷段(朝南)
            self.patrol_machines[lead_id] = PatrolStateMachine(box, initial_hot=True)
            self.patrol_machines[wing_id] = PatrolStateMachine(box, initial_hot=False)

            logging.info(
                f"🧭 [巡逻初始化] box({lead_id}/{wing_id}) x=[{box.x_min:.1f},{box.x_max:.1f}]km y=[{box.y_min:.1f},{box.y_max:.1f}]km tol={self.patrol_machines[lead_id].tolerance:.1f}km"
            )
    
    def _load_baseline_model(self):
        """加载低层控制模型（参考tactical_task.py）"""
        root_dir = get_root_dir()
        f16_path = os.path.join(root_dir, 'model', 'baseline_model.pt')
        tactical_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        f16_native_path = os.getenv(
            "CAP_ENEMY_F16_NATIVE_PATH",
            os.path.join(tactical_root, 'models', 'f16_cap_lowlevel_native.pt'),
        )
        f16_native_meta_path = os.getenv(
            "CAP_ENEMY_F16_NATIVE_META_PATH",
            os.path.join(tactical_root, 'models', 'f16_cap_lowlevel_native.json'),
        )
        su27_path = r"D:\Pycharm\LAG\lqyLAG\scripts\results\SingleControl\1\heading_su27\ppo\su27_baseline_v1\run40\actor_990.pt"
        
        # F16 BaselineActor
        self.baseline_model = None
        self.f16_cap_model = None
        self.f16_cap_command_semantics = "cap_native_15x17x7"
        if os.path.exists(f16_path):
            try:
                state_dict = torch.load(f16_path, map_location='cpu', weights_only=True)
                use_mlp = any(k.startswith('act.mlp') for k in state_dict.keys())
                self.baseline_model = BaselineActor(input_dim=12, use_mlp_actlayer=use_mlp)
                self.baseline_model.load_state_dict(state_dict)
                self.baseline_model.eval()
                logging.info(f"✅ F16 Baseline模型加载成功")
            except Exception as e:
                logging.error(f"❌ 加载F16 Baseline模型失败: {e}")
                self.baseline_model = None
        else:
            logging.warning(f"⚠️ 未找到F16 Baseline模型文件: {f16_path}")
        
        if os.path.exists(f16_native_path):
            try:
                native_state_dict = torch.load(f16_native_path, map_location='cpu', weights_only=True)
                use_mlp = any(k.startswith('act.mlp') for k in native_state_dict.keys())
                self.f16_cap_model = BaselineActor(input_dim=12, use_mlp_actlayer=use_mlp)
                self.f16_cap_model.load_state_dict(native_state_dict)
                self.f16_cap_model.eval()
                if os.path.exists(f16_native_meta_path):
                    with open(f16_native_meta_path, 'r', encoding='utf-8') as meta_file:
                        metadata = json.load(meta_file)
                    self.f16_cap_command_semantics = str(
                        metadata.get('command_semantics', self.f16_cap_command_semantics)
                    )
                semantics_override = str(os.getenv("CAP_ENEMY_F16_NATIVE_COMMAND_SEMANTICS", "")).strip()
                if semantics_override:
                    self.f16_cap_command_semantics = semantics_override
                if 'residual' in self.f16_cap_command_semantics.lower():
                    logging.warning(
                        "SimplePatrolTask only supports legacy 3x5x3 low-level semantics; "
                        "residual native F16 is disabled here."
                    )
                    self.f16_cap_model = None
                logging.info("鉁?F16 CAP Native妯″瀷鍔犺浇鎴愬姛")
            except Exception as e:
                logging.error(f"鉂?鍔犺浇F16 CAP Native妯″瀷澶辫触: {e}")
                self.f16_cap_model = None
        if self.f16_cap_model is not None and not getattr(self, 'enemy_use_f16_native', False):
            logging.info("敌方F16默认使用legacy baseline，禁用CAP Native低层模型")

        # SU27 Baseline
        self.su27_baseline_actor = None
        if os.path.exists(su27_path):
            try:
                self.su27_baseline_actor = BaselineActor(input_dim=12, use_mlp_actlayer=True)
                state_dict = torch.load(su27_path, map_location=torch.device('cpu'), weights_only=True)
                use_mlp = any(k.startswith('act.mlp') for k in state_dict.keys())
                if not use_mlp:
                    self.su27_baseline_actor = BaselineActor(input_dim=12, use_mlp_actlayer=False)
                self.su27_baseline_actor.load_state_dict(state_dict)
                self.su27_baseline_actor.eval()
                logging.info(f"✅ SU27 Baseline模型加载成功")
            except Exception as e:
                logging.error(f"❌ 加载SU27 Baseline模型失败: {e}")
                self.su27_baseline_actor = None
        else:
            logging.warning(f"⚠️ 未找到SU27 Baseline模型文件: {su27_path}")
    
    def normalize_action(self, env, agent_id, action):
        """
        归一化动作（从离散索引转换为连续控制命令）
        
        这个方法会被环境调用，我们需要：
        1. 调用get_action获取(alt_cmd, hdg_cmd, vel_cmd)
        2. 使用低层策略生成连续的控制命令
        """
        # 调用get_action获取离散动作索引
        alt_cmd, hdg_cmd, vel_cmd = self.get_action(env, agent_id)
        
        # 使用低层策略生成控制命令
        return self._use_lowlevel_policy(env, agent_id, alt_cmd, hdg_cmd, vel_cmd)
    
    def _use_lowlevel_policy(self, env, agent_id: str, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int) -> np.ndarray:
        """
        使用低层模型生成底层控制（参考tactical_task.py）
        """
        # 根据阵营选择目标模型类型
        is_enemy = agent_id.startswith('B')
        preferred_type = getattr(self, 'enemy_lowlevel_type' if is_enemy else 'friend_lowlevel_type', 'F16').upper()
        
        use_su27 = (preferred_type == 'SU27' and getattr(self, 'su27_baseline_actor', None) is not None)
        use_f16_native = (
            not use_su27
            and preferred_type == 'F16'
            and ((not is_enemy) or getattr(self, 'enemy_use_f16_native', False))
            and getattr(self, 'f16_cap_model', None) is not None
        )
        use_f16 = (not use_su27 and getattr(self, 'baseline_model', None) is not None)
        
        # 🔥 调试：检查敌方飞机模型加载情况（默认关闭）
        if self.debug_enemy and is_enemy and env.current_step < 5:
            logging.warning(
                f"🔍 [敌方模型检查-{agent_id}] preferred_type={preferred_type}, use_su27={use_su27}, use_f16={use_f16}, baseline_model={'已加载' if self.baseline_model else '未加载'}"
            )
        
        if not use_su27 and not use_f16_native and not use_f16:
            # 没有任何低层模型可用，退回默认控制
            if is_enemy:
                logging.error(f"❌ [敌方模型缺失-{agent_id}] 没有可用的底层模型，使用默认控制")
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        try:
            # 1. 获取原始观测（与tactical_task完全一致）
            raw_obs = self.get_obs(env, agent_id)
            
            # 🔥 关键修复：与tactical_task完全一致，raw_obs直接来自get_obs，已经是numpy数组
            # 不需要转换，保持原始类型（与tactical_task.py第2004行完全一致）
            
            # 🔥 调试：检查raw_obs是否正确获取（默认关闭）
            if self.debug_enemy and is_enemy and env.current_step < 5:
                logging.warning(f"🔍 [raw_obs检查-{agent_id}] raw_obs长度={len(raw_obs)}, raw_obs[:9]={raw_obs[:9]}")
            
            # 2. 构造12维输入（与tactical_task完全一致：使用默认dtype）
            input_obs = np.zeros(12)
            # 🔥 修复：确保索引在有效范围内（与tactical_task一致）
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)
            
            # 🔥 调试：检查敌方飞机的动作索引（默认关闭）
            if self.debug_enemy and is_enemy and env.current_step < 5:
                logging.warning(
                    f"🔍 [敌方动作索引-{agent_id}] alt_cmd={altitude_cmd_id}({self.norm_delta_altitude[altitude_cmd_id]:.3f}), hdg_cmd={heading_cmd_id}({self.norm_delta_heading[heading_cmd_id]:.3f}), vel_cmd={velocity_cmd_id}({self.norm_delta_velocity[velocity_cmd_id]:.3f})"
                )
            
            # 3. 初始化RNN状态（与tactical_task一致）
            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)
            
            # 4. 模型推理
            if use_su27:
                # SU27 BaselineActor 路径
                aircraft = env.agents[agent_id]
                h_sl_m = aircraft.get_property_value(c.position_h_sl_m)
                roll_rad = aircraft.get_property_value(c.attitude_roll_rad)
                pitch_rad = aircraft.get_property_value(c.attitude_pitch_rad)
                u_mps = aircraft.get_property_value(c.velocities_u_mps)
                v_mps = aircraft.get_property_value(c.velocities_v_mps)
                w_mps = aircraft.get_property_value(c.velocities_w_mps)
                vc_mps = aircraft.get_property_value(c.velocities_vc_mps)
                
                su27_raw_obs = np.zeros(9, dtype=np.float32)
                su27_raw_obs[0] = h_sl_m / 5000.0
                su27_raw_obs[1] = np.sin(roll_rad)
                su27_raw_obs[2] = np.cos(roll_rad)
                su27_raw_obs[3] = np.sin(pitch_rad)
                su27_raw_obs[4] = np.cos(pitch_rad)
                su27_raw_obs[5] = u_mps / 340.0
                su27_raw_obs[6] = v_mps / 340.0
                su27_raw_obs[7] = w_mps / 340.0
                su27_raw_obs[8] = vc_mps / 340.0
                
                input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
                input_obs[1] = self.norm_delta_heading[heading_cmd_id]
                input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
                input_obs[3:12] = su27_raw_obs
                input_obs = np.nan_to_num(input_obs, nan=0.0)
                
                obs_tensor = torch.FloatTensor(input_obs).unsqueeze(0)
                rnn_states_tensor = torch.FloatTensor(self._inner_rnn_states[agent_id])
                
                with torch.no_grad():
                    actions, rnn_states_out = self.su27_baseline_actor(
                        obs_tensor,
                        rnn_states_tensor
                    )
                
                action_output = actions.detach().cpu().numpy()[0]
                self._inner_rnn_states[agent_id] = rnn_states_out.detach().cpu().numpy()
            else:
                # F16 BaselineActor 路径（与tactical_task完全一致）
                # 🔥 关键修复：直接使用raw_obs[:9]，不需要再次转换（已在上面处理）
                # raw_obs[0:9]包含：高度、roll_sin、roll_cos、pitch_sin、pitch_cos、u/340、v/340、w/340、vc/340
                # 与tactical_task.py第2064行完全一致：input_obs[3:12] = raw_obs[:9]
                if len(raw_obs) < 9:
                    logging.error(f"❌ [F16路径-{agent_id}] raw_obs长度不足: {len(raw_obs)} < 9，使用零填充")
                    raw_obs_padded = np.zeros(9, dtype=np.float32)
                    raw_obs_padded[:len(raw_obs)] = raw_obs
                    raw_obs = raw_obs_padded
                
                input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
                input_obs[1] = self.norm_delta_heading[heading_cmd_id]
                input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
                input_obs[3:12] = raw_obs[:9]  # 使用前9维自身状态（与tactical_task完全一致）
                input_obs = np.nan_to_num(input_obs, nan=0.0)
                
                # 🔥 修复：确保输入形状正确（与tactical_task保持一致）
                obs_expanded = np.expand_dims(input_obs, axis=0)
                # 确保 rnn_states 为 float32
                if not isinstance(self._inner_rnn_states[agent_id], np.ndarray):
                    self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)
                self._inner_rnn_states[agent_id] = self._inner_rnn_states[agent_id].astype(np.float32)
                active_f16_model = self.f16_cap_model if use_f16_native else self.baseline_model
                _action, _rnn_states = active_f16_model(
                    torch.FloatTensor(obs_expanded),
                    torch.FloatTensor(self._inner_rnn_states[agent_id])
                )
                action_output = _action.detach().cpu().numpy().squeeze(0)
                self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
                
                # 🔥 调试：检查模型输入输出（默认关闭；敌方更吵，默认不打）
                if env.current_step < 5:
                    if (is_enemy and self.debug_enemy) or ((not is_enemy) and self.debug_friend):
                        logging.warning(
                            f"🔍 [{ '敌方' if is_enemy else '友方' }模型-{agent_id}] input_obs[0:3]={input_obs[0:3]}, input_obs[3:12]={input_obs[3:12]}, action_output={action_output}"
                        )
            
            # 5. 归一化输出（与tactical_task完全一致）
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            # 6. 最后安全裁剪：避免异常数值导致直接失控
            clipped = False
            clip_min = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
            clip_max = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            norm_act_clipped = np.clip(norm_act.astype(np.float32), clip_min, clip_max)
            if not np.allclose(norm_act, norm_act_clipped):
                clipped = True
                logging.warning(f"⚠️ [{agent_id}] norm_act 被裁剪: 原始={norm_act}, 裁剪后={norm_act_clipped}")
            norm_act = norm_act_clipped

            # 🔥 调试：检查控制输出（默认关闭；敌方更吵，默认不打）
            if env.current_step < 5:
                if (is_enemy and self.debug_enemy) or ((not is_enemy) and self.debug_friend):
                    current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                    roll_rad = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
                    logging.warning(
                        f"🔍 [{ '敌方' if agent_id.startswith('B') else '友方' }控制输出-{agent_id}] norm_act={norm_act}, current_alt={current_alt:.1f}m, action_output={action_output}, roll={np.rad2deg(roll_rad):.1f}°"
                    )
            
            return norm_act
            
        except Exception as e:
            logging.error(f"❌ [巡逻任务] _use_lowlevel_policy失败 {agent_id}: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            # 返回安全的默认控制命令（轻微拉高，防止坠毁）
            return np.array([0.0, 0.1, 0.0, 0.7])
    
    def get_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        """
        获取巡逻动作
        
        巡逻逻辑：
        - a0100, a0300: 初始航向0度（朝北），到达x=100时转向180度，到达x=0时转向0度
        - a0200, a0400: 初始航向180度（朝南），到达x=0时转向0度，到达x=100时转向180度
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 动作三元组
        """
        # 获取飞机当前位置和状态
        aircraft = env.agents.get(agent_id)
        if aircraft is None or not aircraft.is_alive:
            return ALT_HOLD, HDG_HOLD, VEL_HOLD
        
        # 🔥 敌方：添加航向修正逻辑，保持180°（朝南）航向
        if agent_id.startswith('B'):
            # 获取当前航向（度）
            heading_deg = np.degrees(aircraft.get_property_values([c.attitude_heading_true_rad])[0])
            # 归一化到0-360度
            if heading_deg < 0:
                heading_deg += 360
            
            # 目标航向：180°（朝南）
            target_heading = 180.0
            
            # 计算航向差（度）
            heading_diff = target_heading - heading_deg
            # 处理角度环绕
            if heading_diff > 180:
                heading_diff -= 360
            elif heading_diff < -180:
                heading_diff += 360
            
            # 将航向差转换为norm_delta_heading的索引（0-16）
            heading_diff_rad = np.radians(heading_diff)  # 转换为弧度
            
            # 找到最接近的norm_delta_heading索引
            distances = np.abs(self.norm_delta_heading - heading_diff_rad)
            hdg_cmd = int(np.argmin(distances))
            
            # 如果航向差很小（<5度），使用索引8（保持航向）
            if abs(heading_diff) < 5.0:
                hdg_cmd = HDG_HOLD  # 索引8对应0（保持航向）
            
            # 高度和速度保持
            alt_cmd = ALT_HOLD
            vel_cmd = VEL_HOLD
            
            # 调试日志（可选）
            if env.current_step % 300 == 0:  # 每60秒输出一次
                logging.info(
                    f"🛫 [敌方航向修正-{agent_id}] 当前航向={heading_deg:.2f}°, 目标航向={target_heading:.1f}°, "
                    f"航向差={heading_diff:+.2f}°, hdg_cmd={hdg_cmd}"
                )
            
            return alt_cmd, hdg_cmd, vel_cmd
        
        # 获取位置（NED坐标系，单位：米）
        position = aircraft.get_position()
        x_ned = position[1]  # east (m)
        y_ned = position[0]  # north (m)

        # 转换为公里（用于巡逻状态机）
        x_km = float(x_ned) / 1000.0
        y_km = float(y_ned) / 1000.0

        # 🔥 获取当前纬度（仅用于日志）
        geodetic = aircraft.get_geodetic()
        current_lat = geodetic[1] if geodetic is not None and len(geodetic) > 1 else None
        
        # 获取当前航向（度）
        heading_deg = np.degrees(aircraft.get_property_values([c.attitude_heading_true_rad])[0])
        # 归一化到[0, 360)
        heading_deg = float(heading_deg) % 360.0
        
        # 若巡逻状态机尚未初始化（极少数情况下reset链路不同），这里兜底初始化一次
        if not getattr(self, 'patrol_machines', None):
            self._init_patrol_machines(env)

        # 初始化飞机状态
        if agent_id not in self.aircraft_states:
            # 初始目标航向：优先由巡逻状态机给出；否则使用原默认
            if agent_id in self.patrol_machines:
                init_state = self.patrol_machines[agent_id].state
                target_heading = float(PatrolStateMachine.STATE_HEADINGS.get(init_state, 0.0))
            else:
                target_heading = 0.0 if agent_id in ['A0100', 'A0300'] else 180.0
            
            # 🔥 调试：记录初始航向
            logging.warning(f"🔍 [初始化-{agent_id}] 当前航向={heading_deg:.1f}°, 目标航向={target_heading:.1f}°")
            
            self.aircraft_states[agent_id] = {
                'x': x_km,
                'heading': heading_deg,
                'target_heading': target_heading,
                # 仅用于“转向事件”的方向一致性（避免 180->0 时偶发右转）
                'force_left_turn': False,
                'last_target_heading': target_heading,
                # 当前巡逻状态（用于日志/诊断；核心状态由patrol_machines维护）
                'patrol_state': str(getattr(self.patrol_machines.get(agent_id), 'state', 'UNKNOWN')),
            }
        
        state = self.aircraft_states[agent_id]
        state['x'] = x_km
        state['heading'] = heading_deg
        
        # 巡逻状态机：生成目标航向（含横向过渡段 90/270）
        prev_target_heading = float(state.get('target_heading', 0.0)) % 360.0
        prev_patrol_state = state.get('patrol_state', 'UNKNOWN')

        if agent_id in self.patrol_machines:
            patrol_state, target_heading = self.patrol_machines[agent_id].update(x_km, y_km)
            target_heading = float(target_heading) % 360.0
            state['patrol_state'] = str(patrol_state)
        else:
            # 兜底：若无状态机，保持原行为
            target_heading = prev_target_heading
        
        target_heading = float(target_heading) % 360.0
        state['target_heading'] = target_heading

        # 仅在“真正触发转向点”发生时，开启强制左转（只针对 0<->180 的翻转）
        # 这样不会改变平直段的微小修正方向，避免引入转圈。
        if target_heading != prev_target_heading or state.get('patrol_state') != prev_patrol_state:
            delta = abs(target_heading - prev_target_heading)
            delta = min(delta, 360.0 - delta)
            state['force_left_turn'] = (abs(delta - 180.0) < 1e-6)
            state['last_target_heading'] = target_heading

            # 诊断：记录一次转向事件
            logging.info(
                f"[TURN] {agent_id} {prev_patrol_state} -> {state.get('patrol_state')} | target {prev_target_heading:.1f}° -> {target_heading:.1f}° | cur={heading_deg:.1f}° x={x_km:.1f}km y={y_km:.1f}km"
            )
        
        # 计算航向差（度）：
        # - shortest_diff 用于“是否接近目标”的判定
        # - 在 0<->180 翻转转向事件中，可选择强制左转的 diff
        shortest_diff = target_heading - heading_deg
        if shortest_diff > 180:
            shortest_diff -= 360
        elif shortest_diff < -180:
            shortest_diff += 360

        heading_diff = shortest_diff

        # 在转向事件期间强制左转（避免 180->0/360 时偶发右转）
        if state.get('force_left_turn', False):
            # 若已非常接近目标，解除强制左转
            if abs(shortest_diff) < 5.0:
                state['force_left_turn'] = False
            else:
                # left_amount: 当前航向减到目标航向需要左转的角度 (0..360)
                left_amount = (heading_deg - target_heading) % 360.0
                # 将其转为负角度（左转为负），并限制到 [-180, 0)
                forced_left = -left_amount
                if forced_left < -180.0:
                    forced_left = -180.0
                # forced_left 绝不应为正（若出现则说明角度归一化出错）
                if forced_left > 0.0:
                    logging.warning(
                        f"[TURN-ANOMALY] {agent_id} forced_left became +{forced_left:.2f} (cur={heading_deg:.1f}, tgt={target_heading:.1f})"
                    )
                    forced_left = -abs(forced_left)
                heading_diff = forced_left

                # 诊断：只有当 shortest_diff 指向右转时才打（用于定位“为何会偶发右转”）
                if shortest_diff > 0:
                    logging.warning(
                        f"[TURN-DBG] {agent_id} shortest_diff=+{shortest_diff:.1f}° (would-right) -> force_left {heading_diff:.1f}° | cur={heading_deg:.1f} tgt={target_heading:.1f}"
                    )
        
        # 🔥 修复：僚机A0200和A0400强制左转（逆时针）
        # 当航向差接近±180度时，强制选择左转（索引0，对应-π）
        if agent_id in ['A0200', 'A0400'] and abs(heading_diff) >= 175.0:
            # 强制左转（逆时针）
            heading_diff = -180.0  # 强制左转180度
            hdg_cmd = 0  # 索引0对应-π（左转180°）
        else:
            # 🔥 修复：将航向差转换为norm_delta_heading的索引（0-16）
            # norm_delta_heading有17个值，对应从-π到π的航向修正
            # 索引8对应0（保持航向），索引0对应-π（左转180°），索引16对应π（右转180°）
            heading_diff_rad = np.radians(heading_diff)  # 转换为弧度
            
            # 找到最接近的norm_delta_heading索引
            # 计算每个norm_delta_heading值与heading_diff_rad的距离
            distances = np.abs(self.norm_delta_heading - heading_diff_rad)
            hdg_cmd = int(np.argmin(distances))
            
            # 如果航向差很小（<5度），使用索引8（保持航向）
            if abs(shortest_diff) < 5.0:
                hdg_cmd = HDG_HOLD  # 索引8对应0（保持航向）
        
        # 高度命令：保持中等高度（索引7对应约9000米）
        alt_cmd = ALT_HOLD
        
        # 速度命令：保持中等速度（索引3）
        vel_cmd = VEL_HOLD
        
        # 巡逻状态日志（降频 + 可配置，仅打指定飞机）
        if (
            agent_id in self.patrol_log_agents
            and self.patrol_log_interval_steps > 0
            and env.current_step % self.patrol_log_interval_steps == 0
        ):
            logging.info(
                f"🛫 [巡逻-{agent_id}] x={x_km:.1f}km, 当前航向={heading_deg:.1f}°, 目标航向={target_heading:.1f}°, 航向差={heading_diff:.1f}°, hdg_cmd={hdg_cmd}"
            )

        # 🔍 友方调试：首3步打印高层指令，便于与敌方对比
        if env.current_step < 3:
            logging.warning(
                f"🔍 [友方动作-{agent_id}] alt_cmd={alt_cmd}({self.norm_delta_altitude[alt_cmd]:+.3f}), hdg_cmd={hdg_cmd}({self.norm_delta_heading[hdg_cmd]:+.3f}), vel_cmd={vel_cmd}({self.norm_delta_velocity[vel_cmd]:+.3f})"
            )
        
        return alt_cmd, hdg_cmd, vel_cmd
    
    def step(self, env):
        """
        执行一步仿真（简化版，只实现基础功能）
        
        返回:
            obs: 观测字典 {agent_id: obs_array}
            share_obs: 共享观测字典 {agent_id: share_obs_array}
            rewards: 奖励字典 {agent_id: reward_array}
            dones: 终止标志字典 {agent_id: [done]}
            infos: 信息字典 {agent_id: info_dict}
        """
        try:
            # 初始化step_count
            if not hasattr(self, 'step_count'):
                self.step_count = 0
            self.step_count += 1
            
            # 获取观测
            obs = {agent_id: self.get_obs(env, agent_id) for agent_id in env.agents.keys()}
            
            # 构建共享观测（所有智能体的观测拼接）
            all_obs = np.stack([obs[agent_id] for agent_id in sorted(env.agents.keys())], axis=0)
            share_obs_flat = all_obs.flatten()
            share_obs = {agent_id: share_obs_flat for agent_id in env.agents.keys()}
            
            # 计算奖励（简化版：只使用基础奖励）
            rewards = {}
            dones = {}
            infos = {}
            
            for agent_id in env.agents.keys():
                # 基础奖励：保持飞行给小的正奖励
                reward = 0.1 if env.agents[agent_id].is_alive else 0.0
                rewards[agent_id] = np.array([reward])
                
                # 检查终止条件
                done = not env.agents[agent_id].is_alive
                dones[agent_id] = [done]
                
                # 构建信息字典
                infos[agent_id] = {
                    "step_count": self.step_count,
                    "is_alive": env.agents[agent_id].is_alive,
                }
            
            return obs, share_obs, rewards, dones, infos
            
        except Exception as e:
            logging.error(f"❌ [巡逻任务] step方法失败: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # 返回默认值避免返回None
            obs = {agent_id: np.zeros(self.obs_length) for agent_id in env.agents.keys()}
            share_obs = {agent_id: np.zeros(self.obs_length * len(env.agents)) for agent_id in env.agents.keys()}
            rewards = {agent_id: np.array([0.0]) for agent_id in env.agents.keys()}
            dones = {agent_id: [False] for agent_id in env.agents.keys()}
            infos = {agent_id: {} for agent_id in env.agents.keys()}
            return obs, share_obs, rewards, dones, infos

