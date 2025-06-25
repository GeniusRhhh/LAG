import logging
import time
import traceback

import numpy as np
import torch
from algorithms.utils.buffer import SharedReplayBuffer
from .base_runner import Runner


def _t2n(x):
    """将张量转换为 numpy 数组。"""
    return x.detach().cpu().numpy()


class ShareJSBSimRunner(Runner):
    """增强版多线程 JSBSim 运行器，支持自博弈和战术模板训练。"""

    def load(self):
        """加载环境、策略和缓冲区。"""
        self.obs_space = self.envs.observation_space
        self.share_obs_space = self.envs.share_observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay
        self.use_rule_opponent = getattr(self.all_args, 'use_rule_opponent', True)

        # 新增：战术模板训练配置
        self.use_tactical_templates = getattr(self.all_args, 'use_tactical_templates', True)
        self.template_curriculum = getattr(self.all_args, 'template_curriculum', False)
        self.template_analysis_interval = getattr(self.all_args, 'template_analysis_interval', 100)

        if self.algorithm_name == "mappo":
            from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.mappo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError("Only MAPPO algorithm is supported")

        self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)
        self.buffer = SharedReplayBuffer(self.all_args, self.num_agents // 2 if self.use_selfplay else self.num_agents,
                                         self.obs_space, self.share_obs_space, self.act_space)

        # 战术模板使用统计
        self.template_stats = {
            "usage_count": np.zeros(15),  # 15个模板（0-14）
            "success_rate": np.zeros(15),
            "reward_per_template": np.zeros(15),
            "phase_distribution": {},
            "cooperative_efficiency": 0.0
        }
        if self.use_selfplay:
            from algorithms.utils.selfplay import get_algorithm

            self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)
            self.n_choose_opponents = max(getattr(self.all_args, 'n_choose_opponents', 1), 1)
            if self.n_choose_opponents > self.n_rollout_threads:
                logging.warning(
                    f"Number of opponent choices {self.n_choose_opponents} > number of threads {self.n_rollout_threads}, set to {self.n_rollout_threads}")
                self.n_choose_opponents = self.n_rollout_threads
            self.policy_pool = {'latest': self.all_args.init_elo}
            self.opponent_policy = [
                Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
                for _ in range(self.n_choose_opponents)]
            self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
            self.opponent_obs = np.zeros_like(self.buffer.obs[0])
            self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
            self.opponent_masks = np.ones_like(self.buffer.masks[0])
            if self.use_eval:
                self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space,
                                                   device=self.device)
            logging.info(
                f"Loaded self-play opponents: algorithm={self.all_args.selfplay_algorithm}, opponent count={self.n_choose_opponents}, rule-based opponent={self.use_rule_opponent}")
        else:
            self.opponent_policy = []
            self.opponent_env_split = []

        if self.model_dir is not None:
            self.restore()

        # logging.info(f"Enhanced ShareJSBSimRunner loaded: tactical_templates={self.use_tactical_templates}, "
        #              f"template_curriculum={self.template_curriculum}")

    def run(self):
        """主训练循环，基于现有代码结构的修复版本"""
        self.warmup()
        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads
        win_rates = []
        template_performance_history = []

        # 异常监控
        nan_count = 0
        crash_count = 0
        divergence_count = 0
        last_valid_checkpoint = None

        for episode in range(episodes):
            try:
                episode_rewards = []
                episode_actions = []
                episode_phases = []
                episode_templates = []
                episode_cooperations = []

                # 检查是否需要从检查点恢复
                if divergence_count > 3 and last_valid_checkpoint is not None:
                    logging.warning(f"训练发散 {divergence_count} 次，恢复检查点")
                    self.policy.actor.load_state_dict(last_valid_checkpoint['actor'])
                    self.policy.critic.load_state_dict(last_valid_checkpoint['critic'])
                    divergence_count = 0
                    # 降低学习率
                    for param_group in self.policy.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                        logging.info(f"学习率降低至 {param_group['lr']}")

                for step in range(self.buffer_size):
                    values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                    # 检查动作中的 NaN
                    if np.any(np.isnan(actions)):
                        logging.error(f"步骤 {step} 动作中检测到 NaN")
                        nan_count += 1
                        # 使用安全动作
                        actions = np.zeros_like(actions)
                        actions[:, :, 0] = 7  # Simple_F_Pole
                        actions[:, :, 1] = 0  # 不射击

                    # 执行环境步骤
                    obs, share_obs, rewards, dones, infos = self.envs.step(actions)

                    # 检查环境输出中的 NaN
                    if np.any(np.isnan(obs)):
                        logging.error(f"步骤 {step} 观测值中检测到 NaN")
                        nan_count += 1
                        obs = np.nan_to_num(obs, nan=0.0)
                        share_obs = np.nan_to_num(share_obs, nan=0.0)

                    if np.any(np.isnan(rewards)):
                        logging.error(f"步骤 {step} 奖励中检测到 NaN")
                        nan_count += 1
                        rewards = np.zeros_like(rewards)

                    # 奖励异常检测和限制
                    max_reward = np.max(np.abs(rewards))
                    if max_reward > 50.0:  # 检测异常大的奖励
                        logging.warning(f"步骤 {step} 检测到异常奖励: max_abs={max_reward:.3f}")
                        rewards = np.clip(rewards, -20.0, 20.0)  # 限制极端奖励

                    # 收集训练数据
                    step_rewards = rewards[0, :self.num_agents // 2]
                    episode_rewards.append(step_rewards)
                    episode_actions.append(actions[0, :self.num_agents // 2])

                    # 处理infos - 使用你现有的方式
                    step_phases = []
                    step_templates = []
                    step_cooperations = []

                    for agent_idx in range(self.num_agents // 2):
                        agent_info = None
                        if isinstance(infos, dict) and agent_idx in infos:
                            agent_info = infos[agent_idx]
                        elif isinstance(infos, list) and len(infos) > 0:
                            if isinstance(infos[0], dict) and agent_idx in infos[0]:
                                agent_info = infos[0][agent_idx]
                        elif isinstance(infos, np.ndarray) and len(infos) > 0:
                            if isinstance(infos[0], dict) and agent_idx in infos[0]:
                                agent_info = infos[0][agent_idx]

                        if agent_info is None:
                            agent_info = {}

                        # 收集阶段信息
                        phase = agent_info.get("current_phase", "unknown")
                        step_phases.append(phase)

                        # 收集战术模板信息
                        if hasattr(actions[0], '__len__') and len(actions[0]) > agent_idx:
                            action = actions[0][agent_idx]
                            if hasattr(action, '__len__') and len(action) > 0:
                                template_id = action[0] if isinstance(action[0], (int, np.integer)) else 0
                            else:
                                template_id = 0
                        else:
                            template_id = 0
                        step_templates.append(template_id)

                        # 收集协同信息
                        reward_details = agent_info.get("reward_details", {})
                        cooperation_reward = reward_details.get("TacticalRewardNew", 0) + reward_details.get(
                            "TacticalReward", 0)
                        step_cooperations.append(cooperation_reward)

                    episode_phases.append(step_phases)
                    episode_templates.append(step_templates)
                    episode_cooperations.append(step_cooperations)

                    data = obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic
                    self.insert(data)

                    # 检查崩溃
                    if isinstance(infos, dict):
                        for agent_id, info in infos.items():
                            if isinstance(info, dict) and info.get("crashed", False):
                                crash_count += 1

                self.compute()
                train_infos = self.train()
                self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads
                # **修复任务完成度评估**
                if episode % 10 == 0:
                    try:
                        if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0 and hasattr(self.envs.envs[0],
                                                                                              'task') and hasattr(
                                self.envs.envs[0].task, 'evaluate_first_task_completion'):
                            task_score, task_metrics = self.envs.envs[0].task.evaluate_first_task_completion()
                            logging.info(f"Episode {episode} - Task 1 Completion Score: {task_score:.3f}")
                            train_infos["task_1_completion"] = task_score
                            train_infos.update({f"task_1_{k}": v for k, v in task_metrics.items()})
                        else:
                            logging.warning("Task evaluation not available; skipping.")
                    except Exception as e:
                        logging.error(f"Task evaluation failed: {e}")

                # **修复时间线总结**
                if episode % 10 == 0:
                    try:
                        if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0 and hasattr(self.envs.envs[0],
                                                                                              'get_timeline_summary'):
                            timeline_summary = self.envs.envs[0].get_timeline_summary()
                            logging.info(f"Episode {episode} - Timeline Summary: {timeline_summary}")
                        else:
                            logging.warning("Timeline summary not available; skipping.")
                    except Exception as e:
                        logging.error(f"Timeline summary failed: {e}")
                # 检查训练发散
                if train_infos.get('value_loss', 0) > 1000 or np.isnan(train_infos.get('value_loss', 0)):
                    logging.error(f"训练发散！值损失：{train_infos['value_loss']}")
                    divergence_count += 1
                else:
                    divergence_count = 0
                    # 保存有效检查点
                    if episode % 10 == 0:
                        last_valid_checkpoint = {
                            'actor': self.policy.actor.state_dict(),
                            'critic': self.policy.critic.state_dict()
                        }

                # 战术模板性能分析 - 使用你现有的函数
                if self.use_tactical_templates:
                    flat_templates = []
                    flat_rewards = []
                    flat_phases = []
                    flat_cooperations = []

                    for step_idx in range(len(episode_templates)):
                        for agent_idx in range(self.num_agents // 2):
                            flat_templates.append(episode_templates[step_idx][agent_idx])
                            flat_rewards.append(episode_rewards[step_idx][agent_idx])
                            flat_phases.append(episode_phases[step_idx][agent_idx])
                            flat_cooperations.append(episode_cooperations[step_idx][agent_idx])

                    template_performance = self._analyze_template_performance(
                        flat_templates, flat_rewards, flat_phases, flat_cooperations)
                    template_performance_history.append(template_performance)

                # 收集胜率
                final_info = {}
                if isinstance(infos, dict):
                    final_info = infos
                elif isinstance(infos, (list, np.ndarray)) and len(infos) > 0:
                    final_info = infos[0] if isinstance(infos[0], dict) else {}

                if "win" in final_info:
                    win_rates.append(final_info["win"])

                # 保存模型
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(episode)

                # 定期日志输出
                if episode % self.log_interval == 0:
                    end = time.time()
                    avg_reward = np.mean([r.mean() for r in episode_rewards])
                    win_rate = np.mean(win_rates[-self.log_interval:]) if win_rates else 0

                    # 分析动作分布 - 使用你现有的方式
                    actions_array = np.array(episode_actions)
                    if actions_array.size > 0 and actions_array.ndim >= 3:
                        template_ids = actions_array[:, :, 0]
                        template_dist = {i: np.sum(template_ids == i) / template_ids.size for i in range(15)}
                        if actions_array.shape[2] > 1:
                            shoot_flags = actions_array[:, :, 1]
                            shoot_ratio = np.mean(shoot_flags)
                        else:
                            shoot_ratio = 0.0
                    else:
                        template_dist = {i: 0 for i in range(15)}
                        shoot_ratio = 0.0

                    # 分析阶段分布 - 使用你现有的方式
                    if episode_phases:
                        all_phases = []
                        for step_phases in episode_phases:
                            all_phases.extend(step_phases)
                        phase_counts = {}
                        for phase in all_phases:
                            phase_counts[phase] = phase_counts.get(phase, 0) + 1
                        total = len(all_phases)
                        phase_counts = {p: count / total for p, count in phase_counts.items()}
                    else:
                        phase_counts = {"unknown": 1.0}

                    # 更新战术统计 - 使用你现有的函数
                    self._update_template_stats(template_dist, avg_reward, phase_counts)
                    # **添加阶段转换分析**
                    self._analyze_phase_transitions(episode)

                    # **添加战术效果分析**
                    self._analyze_tactical_effectiveness(episode)
                    # 输出详细日志
                    logging.info(f"\n{'=' * 80}")
                    logging.info(f"Episode {episode}/{episodes} 总结：")
                    logging.info(f"{'=' * 80}")
                    logging.info(f"训练速度: {int(self.total_num_steps / (end - start))} FPS")
                    logging.info(f"平均奖励: {avg_reward:.3f}")
                    logging.info(f"胜率: {win_rate:.3f}")
                    logging.info(f"模板分布: {self._format_template_distribution(template_dist)}")
                    logging.info(f"射击比例: {shoot_ratio:.3f}")
                    logging.info(f"阶段分布: {phase_counts}")
                    logging.info(f"异常统计 - NaN: {nan_count}, 崩溃: {crash_count}, 发散: {divergence_count}")
                    logging.info(
                        f"训练损失 - 值: {train_infos.get('value_loss', 0):.3f}, 策略: {train_infos.get('policy_loss', 0):.3f}")
                    logging.info(f"{'=' * 80}\n")

                    # 战术模板分析日志 - 使用你现有的函数
                    if self.use_tactical_templates and episode % self.template_analysis_interval == 0:
                        self._log_tactical_analysis(template_performance_history[
                                                    -self.template_analysis_interval:] if template_performance_history else [])

                    # 更新训练信息
                    train_infos["average_episode_rewards"] = avg_reward
                    train_infos["win_rate"] = win_rate
                    train_infos["template_efficiency"] = self._calculate_template_efficiency()
                    train_infos["cooperative_score"] = np.mean(
                        [np.mean(coop) for coop in episode_cooperations]) if episode_cooperations else 0

                    self.log_info(train_infos, self.total_num_steps)

                    # 重置异常计数
                    if nan_count == 0 and crash_count == 0:
                        logging.info("此区间无异常，训练稳定")
                    nan_count = 0
                    crash_count = 0

                    # 自适应对手切换
                    if win_rate > 0.75 and self.use_rule_opponent:
                        logging.info("胜率较高，切换至自博弈模式")
                        self.use_rule_opponent = False

                # 评估
                if episode % self.eval_interval == 0 and self.use_eval:
                    self.eval(self.total_num_steps)

            except Exception as e:
                logging.error(f"Episode {episode} 错误: {e}")
                logging.error(f"回溯: {traceback.format_exc()}")
                continue

    def _analyze_template_performance(self, templates, rewards, phases, cooperations):
        """分析战术模板性能"""
        performance = {
            "template_rewards": np.zeros(15),
            "template_usage": np.zeros(15),
            "phase_effectiveness": {},
            "cooperation_scores": []
        }

        if not templates or not rewards:
            return performance

        templates_array = np.array(templates)
        rewards_array = np.array(rewards)

        # 计算每个模板的平均奖励
        for template_id in range(15):
            mask = templates_array == template_id
            if np.any(mask):
                performance["template_rewards"][template_id] = np.mean(rewards_array[mask])
                performance["template_usage"][template_id] = np.sum(mask)

        # 分析阶段效果
        if phases:
            phases_array = np.array(phases)
            unique_phases = np.unique(phases_array)
            for phase in unique_phases:
                phase_mask = phases_array == phase
                if np.any(phase_mask):
                    performance["phase_effectiveness"][phase] = np.mean(rewards_array[phase_mask])

        # 协同效果
        if cooperations:
            performance["cooperation_scores"] = np.mean(cooperations)

        return performance

    def _update_template_stats(self, template_dist, avg_reward, phase_counts):
        """更新战术模板统计"""
        for template_id, usage_ratio in template_dist.items():
            if usage_ratio > 0:
                # 使用量统计
                self.template_stats["usage_count"][template_id] += usage_ratio

                # 奖励统计（移动平均）
                alpha = 0.1
                self.template_stats["reward_per_template"][template_id] = (
                        (1 - alpha) * self.template_stats["reward_per_template"][template_id] +
                        alpha * avg_reward * usage_ratio
                )

        # 更新阶段分布
        for phase, count in phase_counts.items():
            if phase not in self.template_stats["phase_distribution"]:
                self.template_stats["phase_distribution"][phase] = 0
            self.template_stats["phase_distribution"][phase] = (
                    0.9 * self.template_stats["phase_distribution"][phase] + 0.1 * count
            )

    def _analyze_phase_transitions(self, episode):
        try:
            if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0 and hasattr(self.envs.envs[0],
                                                                                  'task') and hasattr(
                    self.envs.envs[0].task, 'timeline_events'):
                events = self.envs.envs[0].task.timeline_events
                if events:
                    recent_events = [e for e in events if e.get('step', 0) > episode * self.buffer_size - 100]
                    if recent_events:
                        phase_transitions = {}
                        for event in recent_events:
                            transition = event.get('phase_transition', '')
                            if transition:
                                phase_transitions[transition] = phase_transitions.get(transition, 0) + 1
                        logging.info(f"Recent Phase Transitions: {phase_transitions}")
                        full_sequence_agents = set()
                        for event in recent_events:
                            if 'effect_assessment' in event.get('phase_transition', ''):
                                full_sequence_agents.add(event.get('agent_id'))
                        if full_sequence_agents:
                            logging.info(f"Agents completed full sequence: {full_sequence_agents}")
            else:
                logging.warning("Timeline events not available; skipping phase transition analysis.")
        except Exception as e:
            logging.error(f"Phase transition analysis failed: {e}")

    def _analyze_tactical_effectiveness(self, episode):
        try:
            if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0 and hasattr(self.envs.envs[0],
                                                                                  'task') and hasattr(
                    self.envs.envs[0].task, 'decision_log'):
                decisions = self.envs.envs[0].task.decision_log
                if decisions:
                    recent_decisions = decisions[-100:] if len(decisions) >= 100 else decisions
                    phase_template_usage = {}
                    for decision in recent_decisions:
                        phase = decision.get('phase', 'unknown')
                        template = decision.get('template', 'unknown')
                        if phase not in phase_template_usage:
                            phase_template_usage[phase] = {}
                        phase_template_usage[phase][template] = phase_template_usage[phase].get(template, 0) + 1
                    logging.info("=== Tactical Effectiveness Analysis ===")
                    for phase, templates in phase_template_usage.items():
                        total = sum(templates.values())
                        template_ratios = {t: f"{count / total:.2f}" for t, count in templates.items()}
                        logging.info(f"{phase}: {template_ratios}")
                    logging.info("=======================================")
            else:
                logging.warning("Decision log not available; skipping tactical effectiveness analysis.")
        except Exception as e:
            logging.error(f"Tactical effectiveness analysis failed: {e}")
    def _calculate_template_efficiency(self):
        """计算战术模板整体效率"""
        total_usage = np.sum(self.template_stats["usage_count"])
        if total_usage == 0:
            return 0.0

        # 加权平均奖励
        weighted_reward = np.sum(
            self.template_stats["usage_count"] * self.template_stats["reward_per_template"]
        ) / total_usage

        # 多样性奖励（鼓励使用多种模板）
        usage_distribution = self.template_stats["usage_count"] / total_usage
        diversity = -np.sum(usage_distribution * np.log(usage_distribution + 1e-8))
        max_diversity = np.log(15)  # 15个模板的最大熵
        diversity_score = diversity / max_diversity

        return weighted_reward * (1 + 0.1 * diversity_score)

    def _format_template_distribution(self, template_dist):
        """格式化模板分布显示"""
        template_names = [
            "No_Template", "Crank", "Beam", "Notch", "Skate", "Short_Skate",
            "Banzai", "Simple_F_Pole", "Advanced_F_Pole", "Pincer",
            "Defensive_Split", "High_Low", "Engaging_Trail", "Loose_Deuce", "Defensive_Sequence"
        ]

        formatted = {}
        for template_id, usage in template_dist.items():
            if usage > 0.01:  # 只显示使用率超过1%的模板
                if template_id < len(template_names):
                    formatted[template_names[template_id]] = f"{usage:.3f}"

        return formatted

    def _log_tactical_analysis(self, performance_history):
        """记录战术分析日志"""
        if not performance_history:
            return

        logging.info("=== Tactical Analysis ===")

        # 最佳模板分析
        avg_template_rewards = np.mean([p["template_rewards"] for p in performance_history], axis=0)
        avg_template_usage = np.mean([p["template_usage"] for p in performance_history], axis=0)

        template_names = [
            "No_Template", "Crank", "Beam", "Notch", "Skate", "Short_Skate",
            "Banzai", "Simple_F_Pole", "Advanced_F_Pole", "Pincer",
            "Defensive_Split", "High_Low", "Engaging_Trail", "Loose_Deuce", "Defensive_Sequence"
        ]

        # Top 5 最有效模板
        effective_templates = []
        for i in range(15):
            if avg_template_usage[i] > 0:
                effectiveness = avg_template_rewards[i] / max(avg_template_usage[i], 1e-6)
                effective_templates.append((i, template_names[i], effectiveness, avg_template_usage[i]))

        effective_templates.sort(key=lambda x: x[2], reverse=True)

        logging.info("Top 5 Most Effective Templates:")
        for i, (template_id, name, effectiveness, usage) in enumerate(effective_templates[:5]):
            logging.info(f"  {i + 1}. {name}: Effectiveness={effectiveness:.3f}, Usage={usage:.3f}")

        # 阶段分析
        phase_performance = {}
        for p in performance_history:
            for phase, reward in p["phase_effectiveness"].items():
                if phase not in phase_performance:
                    phase_performance[phase] = []
                phase_performance[phase].append(reward)

        logging.info("Phase Performance:")
        for phase, rewards in phase_performance.items():
            avg_reward = np.mean(rewards)
            logging.info(f"  {phase}: {avg_reward:.3f}")

        # 协同效果
        avg_cooperation = np.mean([p["cooperation_scores"] for p in performance_history])
        logging.info(f"Average Cooperation Score: {avg_cooperation:.3f}")
        logging.info("========================")

    def warmup(self):
        """预热环境，初始化缓冲区。"""
        obs, share_obs = self.envs.reset()
        if self.use_selfplay:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """收集一步数据。"""
        self.policy.prep_rollout()

        # 检查并修复观测数据
        obs = np.concatenate(self.buffer.obs[step])
        rnn_states_actor = np.concatenate(self.buffer.rnn_states_actor[step])

        # 检查输入数据是否有NaN
        if np.isnan(obs).any():
            logging.error(f"NaN detected in obs at step {step}")
            # 统计NaN的位置
            nan_indices = np.where(np.isnan(obs))
            logging.error(f"NaN indices: {nan_indices}")

            # 替换NaN值
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

            # 如果NaN太多，重置环境
            nan_ratio = np.sum(np.isnan(obs)) / obs.size
            if nan_ratio > 0.1:  # 超过10%是NaN
                logging.warning("Too many NaN values, resetting environment")
                obs, share_obs = self.envs.reset()
                if self.use_selfplay:
                    self.opponent_obs = obs[:, self.num_agents // 2:, ...]
                    obs = obs[:, :self.num_agents // 2, ...]
                    share_obs = share_obs[:, :self.num_agents // 2, ...]
                self.buffer.obs[0] = obs.copy()
                self.buffer.share_obs[0] = share_obs.copy()
                self.buffer.step = 0
                # 重新收集第0步
                return self.collect(0)

        # 检查RNN状态
        if np.isnan(rnn_states_actor).any():
            logging.error(f"NaN detected in rnn_states at step {step}")
            rnn_states_actor = np.zeros_like(rnn_states_actor)

        # 获取动作
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            obs,
            rnn_states_actor,
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]))

        # 先转换为numpy数组
        values_np = _t2n(values)
        actions_np = _t2n(actions)
        action_log_probs_np = _t2n(action_log_probs)
        rnn_states_actor_np = _t2n(rnn_states_actor)
        rnn_states_critic_np = _t2n(rnn_states_critic)

        # 检查输出是否有NaN
        if np.isnan(values_np).any():
            logging.error("NaN in values output")
            values_np = np.zeros_like(values_np)
        if np.isnan(actions_np).any():
            logging.error("NaN in actions output")
            actions_np = np.zeros_like(actions_np)
        if np.isnan(action_log_probs_np).any():
            logging.error("NaN in action_log_probs output")
            action_log_probs_np = np.zeros_like(action_log_probs_np)

        # 转换为numpy数组并分割
        values = np.array(np.split(values_np, self.n_rollout_threads))
        actions = np.array(np.split(actions_np, self.n_rollout_threads))
        action_log_probs = np.array(np.split(action_log_probs_np, self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(rnn_states_actor_np, self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(rnn_states_critic_np, self.n_rollout_threads))

        # 处理自博弈对手
        if self.use_selfplay:
            opponent_actions = np.zeros_like(actions)
            if self.use_rule_opponent:
                opponent_actions = actions  # 蓝方动作由环境生成
            else:
                for policy_idx, policy in enumerate(self.opponent_policy):
                    env_idx = self.opponent_env_split[policy_idx]

                    # 检查对手观测
                    opponent_obs_batch = np.concatenate(self.opponent_obs[env_idx])
                    if np.isnan(opponent_obs_batch).any():
                        logging.warning("NaN in opponent obs, using zeros")
                        opponent_obs_batch = np.nan_to_num(opponent_obs_batch, nan=0.0)

                    opponent_action, opponent_rnn_states = policy.act(
                        opponent_obs_batch,
                        np.concatenate(self.opponent_rnn_states[env_idx]),
                        np.concatenate(self.opponent_masks[env_idx]))

                    opponent_action_np = _t2n(opponent_action)
                    opponent_rnn_states_np = _t2n(opponent_rnn_states)

                    opponent_actions[env_idx] = np.array(np.split(opponent_action_np, len(env_idx)))
                    self.opponent_rnn_states[env_idx] = np.array(np.split(opponent_rnn_states_np, len(env_idx)))

            actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    @torch.no_grad()
    def compute(self):
        """计算回报值。"""
        self.policy.prep_rollout()
        next_values = self.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def insert(self, data):
        """插入数据到缓冲区。"""
        obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
        dones = dones.squeeze(axis=-1)
        dones_env = np.all(dones, axis=-1)
        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]),
                                                       dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]),
                                                        dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if self.use_selfplay:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            self.opponent_masks = masks[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
            actions = actions[:, :self.num_agents // 2, ...]
            rewards = rewards[:, :self.num_agents // 2, ...]
            masks = masks[:, :self.num_agents // 2, ...]
            active_masks = active_masks[:, :self.num_agents // 2, ...]

        self.buffer.insert(obs, share_obs, actions, rewards, masks, action_log_probs, values,
                           rnn_states_actor, rnn_states_critic, active_masks=active_masks)

    def save(self, episode):
        """保存当前策略模型。"""
        torch.save(self.policy.actor.state_dict(), str(self.save_dir) + "/actor_latest.pt")
        torch.save(self.policy.critic.state_dict(), str(self.save_dir) + "/critic_latest.pt")
        torch.save(self.policy.actor.state_dict(), str(self.save_dir) + f"/actor_episode_{episode}.pt")
        torch.save(self.policy.critic.state_dict(), str(self.save_dir) + f"/critic_episode_{episode}.pt")

        # 保存战术统计
        np.save(str(self.save_dir) + f"/template_stats_episode_{episode}.npy", self.template_stats)

        if self.use_selfplay and not self.use_rule_opponent:
            policy_id = f"episode_{episode}"
            self.policy_pool[policy_id] = self.all_args.init_elo
            torch.save(self.policy.state_dict(), str(self.save_dir) + f"/policy_{policy_id}.pt")
            for idx, policy in enumerate(self.opponent_policy):
                available_policies = list(self.policy_pool.keys())
                selected_policy = np.random.choice(available_policies)
                policy.load_state_dict(torch.load(str(self.save_dir) + f"/policy_{selected_policy}.pt"))

        logging.info(f"Saved model and tactical stats: episode_{episode}")

    @torch.no_grad()
    def eval(self, total_num_steps):
        """评估当前策略，使用平衡的模板选择"""
        logging.info("Starting enhanced evaluation with balanced template selection...")

        # **修复模板推荐逻辑**
        original_recommendations = {}
        try:
            if hasattr(self.eval_envs, 'envs') and len(self.eval_envs.envs) > 0 and hasattr(self.eval_envs.envs[0],
                                                                                            'task'):
                task = self.eval_envs.envs[0].task
                for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']:
                    if hasattr(task, 'phase_recommended_templates'):
                        original_recommendations[agent_id] = getattr(task.phase_recommended_templates, agent_id, [0])
        except Exception as e:
            logging.error(f"Failed to save original template recommendations: {e}")
        # **设置评估时的平衡模板选择**
        balanced_templates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        try:
            if hasattr(self.eval_envs, 'envs') and len(self.eval_envs.envs) > 0 and hasattr(self.eval_envs.envs[0],
                                                                                            'task'):
                task = self.eval_envs.envs[0].task
                if not hasattr(task, 'phase_recommended_templates'):
                    task.phase_recommended_templates = {}
                for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']:
                    task.phase_recommended_templates[agent_id] = balanced_templates
        except Exception as e:
            logging.error(f"Failed to set balanced templates: {e}")
        total_episodes, eval_episode_rewards = 0, []
        eval_template_usage = np.zeros(15)
        eval_phase_performance = {}

        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_obs, eval_share_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, *self.buffer.rnn_states_actor.shape[3:]),
            dtype=np.float32)

        if self.use_selfplay and not self.use_rule_opponent:
            eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in
                                     range(self.n_choose_opponents)]
            eval_each_episodes = self.eval_episodes // self.n_choose_opponents
            eval_cur_opponent_idx = 0

        while total_episodes < self.eval_episodes:
            if self.use_selfplay and not self.use_rule_opponent and total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(
                    torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt', weights_only=True))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                eval_obs, eval_share_obs = self.eval_envs.reset()
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_rnn_states = np.zeros(
                    (self.n_eval_rollout_threads, self.num_agents, *self.buffer.rnn_states_actor.shape[3:]),
                    dtype=np.float32)
                eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
                eval_obs = eval_obs[:, :self.num_agents // 2:, ...]
                eval_opponent_masks = np.ones((self.n_eval_rollout_threads, self.num_agents // 2, 1), dtype=np.float32)
                eval_opponent_rnn_states = np.zeros(
                    (self.n_eval_rollout_threads, self.num_agents // 2, *self.buffer.rnn_states_actor.shape[3:]),
                    dtype=np.float32)

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states_ego = self.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states[:, :self.num_agents // 2, ...]),
                np.concatenate(eval_masks),
                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states_ego = np.array(np.split(_t2n(eval_rnn_states_ego), self.n_eval_rollout_threads))

            # 记录战术模板使用
            if eval_actions.size > 0 and eval_actions.ndim >= 3:
                template_ids = eval_actions[:, :, 0]
                for template_id in range(15):
                    eval_template_usage[template_id] += np.sum(template_ids == template_id)

            if self.use_selfplay and not self.use_rule_opponent:
                eval_opponent_actions, eval_opponent_rnn_states = self.eval_opponent_policy.act(
                    np.concatenate(eval_opponent_obs),
                    np.concatenate(eval_opponent_rnn_states),
                    np.concatenate(eval_opponent_masks),
                    deterministic=True)
                eval_opponent_actions = np.array(np.split(_t2n(eval_opponent_actions), self.n_eval_rollout_threads))
                eval_opponent_rnn_states = np.array(
                    np.split(_t2n(eval_opponent_rnn_states), self.n_rollout_threads))
                eval_actions = np.concatenate((eval_actions, eval_opponent_actions), axis=1)
                eval_rnn_states[:, :self.num_agents // 2, ...] = eval_rnn_states_ego
            else:
                eval_rnn_states = eval_rnn_states_ego

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            if self.use_selfplay:
                eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]

            eval_cumulative_rewards[:, :self.num_agents // 2, :] += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)

            if eval_dones_env.any():
                eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env])
                eval_cumulative_rewards[eval_dones_env] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            if eval_dones_env.any():
                eval_masks[eval_dones_env] = np.zeros((eval_dones_env.sum(), self.num_agents, 1), dtype=np.float32)
                eval_rnn_states[eval_dones_env, :self.num_agents // 2, ...] = np.zeros(
                    (eval_dones_env.sum(), self.num_agents // 2, *self.buffer.rnn_states_actor.shape[3:]),
                    dtype=np.float32)
                if self.use_selfplay and not self.use_rule_opponent:
                    eval_opponent_masks[eval_dones_env] = np.zeros(
                        (eval_dones_env.sum(), self.num_agents // 2, 1), dtype=np.float32)
                    eval_opponent_rnn_states[eval_dones_env] = np.zeros(
                        (eval_dones_env.sum(), self.num_agents // 2, *self.buffer.rnn_states_actor.shape[3:]),
                        dtype=np.float32)

            if self.use_selfplay and not self.use_rule_opponent:
                eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
                eval_obs = eval_obs[:, :self.num_agents // 2, ...]

# **修复模板推荐恢复**
        try:
            if original_recommendations and hasattr(self.eval_envs, 'envs') and len(self.eval_envs.envs) > 0 and hasattr(self.eval_envs.envs[0], 'task'):
                task = self.eval_envs.envs[0].task
                for agent_id, templates in original_recommendations.items():
                    if hasattr(task, 'phase_recommended_templates'):
                        task.phase_recommended_templates[agent_id] = templates
        except Exception as e:
            logging.error(f"Failed to restore original template recommendations: {e}")

        eval_infos = {}
        if eval_episode_rewards:
            eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean()
        else:
            eval_infos['eval_average_episode_rewards'] = 0.0
        # 战术模板评估分析
        if self.use_tactical_templates:
            total_template_usage = np.sum(eval_template_usage)
            if total_template_usage > 0:
                template_distribution = eval_template_usage / total_template_usage
                eval_infos['eval_template_diversity'] = -np.sum(
                    template_distribution * np.log(template_distribution + 1e-8)) / np.log(15)
                eval_infos['eval_most_used_template'] = np.argmax(eval_template_usage)

                template_names = [
                    "No_Template", "Crank", "Beam", "Notch", "Skate", "Short_Skate",
                    "Banzai", "Simple_F_Pole", "Advanced_F_Pole", "Pincer",
                    "Defensive_Split", "High_Low", "Engaging_Trail", "Loose_Duce", "Defensive_Sequence"
                ]

                logging.info("Evaluation Template Usage:")
                for i, usage in enumerate(eval_template_usage):
                    if usage > 0:
                        logging.info(f" {template_names[i]}: {usage} times ({usage / total_template_usage * 100:.1f}%)")
            else:
                eval_infos['eval_template_diversity'] = 0.0
                eval_infos['eval_most_used_template'] = 0

        logging.info(f"Evaluation completed - Average reward: {eval_infos['eval_average_episode_rewards']:.3f}")
        self.log_info(eval_infos, total_num_steps)

        if self.use_selfplay:
            self.reset_opponent()

    def reset_opponent(self):
        """重置对手策略和缓冲区。"""
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{choose_idx}.pt'))
            policy.prep_rollout()
        logging.info(f"Selected training opponents: {choose_opponents}")

        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)
        self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
        self.opponent_masks = np.ones_like(self.opponent_masks)

        obs, share_obs = self.envs.reset()
        if self.all_args.n_choose_opponents > 0:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    def get_tactical_insights(self):
        """获取战术洞察报告"""
        insights = {
            "template_efficiency": self._calculate_template_efficiency(),
            "most_effective_templates": [],
            "phase_performance": self.template_stats["phase_distribution"].copy(),
            "cooperation_level": self.template_stats["cooperative_efficiency"],
            "training_recommendations": []
        }

        # 找出最有效的模板
        template_names = [
            "No_Template", "Crank", "Beam", "Notch", "Skate", "Short_Skate",
            "Banzai", "Simple_F_Pole", "Advanced_F_Pole", "Pincer",
            "Defensive_Split", "High_Low", "Engaging_Trail", "Loose_Deuce", "Defensive_Sequence"
        ]

        for i in range(15):
            if self.template_stats["usage_count"][i] > 0:
                efficiency = self.template_stats["reward_per_template"][i] / max(
                    self.template_stats["usage_count"][i],
                    1e-6)
                insights["most_effective_templates"].append({
                    "id": i,
                    "name": template_names[i],
                    "efficiency": efficiency,
                    "usage": self.template_stats["usage_count"][i]
                })

        insights["most_effective_templates"].sort(key=lambda x: x["efficiency"], reverse=True)

        # 生成训练建议
        total_usage = np.sum(self.template_stats["usage_count"])
        if total_usage > 0:
            usage_distribution = self.template_stats["usage_count"] / total_usage

            # 检查模板多样性
            diversity = -np.sum(usage_distribution * np.log(usage_distribution + 1e-8)) / np.log(15)
            if diversity < 0.5:
                insights["training_recommendations"].append(
                    "Increase template diversity - consider curriculum learning")

            # 检查协同战术使用
            cooperative_templates = [9, 10, 11, 12,
                                     13]  # Pincer, Defensive_Split, High_Low, Engaging_Trail, Loose_Deuce
            cooperative_usage = np.sum([usage_distribution[i] for i in cooperative_templates])
            if cooperative_usage < 0.2:
                insights["training_recommendations"].append("Increase cooperative tactics usage")

            # 检查防御战术
            defensive_templates = [2, 3, 14]  # Beam, Notch, Defensive_Sequence
            defensive_usage = np.sum([usage_distribution[i] for i in defensive_templates])
            if defensive_usage < 0.15:
                insights["training_recommendations"].append("Improve defensive maneuvering")

        return insights