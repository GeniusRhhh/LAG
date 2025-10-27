#!/usr/bin/env python
"""
评估SU-27 Baseline模型

使用方法：
python evaluate_su27_model.py --model_path results/.../actor_100.pt --episodes 10
"""
import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from envs.JSBSim.envs import SingleControlEnv
from algorithms.ppo.ppo_actor import PPOActor
from gymnasium import spaces

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_model(model_path, num_episodes=10, render_acmi=False):
    """
    评估训练好的模型
    
    Args:
        model_path: 模型文件路径（actor_xxx.pt）
        num_episodes: 评估的episode数量
        render_acmi: 是否生成ACMI文件
    """
    logging.info("=" * 80)
    logging.info("🧪 SU-27 Baseline模型评估")
    logging.info("=" * 80)
    logging.info(f"模型路径: {model_path}")
    logging.info(f"评估episodes: {num_episodes}")
    logging.info("=" * 80)
    
    # 1. 加载模型
    logging.info("\n1️⃣  加载模型...")
    
    # 创建PPOActor需要的参数（需要匹配训练时的配置）
    class Args:
        # 网络结构参数
        hidden_size = "128 128"
        act_hidden_size = "128 128"
        activation_id = 1  # ReLU
        use_feature_normalization = False
        gain = 0.01
        
        # RNN参数
        use_recurrent_policy = True
        recurrent_hidden_size = 128
        recurrent_hidden_layers = 1
        use_naive_recurrent_policy = False
        
        # 动作层参数
        use_mlp_actlayer = True
        
        # 其他参数
        use_prior = False
    
    args = Args()
    obs_space = spaces.Box(low=-10, high=10., shape=(12,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([41, 41, 41, 30])
    
    try:
        actor = PPOActor(args, obs_space, act_space, device=torch.device('cpu'))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        actor.load_state_dict(checkpoint)
        actor.eval()
        logging.info("✅ 模型加载成功")
    except Exception as e:
        logging.error(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 2. 创建环境
    logging.info("\n2️⃣  创建评估环境...")
    env = SingleControlEnv(config_name='1/heading_su27')
    logging.info("✅ 环境创建成功")
    
    # 3. 运行评估
    logging.info(f"\n3️⃣  开始评估（{num_episodes} episodes）...")
    
    results = {
        'rewards': [],
        'episode_lengths': [],
        'success_count': 0,
        'crash_count': 0,
        'heading_errors': [],
        'altitude_deviations': []
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        rnn_states_actor = np.zeros((1, 1, 128))
        masks = np.ones((1, 1))
        
        episode_reward = 0
        episode_length = 0
        crashed = False
        max_steps = 1000
        
        # ACMI文件（可选）
        if render_acmi:
            acmi_path = f"eval_su27_ep{episode}.acmi"
            env.render(mode='txt', filepath=acmi_path)
        
        for step in range(max_steps):
            # 使用模型预测动作
            with torch.no_grad():
                actor.eval()
                action, action_log_probs, rnn_states_actor = actor(
                    torch.FloatTensor(obs),
                    torch.FloatTensor(rnn_states_actor),
                    torch.FloatTensor(masks),
                    deterministic=True  # 评估时使用确定性动作
                )
                action = action.cpu().numpy()
            
            # 转换为环境格式 (1, 1, 4)
            action_env = action.reshape(1, 1, -1)
            
            # 执行动作
            obs, reward, done, info = env.step(action_env)
            
            if render_acmi:
                env.render(mode='txt', filepath=acmi_path)
            
            episode_reward += reward.item() if isinstance(reward, np.ndarray) else reward
            episode_length += 1
            
            # 检查是否完成
            if isinstance(done, np.ndarray):
                done = done.item()
            
            if done:
                # 检查是否坠毁
                agent_id = list(env.agents.keys())[0]
                agent = env.agents[agent_id]
                altitude = agent.get_position()[2]
                
                if altitude < 2500:  # 坠毁
                    crashed = True
                    results['crash_count'] += 1
                else:
                    results['success_count'] += 1
                
                break
        
        results['rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        
        status = "❌ 坠毁" if crashed else "✅ 成功"
        logging.info(f"  Episode {episode+1}/{num_episodes}: {status} | "
                    f"Reward={episode_reward:.2f} | Length={episode_length} steps")
    
    env.close()
    
    # 4. 统计结果
    logging.info("\n" + "=" * 80)
    logging.info("📊 评估结果")
    logging.info("=" * 80)
    
    avg_reward = np.mean(results['rewards'])
    std_reward = np.std(results['rewards'])
    avg_length = np.mean(results['episode_lengths'])
    success_rate = results['success_count'] / num_episodes * 100
    
    logging.info(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    logging.info(f"平均长度: {avg_length:.1f} steps")
    logging.info(f"成功率: {success_rate:.1f}% ({results['success_count']}/{num_episodes})")
    logging.info(f"坠毁次数: {results['crash_count']}/{num_episodes}")
    logging.info(f"最高奖励: {max(results['rewards']):.2f}")
    logging.info(f"最低奖励: {min(results['rewards']):.2f}")
    
    # 5. 评估等级
    logging.info("\n" + "=" * 80)
    if avg_reward > 30 and success_rate > 90:
        grade = "🌟 优秀"
        comment = "模型已经训练得很好，可以使用了！"
    elif avg_reward > 0 and success_rate > 70:
        grade = "👍 良好"
        comment = "模型基本可用，建议继续训练以提升性能。"
    elif avg_reward > -50 and success_rate > 50:
        grade = "⚠️  一般"
        comment = "模型还在学习中，建议继续训练。"
    else:
        grade = "❌ 较差"
        comment = "模型训练不足，需要更多训练时间。"
    
    logging.info(f"模型评级: {grade}")
    logging.info(f"建议: {comment}")
    logging.info("=" * 80)
    
    return results


def compare_models(model_paths, num_episodes=10):
    """
    对比多个模型的性能
    
    Args:
        model_paths: 模型文件路径列表
        num_episodes: 每个模型评估的episode数量
    """
    logging.info("=" * 80)
    logging.info("📊 模型对比评估")
    logging.info("=" * 80)
    
    all_results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem  # 例如：actor_100
        logging.info(f"\n评估模型: {model_name}")
        results = evaluate_model(model_path, num_episodes, render_acmi=False)
        if results:
            all_results[model_name] = results
    
    # 对比结果
    logging.info("\n" + "=" * 80)
    logging.info("📈 模型对比结果")
    logging.info("=" * 80)
    logging.info(f"{'模型':<15} {'平均奖励':<12} {'成功率':<10} {'平均长度':<10}")
    logging.info("-" * 80)
    
    for model_name, results in all_results.items():
        avg_reward = np.mean(results['rewards'])
        success_rate = results['success_count'] / num_episodes * 100
        avg_length = np.mean(results['episode_lengths'])
        logging.info(f"{model_name:<15} {avg_reward:>10.2f}  {success_rate:>8.1f}%  {avg_length:>8.1f}")
    
    # 找出最佳模型
    best_model = max(all_results.items(), key=lambda x: np.mean(x[1]['rewards']))
    logging.info("\n" + "=" * 80)
    logging.info(f"🏆 最佳模型: {best_model[0]}")
    logging.info(f"   平均奖励: {np.mean(best_model[1]['rewards']):.2f}")
    logging.info("=" * 80)


def find_best_model(results_dir):
    """
    自动找到results目录中最好的模型
    
    Args:
        results_dir: results目录路径
    """
    results_path = Path(results_dir)
    
    # 查找所有actor_*.pt文件
    actor_files = list(results_path.rglob("actor_*.pt"))
    actor_files = [f for f in actor_files if f.stem != "actor_latest"]
    
    if not actor_files:
        logging.error("❌ 未找到任何模型文件")
        return None
    
    logging.info(f"找到 {len(actor_files)} 个模型文件")
    
    # 按episode数排序
    actor_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    
    # 显示所有模型
    logging.info("\n可用的模型:")
    for i, f in enumerate(actor_files[-10:], 1):  # 只显示最后10个
        logging.info(f"  {i}. {f.relative_to(results_path)}")
    
    return actor_files


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估SU-27 Baseline模型')
    parser.add_argument('--model_path', type=str, help='模型文件路径')
    parser.add_argument('--results_dir', type=str,
                        default='C:\\Users\\ZRF\\PycharmProjects\\LAG\\scripts\\results\\SU27_Baseline\\1\\heading_su27\\ppo\\check\\run2',
                       #default='../../results/SU27_Baseline/1/heading_su27/ppo/check/run2',
                       help='results目录路径')
    parser.add_argument('--episodes', type=int, default=10,
                       help='评估的episode数量')
    parser.add_argument('--render', action='store_true',
                       help='是否生成ACMI文件')
    parser.add_argument('--compare', action='store_true',
                       help='对比多个模型')
    parser.add_argument('--auto', action='store_true',
                       help='自动找到并评估最新的模型')
    
    args = parser.parse_args()
    
    if args.auto:
        # 自动模式：找到最新的模型并评估
        models = find_best_model(args.results_dir)
        if models:
            latest_model = models[-1]
            logging.info(f"\n使用最新模型: {latest_model}")
            evaluate_model(str(latest_model), args.episodes, args.render)
    
    elif args.compare:
        # 对比模式：对比最后5个模型
        models = find_best_model(args.results_dir)
        if models and len(models) >= 5:
            compare_models([str(m) for m in models[-5:]], args.episodes)
        else:
            logging.error("模型数量不足，无法对比")
    
    elif args.model_path:
        # 单模型评估
        evaluate_model(args.model_path, args.episodes, args.render)
    
    else:
        # 交互模式
        models = find_best_model(args.results_dir)
        if models:
            print("\n请选择要评估的模型:")
            print("1. 评估最新模型")
            print("2. 对比最后5个模型")
            print("3. 手动输入模型路径")
            
            choice = input("\n请输入选项 (1/2/3): ").strip()
            
            if choice == '1':
                evaluate_model(str(models[-1]), args.episodes, args.render)
            elif choice == '2':
                if len(models) >= 5:
                    compare_models([str(m) for m in models[-5:]], args.episodes)
                else:
                    logging.warning("模型数量不足5个，评估所有模型")
                    compare_models([str(m) for m in models], args.episodes)
            elif choice == '3':
                model_path = input("请输入模型路径: ").strip()
                evaluate_model(model_path, args.episodes, args.render)
