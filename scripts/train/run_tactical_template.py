import os
import logging
import argparse
import numpy as np
import yaml
import json
from datetime import datetime
from envs.JSBSim.envs import TacticalTemplateTestEnv


def setup_logging(log_level="INFO", log_file=None):
    """设置日志"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def validate_template_id(template_id):
    """验证模板ID有效性"""
    if template_id < 1 or template_id > 14:
        raise ValueError(f"Template ID must be between 1-14, got {template_id}")
    template_names = [
        "No_Template", "Crank", "Beam", "Notch", "Skate", "Short_Skate",
        "Banzai", "Simple_F_Pole", "Advanced_F_Pole", "Pincer",
        "Defensive_Split", "High_Low", "Engaging_Trail", "Loose_Deuce", "Defensive_Sequence"
    ]
    return template_names[template_id]

def run_single_template_enhanced(template_id: int, config_path: str, output_dir: str,
                                render_enabled: bool = True, max_steps: int = 2000):
    """增强版单模板测试"""
    template_name = validate_template_id(template_id)
    logging.info(f"Starting enhanced test for Template {template_id}: {template_name}")

    # 使用绝对路径确保目录正确
    output_dir = os.path.abspath(output_dir)
    template_output_dir = os.path.join(output_dir, f"template_{template_id}_{template_name}")
    os.makedirs(template_output_dir, exist_ok=True)
    final_render_path = os.path.join(template_output_dir, "Crank_final.acmi")
    logging.info(f"Output directory: {template_output_dir}, ACMI file: {final_render_path}")

    # 验证配置文件
    full_config_path = os.path.join('E:/Pycharm/LAG/envs/JSBSim/configs', f'{config_path}.yaml')
    if not os.path.exists(full_config_path):
        logging.error(f"Config file {full_config_path} does not exist")
        return None

    try:
        # 创建增强测试环境
        env = TacticalTemplateTestEnv(config_path)
        env.set_test_template(template_id)

        # 设置测试配置
        env.test_config.update({
            "enable_detailed_logging": True,
            "record_maneuver_data": True,
            "auto_analysis": True,
            "visualization_mode": "detailed"
        })

        # 重置环境
        obs, share_obs = env.reset()

        total_steps = 0
        step_data = []
        maneuver_log = []

        logging.info(f"Starting test execution for {template_name}")
        logging.info(f"Expected behavior: {env.task.test_scenarios.get(template_id, {}).get('description', 'Unknown')}")

        # 主测试循环
        while total_steps < max_steps:
            # 构造动作：强制使用测试模板
            actions = np.array([[[template_id, 0], [0, 0]]])  # [我方动作, 敌方动作(由任务控制)]

            # 执行步骤
            obs, share_obs, rewards, dones, infos = env.step(actions)
            total_steps += 1

            # 每步渲染
            if render_enabled:
                env.render("txt", final_render_path)

            # 记录详细数据
            if total_steps % 10 == 0:
                step_info = {
                    "step": total_steps,
                    "my_obs": obs[0].tolist() if len(obs) > 0 else [],
                    "rewards": rewards[0].tolist() if len(rewards) > 0 else [],
                    "test_metrics": infos.get(0, {}).get("test_metrics", {}) if isinstance(infos, dict) else {}
                }
                step_data.append(step_info)

            # 记录重要的机动事件
            if isinstance(infos, dict) and 0 in infos:
                test_metrics = infos[0].get("test_metrics", {})
                exec_count = test_metrics.get("template_execution_count", 0)
                if exec_count > len(maneuver_log):
                    maneuver_log.append({
                        "step": total_steps,
                        "execution_number": exec_count,
                        "template_id": template_id,
                        "template_name": template_name
                    })
                    logging.info(f"Step {total_steps}: Executed {template_name} maneuver #{exec_count}")

                missile_events = test_metrics.get("missile_events", [])
                if missile_events:
                    latest_event = missile_events[-1]
                    if latest_event.get("step") == total_steps:
                        logging.info(f"Step {total_steps}: Missile event - {latest_event.get('type')}")

            # 检查完成条件
            if np.any(dones):
                logging.info(f"Test completed due to termination condition at step {total_steps}")
                break

            # 检查测试目标达成
            if env._check_test_objectives_met():
                logging.info(f"Test objectives met at step {total_steps}")
                break

        # 获取测试结果
        test_results = env.get_test_results()

        # 保存详细数据
        detailed_data = {
            "template_info": {
                "id": template_id,
                "name": template_name,
                "scenario": env.task.test_scenarios.get(template_id, {})
            },
            "test_results": test_results,
            "execution_log": maneuver_log,
            "step_data": step_data[-50:],  # 保存最后50步的详细数据
            "test_summary": {
                "total_steps": total_steps,
                "completion_reason": "max_steps" if total_steps >= max_steps else "objectives_met",
                "maneuver_executions": len(maneuver_log),
                "test_timestamp": datetime.now().isoformat()
            }
        }

        # 保存结果文件
        results_file = os.path.join(template_output_dir, f"{template_name}_detailed_results.json")
        with open(results_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)

        logging.info(f"Test completed for {template_name}: {total_steps} steps, {len(maneuver_log)} executions")
        logging.info(f"Results saved to: {template_output_dir}")

        env.close()
        return detailed_data

    except Exception as e:
        logging.error(f"Error testing template {template_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def run_all_templates_enhanced(config_path: str, output_dir: str = "./enhanced_test_results",
                               render_enabled: bool = True, max_steps: int = 2000):
    """增强版全模板测试"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"batch_test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 设置批量测试日志
    batch_log_file = os.path.join(output_dir, "batch_test.log")
    setup_logging("INFO", batch_log_file)

    logging.info("=" * 80)
    logging.info("ENHANCED TACTICAL TEMPLATE BATCH TEST")
    logging.info("=" * 80)

    all_results = {}
    test_summary = {
        "start_time": datetime.now().isoformat(),
        "config": config_path,
        "max_steps": max_steps,
        "render_enabled": render_enabled,
        "templates_tested": [],
        "templates_failed": [],
        "overall_statistics": {}
    }

    # 测试模板1-14
    for template_id in range(1, 15):
        template_name = validate_template_id(template_id)

        logging.info(f"\n{'=' * 50}")
        logging.info(f"Testing Template {template_id}: {template_name}")
        logging.info(f"{'=' * 50}")

        try:
            results = run_single_template_enhanced(
                template_id, config_path, output_dir, render_enabled, max_steps
            )

            if results:
                all_results[template_id] = results
                test_summary["templates_tested"].append({
                    "id": template_id,
                    "name": template_name,
                    "status": "success",
                    "executions": results["test_summary"]["maneuver_executions"],
                    "steps": results["test_summary"]["total_steps"]
                })

                # 输出简要结果
                test_results = results["test_results"]
                print(f"\n{template_name} Test Results:")
                print(f"  Total Steps: {results['test_summary']['total_steps']}")
                print(f"  Maneuver Executions: {results['test_summary']['maneuver_executions']}")

                if "maneuver_quality" in test_results:
                    quality = test_results["maneuver_quality"].get("avg_quality_score", 0)
                    print(f"  Avg Maneuver Quality: {quality:.3f}")

                if "distance_management" in test_results:
                    distance_stats = test_results["distance_management"]
                    print(
                        f"  Distance Range: {distance_stats.get('min_distance', 0):.0f} - {distance_stats.get('max_distance', 0):.0f}m")

                if "missile_events" in test_results:
                    missile_count = test_results["missile_events"].get("total_missiles", 0)
                    print(f"  Missile Events: {missile_count}")

                # 模板特定指标
                if "template_specific_metrics" in test_results:
                    specific = test_results["template_specific_metrics"]
                    if template_id == 1 and "crank_angle_accuracy" in specific:
                        acc = specific["crank_angle_accuracy"].get("avg_accuracy", 0)
                        print(f"  Crank Angle Accuracy: {acc:.3f}")
                    elif template_id == 2 and "beam_perpendicular_accuracy" in specific:
                        acc = specific["beam_perpendicular_accuracy"].get("avg_accuracy", 0)
                        print(f"  Beam Perpendicular Accuracy: {acc:.3f}")
                    elif template_id == 3 and "notch_altitude_effectiveness" in specific:
                        eff = specific["notch_altitude_effectiveness"].get("avg_effectiveness", 0)
                        print(f"  Notch Altitude Effectiveness: {eff:.3f}")

            else:
                test_summary["templates_failed"].append({
                    "id": template_id,
                    "name": template_name,
                    "reason": "execution_failed"
                })

        except Exception as e:
            logging.error(f"Failed to test template {template_id}: {e}")
            test_summary["templates_failed"].append({
                "id": template_id,
                "name": template_name,
                "reason": str(e)
            })
            continue

    # 生成整体统计
    test_summary["end_time"] = datetime.now().isoformat()
    test_summary["total_templates"] = 14
    test_summary["successful_tests"] = len(test_summary["templates_tested"])
    test_summary["failed_tests"] = len(test_summary["templates_failed"])

    if test_summary["templates_tested"]:
        avg_executions = np.mean([t["executions"] for t in test_summary["templates_tested"]])
        avg_steps = np.mean([t["steps"] for t in test_summary["templates_tested"]])
        test_summary["overall_statistics"] = {
            "avg_executions_per_test": avg_executions,
            "avg_steps_per_test": avg_steps,
            "success_rate": test_summary["successful_tests"] / test_summary["total_templates"]
        }

    # 保存批量测试结果
    batch_results_file = os.path.join(output_dir, "batch_test_results.json")
    with open(batch_results_file, 'w') as f:
        json.dump({
            "test_summary": test_summary,
            "all_results": all_results
        }, f, indent=2)

    # 生成测试报告
    generate_test_report(output_dir, test_summary, all_results)

    logging.info(
        f"\nBatch test completed: {test_summary['successful_tests']}/{test_summary['total_templates']} templates tested successfully")
    logging.info(f"Results saved to: {output_dir}")

    return all_results


def generate_test_report(output_dir: str, test_summary: dict, all_results: dict):
    """生成测试报告"""

    report_file = os.path.join(output_dir, "TEST_REPORT.md")

    with open(report_file, 'w') as f:
        f.write("# Tactical Template Test Report\n\n")
        f.write(f"**Generated:** {test_summary['end_time']}\n\n")
        f.write(f"**Config:** {test_summary['config']}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total Templates:** {test_summary['total_templates']}\n")
        f.write(f"- **Successful Tests:** {test_summary['successful_tests']}\n")
        f.write(f"- **Failed Tests:** {test_summary['failed_tests']}\n")
        f.write(f"- **Success Rate:** {test_summary['overall_statistics'].get('success_rate', 0) * 100:.1f}%\n\n")

        if test_summary["templates_tested"]:
            f.write("## Successful Tests\n\n")
            f.write("| Template | Name | Executions | Steps | Quality Score |\n")
            f.write("|----------|------|------------|-------|---------------|\n")

            for template_info in test_summary["templates_tested"]:
                template_id = template_info["id"]
                if template_id in all_results:
                    test_results = all_results[template_id]["test_results"]
                    quality = test_results.get("maneuver_quality", {}).get("avg_quality_score", 0)
                    f.write(
                        f"| {template_id} | {template_info['name']} | {template_info['executions']} | {template_info['steps']} | {quality:.3f} |\n")

        if test_summary["templates_failed"]:
            f.write("\n## Failed Tests\n\n")
            f.write("| Template | Name | Reason |\n")
            f.write("|----------|------|--------|\n")

            for failed_info in test_summary["templates_failed"]:
                f.write(f"| {failed_info['id']} | {failed_info['name']} | {failed_info['reason']} |\n")

        f.write("\n## Detailed Analysis\n\n")

        # 分析各类模板的表现
        defensive_templates = [1, 2, 3, 14]
        offensive_templates = [4, 5, 6, 7, 8]
        cooperative_templates = [9, 10, 11, 12, 13]

        for template_ids, category_name in [
            (defensive_templates, "Defensive Maneuvers"),
            (offensive_templates, "Offensive Tactics"),
            (cooperative_templates, "Cooperative Tactics")
        ]:
            f.write(f"### {category_name}\n\n")

            category_results = [all_results[tid] for tid in template_ids if tid in all_results]
            if category_results:
                avg_quality = np.mean([r["test_results"].get("maneuver_quality", {}).get("avg_quality_score", 0)
                                       for r in category_results])
                avg_executions = np.mean([r["test_summary"]["maneuver_executions"] for r in category_results])

                f.write(f"- **Average Quality Score:** {avg_quality:.3f}\n")
                f.write(f"- **Average Executions:** {avg_executions:.1f}\n")
                f.write(f"- **Templates Tested:** {len(category_results)}/{len(template_ids)}\n\n")
            else:
                f.write("- No successful tests in this category\n\n")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Tactical Template Testing")
    parser.add_argument("--config", type=str, default="test_tactical_1v1",
                        help="Configuration file name (without .yaml)")
    parser.add_argument("--template_id", type=int, default=4,
                        help="Test specific template ID (1-14), if None test all")
    parser.add_argument("--output_dir", type=str, default="./enhanced_test_results",
                        help="Output directory for results and ACMI files")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Maximum steps per test")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable ACMI file generation")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    logging.info("Enhanced Tactical Template Testing System")
    logging.info(f"Config: {args.config}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"Max Steps: {args.max_steps}")
    logging.info(f"Render: {not args.no_render}")

    if args.template_id:
        # 单模板测试
        template_name = validate_template_id(args.template_id)
        logging.info(f"Testing single template: {args.template_id} ({template_name})")

        results = run_single_template_enhanced(
            args.template_id,
            args.config,
            args.output_dir,
            not args.no_render,
            args.max_steps
        )

        if results:
            print(f"\nEnhanced Test Results for Template {args.template_id} ({template_name}):")
            print("=" * 60)

            test_results = results["test_results"]
            for key, value in test_results.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            print(f"  {sub_key}: {sub_value:.3f}")
                        else:
                            print(f"  {sub_key}: {sub_value}")
                elif isinstance(value, (int, float)):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
        else:
            print("Test failed!")

    else:
        # 全模板测试
        logging.info("Running enhanced batch test for all templates")
        run_all_templates_enhanced(
            args.config,
            args.output_dir,
            not args.no_render,
            args.max_steps
        )


if __name__ == "__main__":
    main()