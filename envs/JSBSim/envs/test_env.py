import logging
import numpy as np
import os
from .multiplecombat_env import MultipleCombatEnv
from ..tasks.tactical_template_test_task import TacticalTemplateTestTask
from ..utils.TacviewRenderer import TacviewRenderer


class TacticalTemplateTestEnv(MultipleCombatEnv):
    """增强版战术模板测试环境 - 支持真实1v1战术测试"""

    def __init__(self, config_name: str):
        # 调试 config_name
        logging.info(f"TacticalTemplateTestEnv config_name: {config_name}, type: {type(config_name)}")

        # 确保 config_name 是字符串
        if not isinstance(config_name, str):
            raise ValueError(f"config_name must be a string, got {type(config_name)}")
        # 关键：用独立变量保存原始字符串路径，不被父类覆盖
        self.original_config_path = config_name  # 原始字符串路径

        # 其他初始化逻辑
        self.test_results = {}
        self.test_template_id = 1
        self.test_config = {
            "enable_detailed_logging": True,
            "record_maneuver_data": True,
            "auto_analysis": True,
            "visualization_mode": "detailed"
        }

        # 调用父类初始化（父类可能修改 self.config，不影响我们保存的 original_config_path）
        super().__init__(config_name)
        self.load_task()

    def load_task(self):
        """加载增强版测试任务"""
        self.task = TacticalTemplateTestTask(self.original_config_path)
        if hasattr(self, 'tactical_distances'):
            self.task.tactical_distances = self.tactical_distances
        else:
            self.task.tactical_distances = {
                "detection_range": 80000,
                "engagement_range": 60000,
                "launch_range": 40000,
                "mar_range": 20000,
                "wez_range": 32000,
                "rmax": 60000,
                "rmin": 5000
            }
        if hasattr(self, 'test_template_id'):
            self.task.test_template_id = self.test_template_id
        logging.info(f"Loaded Enhanced TacticalTemplateTestTask with template_id={self.task.test_template_id}")

    def set_test_template(self, template_id: int):
        """设置要测试的战术模板"""
        self.test_template_id = template_id
        if hasattr(self, 'task'):
            self.task.test_template_id = template_id
        logging.info(f"Set test template to {template_id}")

    def _setup_formation_relationships(self):
        """设置1v1关系"""
        available_agents = list(self._jsbsims.keys())
        logging.info(f"Setting up enhanced 1v1 relationships for agents: {available_agents}")
        if "A0100" in self._jsbsims and "B0100" in self._jsbsims:
            self._jsbsims["A0100"].partners = []
            self._jsbsims["A0100"].enemies = [self._jsbsims["B0100"]]
            self._jsbsims["B0100"].partners = []
            self._jsbsims["B0100"].enemies = [self._jsbsims["A0100"]]
            self._setup_test_initial_conditions()
            logging.info("Enhanced 1v1 relationships established: A0100 vs B0100")
        else:
            logging.error(f"Expected agents A0100 and B0100, but found: {available_agents}")
            raise ValueError(f"Invalid agent configuration for 1v1 test: {available_agents}")

    def _setup_test_initial_conditions(self):
        """根据测试模板设置初始条件"""
        if not hasattr(self, 'task') or not hasattr(self.task, 'test_scenarios'):
            return
        scenario = self.task.test_scenarios.get(self.test_template_id, {})
        initial_distance = scenario.get("initial_distance", 60000)
        enemy_heading_offset = scenario.get("enemy_heading_offset", 0)
        try:
            if "B0100" in self._jsbsims:
                if enemy_heading_offset != 0:
                    current_heading = self._jsbsims["B0100"].get_rpy()[2]
                    new_heading = current_heading + np.deg2rad(enemy_heading_offset)
                    # 假设有方法设置航向
                    # self._jsbsims["B0100"].set_property_value(Catalog.attitude_heading_true_rad, new_heading)
                logging.info(f"Setup initial conditions for template {self.test_template_id}: "
                             f"distance={initial_distance}m, enemy_offset={enemy_heading_offset}deg")
        except Exception as e:
            logging.error(f"Failed to setup initial conditions: {e}")

    def step(self, action):
        """增强版测试步骤"""
        obs, share_obs, rewards, dones, infos = super().step(action)

        # 每步渲染
        template_name = self._get_template_name(self.test_template_id)
        output_dir = os.path.abspath("./test_results")
        target_dir = os.path.join(output_dir, f"template_{self.test_template_id}_{template_name}")
        os.makedirs(target_dir, exist_ok=True)  # 确保目录存在
        filepath = os.path.join(target_dir, f"{template_name}_final.acmi")
        self.render(mode="txt", filepath=filepath)
        if any(dones.flatten()):
            logging.info(f"Step {self.current_step}: Termination triggered by dones={dones}")
        if self.current_step >= getattr(self.task, 'max_steps', 2000):
            logging.info(f"Step {self.current_step}: Termination triggered by max_steps limit")

        # 检查测试完成条件
        test_completed = (any(dones.flatten()) or
                          self.current_step >= getattr(self.task, 'max_steps', 2000) )

        if test_completed:
            self.test_results = self.task.get_test_summary()
            if self.test_config["enable_detailed_logging"]:
                self._log_detailed_test_results()
            # 关闭渲染器
            if hasattr(self, 'tacview_renderer'):
                self.tacview_renderer.close()

        return obs, share_obs, rewards, dones, infos

    def _check_test_objectives_met(self):
        """检查测试目标是否达成"""
        return False

    def _log_detailed_test_results(self):
        logging.info("=" * 80)
        logging.info("TACTICAL TEMPLATE TEST RESULTS")
        logging.info("=" * 80)
        template_name = self._get_template_name(self.test_template_id)
        logging.info(f"Template Tested: {template_name}")
        for key, value in self.test_results.items():
            if key == "analysis":
                logging.info(f"Overall Score: {value['overall_score']:.2f}/10")
                logging.info(f"Maneuver Execution: {value['maneuver_execution']:.2f}/10")
                logging.info(f"Tactical Effectiveness: {value['tactical_effectiveness']:.2f}/10")
                if value["recommendations"]:
                    logging.info("Recommendations:")
                    for rec in value["recommendations"]:
                        logging.info(f"  - {rec}")
            elif isinstance(value, dict):
                logging.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        logging.info(f"  {sub_key}: {sub_value:.3f}")
                    else:
                        logging.info(f"  {sub_key}: {sub_value}")
            elif isinstance(value, (int, float)):
                logging.info(f"{key}: {value:.3f}")
            else:
                logging.info(f"{key}: {value}")
        logging.info("=" * 80)


    def _get_template_name(self, template_id):
        """获取模板名称"""
        template_names = [
            "No_Template", "Crank", "Beam", "Notch", "Skate", "Short_Skate",
            "Banzai", "Simple_F_Pole", "Advanced_F_Pole", "Pincer",
            "Defensive_Split", "High_Low", "Engaging_Trail", "Loose_Deuce", "Defensive_Sequence"
        ]
        return template_names[template_id] if template_id < len(template_names) else "Unknown"

    def get_test_results(self):
        """获取测试结果"""
        return self.test_results

    def reset(self):
        """重置测试环境"""
        self.test_results = {}
        obs, share_obs = super().reset()
        if hasattr(self.task, 'test_template_id'):
            self._setup_test_initial_conditions()
        return obs, share_obs

    def render(self, mode="txt", filepath=None, tacview=None):
        """增强渲染 - 包含测试信息"""
        if not filepath:
            template_name = self._get_template_name(self.test_template_id)
            output_dir = os.path.abspath("./test_results")
            target_dir = os.path.join(output_dir, f"template_{self.test_template_id}_{template_name}")
            os.makedirs(target_dir, exist_ok=True)
            filepath = os.path.join(target_dir, "Crank_final.acmi")

        # 初始化渲染器（仅在第一次调用时）
        if not hasattr(self, 'tacview_renderer'):
            self.tacview_renderer = TacviewRenderer(filepath)
            self.tacview_renderer.initialize_acmi(filepath)

        super().render(mode, filepath, tacview)

        if self.test_config["visualization_mode"] == "detailed":
            self._render_test_overlay(filepath)

    def _render_test_overlay(self, filepath):
        """渲染测试覆盖信息"""
        try:
            template_name = self._get_template_name(self.test_template_id)
            base_path = filepath.replace('.acmi', '')
            test_info_path = f"{base_path}_test_info.txt"
            with open(test_info_path, 'w', encoding='utf-8') as f:
                f.write(f"Tactical Template Test: {template_name}\n")
                f.write(f"Step: {self.current_step}\n")
                f.write(f"Template ID: {self.test_template_id}\n")
                if hasattr(self.task, 'test_metrics'):
                    f.write(f"Executions: {self.task.test_metrics.get('template_execution_count', 0)}\n")
        except Exception as e:
            logging.warning(f"Failed to render test overlay: {e}")

    def close(self):
        """关闭环境"""
        if self.test_results and self.test_config["enable_detailed_logging"]:
            logging.info("FINAL TEST REPORT:")
            self._log_detailed_test_results()
        if hasattr(self, 'tacview_renderer'):
            self.tacview_renderer.close()
        super().close()