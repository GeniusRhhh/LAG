from .termination_condition_base import BaseTerminationCondition


class RecoverySuccess(BaseTerminationCondition):
    """Terminate an episode once the aircraft has stayed inside a stable envelope."""

    def __init__(self, config):
        super().__init__(config)
        self.required_stable_steps = int(getattr(config, "RecoverySuccess_required_stable_steps", 30))

    def get_termination(self, task, env, agent_id, info={}):
        stable_steps = int(getattr(task, "_stable_steps", {}).get(agent_id, 0))
        done = stable_steps >= self.required_stable_steps
        if done:
            info["recovery_success"] = True
            info["recovery_stable_steps"] = stable_steps
            self.log(f"{agent_id} recovered successfully. Total Steps={env.current_step}")
        success = done
        return done, success, info
