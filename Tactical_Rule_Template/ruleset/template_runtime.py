# template_runtime.py
# 统一的模板语言运行时：加载YAML -> 绑定参数 -> 每拍执行 -> 日志
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
import time, yaml, importlib, os

# ========== 数据结构 ==========
@dataclass
class ParamRange:
    low: float
    high: float
    default: float

@dataclass
class TemplateStage:
    name: str
    action: Any                   # str 或 {"type":"call","fn":"..."} / 内置原语名字
    args: Optional[Dict[str, Any]] = None
    transition: Optional[Any] = None
    avionics_ops: Optional[List[str]] = None

@dataclass
class TacticalTemplate:
    name: str
    meta: Dict[str, Any]
    params: Dict[str, Any]                 # 可能是常数或 {low,high,default}
    guards: Dict[str, Any]
    stages: List[TemplateStage]
    avionics_ops: List[str]

# ========== 加载器 ==========
def load_template(file_path: str) -> TacticalTemplate:
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    meta = data.get("meta", {})
    name = meta.get("name", os.path.splitext(os.path.basename(file_path))[0])
    params = data.get("params", {})
    guards = data.get("guards", {})
    stages = []
    for s in data.get("stages", []):
        stages.append(TemplateStage(
            name=s.get("name"),
            action=s.get("action"),
            args=s.get("args"),
            transition=s.get("transition") or s.get("transition_condition"),
            avionics_ops=s.get("avionics_ops", [])
        ))
    return TacticalTemplate(
        name=name,
        meta=meta,
        params=params,
        guards=guards,
        stages=stages,
        avionics_ops=data.get("avionics_ops", [])
    )

# ========== 参数绑定（支持 RL 覆盖） ==========
def bind_params(param_spec: Dict[str, Any], overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    bound = {}
    for k, spec in param_spec.items():
        if isinstance(spec, dict) and {"low","high","default"} <= set(spec.keys()):
            v = spec.get("default")
            if overrides and k in overrides:
                v = overrides[k]
            v = float(max(spec["low"], min(spec["high"], float(v))))
            bound[k] = v
        else:
            bound[k] = float(spec)
    return bound

def rl_vec_to_overrides(param_spec: Dict[str,Any], keys_order: List[str], z: List[float]) -> Dict[str,float]:
    out = {}
    for k, zi in zip(keys_order, z):
        spec = param_spec[k]
        lo, hi = float(spec["low"]), float(spec["high"])
        out[k] = lo + (zi + 1.0)*0.5*(hi - lo)   # z∈[-1,1] -> [lo,hi]
    return out

# ========== 简单日志 ==========
class TemplateLogger:
    def __init__(self):
        self.stage_trans = []   # (step_or_time, "Enter stage: X")
        self.avionics = []      # (step_or_time, "op: radar_on")
    def log_stage(self, name): self.stage_trans.append((time.time(), f"Enter stage: {name}"))
    def log_avionics(self, op): self.avionics.append((time.time(), f"op: {op}"))
    def dump(self): return {"stages": self.stage_trans, "avionics": self.avionics}

# ========== 执行器 ==========
def _resolve_value(v, params: Dict[str,float], sensor: Dict[str,Any]):
    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
        key = v[2:-1]
        return params.get(key, sensor.get(key))
    return v

def _resolve_args(d, params, sensor):
    if not d: return {}
    return {k: _resolve_value(v, params, sensor) for k,v in d.items()}

def _sensor_snapshot(env, agent_id, radar):
    ego = env.agents[agent_id]
    enemy = ego.enemies[0] if ego.enemies else None
    rel_dist = 1e9; locked = False
    if enemy:
        ep = ego.get_position(); tp = enemy.get_position()
        dx,dy,dz = tp[0]-ep[0], tp[1]-ep[1], tp[2]-ep[2]
        rel_dist = (dx*dx+dy*dy+dz*dz)**0.5
        locked = True   # 简化：若需要严格雷达锁，请改接你的雷达模块
    incoming = list(getattr(ego,"under_missiles", []))
    return {"rel_dist": rel_dist, "locked": locked, "incoming": incoming}

class TemplateSession:
    """
    一次模板实例（可跨多拍执行）。每拍调用 step()，返回 [alt,heading,vel,shoot]。
    """
    def __init__(self, template: TacticalTemplate, env, agent_id, radar, ctx=None,
                 action_module: str = "tactical_templates_v2", param_overrides: Dict[str,float]=None):
        self.tpl = template
        self.env, self.agent_id, self.radar, self.ctx = env, agent_id, radar, (ctx or {})
        self.params = bind_params(template.params, param_overrides)
        self.stage_idx = 0
        self.logger = TemplateLogger()

        # —— 仅改这里：先尝试绝对导入；失败则按“当前包”相对导入 —— #
        try:
            self.mod = importlib.import_module(action_module)
        except ModuleNotFoundError:
            base_pkg = __package__ or "Tactical_Rule_Template.ruleset"
            self.mod = importlib.import_module(f".{action_module.lstrip('.')}", package=base_pkg)

        if self.tpl.stages:
            self.logger.log_stage(self.tpl.stages[0].name)
            for op in (self.tpl.stages[0].avionics_ops or []):
                self._do_avionics(op)

    def _do_avionics(self, op):
        # 这里先只记日志，需要接设备就自己扩展
        self.logger.log_avionics(op)

    def step(self):
        if self.stage_idx >= len(self.tpl.stages):
            return [1,2,1,0]  # 完成后返回一个“保持”
        s = self.tpl.stages[self.stage_idx]
        sensor = _sensor_snapshot(self.env, self.agent_id, self.radar)
        # 调动作函数
        action = [1,2,1,0]
        if isinstance(s.action, str):
            fn = getattr(self.mod, s.action, None)
            if callable(fn):
                # 兼容有/无 ctx 的旧函数
                try:
                    action = fn(self.env, self.agent_id, self.radar, self.ctx)
                except TypeError:
                    action = fn(self.env, self.agent_id, self.radar)
        elif isinstance(s.action, dict) and s.action.get("type") == "call":
            fn = getattr(self.mod, s.action["fn"], None)
            args = _resolve_args(s.args, self.params, sensor)
            if callable(fn):
                # 尽量把 ctx 放上；如果签名不接收 kwargs，则只传必须参数
                try:
                    action = fn(self.env, self.agent_id, self.radar, self.ctx, **args)
                except TypeError:
                    try:
                        action = fn(self.env, self.agent_id, self.radar, **args)
                    except TypeError:
                        action = fn(self.env, self.agent_id, self.radar)
        else:
            # 可在此实现内置原语: turn_cold / extend_cold / press_in 等
            pass

        # 阶段转移（最小化：仅支持表达式/布尔；复杂组合可后续加）
        if s.transition:
            cond_expr = None
            if isinstance(s.transition, str):
                # 允许引用 guards 名
                cond_expr = self.tpl.guards.get(s.transition, s.transition)
            else:
                cond_expr = s.transition
            if isinstance(cond_expr, str):
                # 安全上下文：只允许 params/sensor
                ctx = {}; ctx.update(self.params); ctx.update(sensor)
                try:
                    ok = eval(cond_expr, {"__builtins__": None}, ctx)
                except Exception:
                    ok = False
            elif isinstance(cond_expr, bool):
                ok = cond_expr
            else:
                ok = False
            if ok:
                self.stage_idx += 1
                if self.stage_idx < len(self.tpl.stages):
                    ns = self.tpl.stages[self.stage_idx]
                    self.logger.log_stage(ns.name)
                    for op in (ns.avionics_ops or []):
                        self._do_avionics(op)

        return action

    def finished(self):
        return self.stage_idx >= len(self.tpl.stages)

    def logs(self):
        return self.logger.dump()

# 便捷函数：一步调用（无状态执行可用）
_sessions: Dict[str, TemplateSession] = {}

def run_yaml_step(template_file: str, env, agent_id, radar, ctx=None,
                  action_module: str = "tactical_templates_v2",
                  param_overrides: Dict[str,float]=None):
    key = f"{agent_id}:{template_file}"
    sess = _sessions.get(key)
    if sess is None or sess.finished():
        tpl = load_template(template_file)
        sess = TemplateSession(tpl, env, agent_id, radar, ctx=ctx,
                               action_module=action_module,
                               param_overrides=param_overrides)
        _sessions[key] = sess
    return sess.step()