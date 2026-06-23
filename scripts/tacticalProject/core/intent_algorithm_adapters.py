"""Intent recognition algorithm adapters."""

import logging
import os
import re
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from utils.trace_logger import trace_event
except Exception:  # pragma: no cover
    trace_event = None


LOG = logging.getLogger(__name__)

_STATUS_CODE = {
    "unknown": 0,
    "search": 1,
    "lock_on": 2,
    "track": 3,
}


def _aircraft_key(ac: Any) -> str:
    for attr in ("agent_id", "callsign", "name", "uid", "id"):
        try:
            value = getattr(ac, attr, None)
        except Exception:
            value = None
        if value:
            return str(value)
    return f"obj_{id(ac)}"


def _safe_get_pos(ac: Any) -> np.ndarray:
    try:
        return np.asarray(ac.get_position(), dtype=np.float64)
    except Exception:
        return np.zeros(3, dtype=np.float64)


def _safe_get_vel(ac: Any) -> np.ndarray:
    try:
        return np.asarray(ac.get_velocity(), dtype=np.float64)
    except Exception:
        return np.zeros(3, dtype=np.float64)


def _resolve_agent_id(ac: Any, env: Any = None) -> str:
    key = _aircraft_key(ac)
    if not key.startswith("obj_"):
        return key
    agents = getattr(env, "agents", {}) or {}
    for agent_id, sim in agents.items():
        if sim is ac:
            return str(agent_id)
    return key


def _pairmate_id(agent_id: str) -> Optional[str]:
    match = re.match(r"^([A-Za-z]+)(\d{4})$", str(agent_id))
    if not match:
        return None
    prefix, digits = match.groups()
    if not digits.endswith("00"):
        return None
    flight_index = int(digits[:-2])
    mate_index = flight_index + 1 if flight_index % 2 == 1 else flight_index - 1
    if mate_index <= 0:
        return None
    return f"{prefix}{mate_index:02d}00"


def _normalise_radar_mode(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    if text in {"lock", "lock_on", "stt"}:
        return "lock_on"
    if text in {"track", "tws"}:
        return "track"
    if text in {"search"}:
        return "search"
    return "unknown"


def _status_code_from_aircraft(ac: Any) -> int:
    mode = _normalise_radar_mode(getattr(ac, "radar_mode", None))
    return _STATUS_CODE.get(mode, 0)


class IntentLSTM(nn.Module):
    """Runtime model that matches bvr_intent_new training script."""

    def __init__(
        self,
        input_dim_num: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        use_status_embed: bool = False,
        status_vocab_size: int = 4,
        status_emb_dim: int = 4,
        bidirectional: bool = False,
        use_self_attn: bool = False,
        attn_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.use_status = use_status_embed
        self.use_self_attn = use_self_attn
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        feat_in = input_dim_num
        if self.use_status:
            self.status_emb = nn.Embedding(status_vocab_size, status_emb_dim)
            feat_in += status_emb_dim
        else:
            self.status_emb = None

        self.pre_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=feat_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(0.0 if num_layers == 1 else dropout),
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_directions
        self.ln = nn.LayerNorm(lstm_out_dim) if use_layernorm else nn.Identity()

        if self.use_self_attn:
            if attn_dim is None:
                attn_dim = lstm_out_dim
            self.attn_W = nn.Linear(lstm_out_dim, attn_dim, bias=True)
            self.attn_u = nn.Linear(attn_dim, 1, bias=False)
            self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc = nn.Linear(lstm_out_dim, num_classes)
        else:
            self.post_dropout = nn.Identity()
            self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x_num: torch.Tensor, x_status: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_status and x_status is not None:
            emb = self.status_emb(x_status)
            x = torch.cat([x_num, emb], dim=-1)
        else:
            x = x_num

        x = self.pre_dropout(x)
        hidden, _ = self.lstm(x)

        if self.use_self_attn:
            mid = torch.tanh(self.attn_W(hidden))
            scores = self.attn_u(mid).squeeze(-1)
            alpha = torch.softmax(scores, dim=1)
            context = torch.sum(hidden * alpha.unsqueeze(-1), dim=1)
            context = self.ln(context)
            context = self.post_dropout(context)
            return self.fc(context)

        last = hidden[:, -1, :]
        last = self.ln(last)
        return self.fc(last)


class _BaseBVRIntentAdapter:
    def __init__(self, name: str, default_weight: Optional[Path] = None):
        self.name = name
        self.default_weight = default_weight
        self._warned_missing = False
        self._last_infer_info: Dict[str, Dict[str, Any]] = {}

    def get_last_infer_info(self, enemy_aircraft: Any) -> Optional[Dict[str, Any]]:
        return self._last_infer_info.get(_aircraft_key(enemy_aircraft))

    def _default_intent(self, enemy_aircraft: Any, my_aircraft: Any, env: Any) -> str:
        return "NEUTRAL"

    def recognize(self, enemy_aircraft: Any, my_aircraft: Any, env: Any) -> str:
        try:
            predicted = self._predict(enemy_aircraft, my_aircraft, env)
            if predicted is not None:
                return predicted
        except Exception as exc:  # pragma: no cover
            key = _aircraft_key(enemy_aircraft)
            LOG.error("%s inference failed for %s: %s", self.name, key, exc, exc_info=True)
            if trace_event is not None:
                trace_event(
                    事件="意图识别-模型推理异常",
                    env=env,
                    模块="intent_algorithm_adapters",
                    类型="INTENT",
                    状态="EXCEPTION",
                    我机=_aircraft_key(my_aircraft),
                    敌机=key,
                    说明=f"{self.name} 推理失败，回退规则算法",
                    数据={"exception": str(exc)},
                )
        return self._default_intent(enemy_aircraft, my_aircraft, env)

    def _predict(self, enemy_aircraft: Any, my_aircraft: Any, env: Any) -> Optional[str]:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight_path": str(self.default_weight) if self.default_weight else None,
            "ready": bool(self.default_weight and self.default_weight.exists()),
        }


class BVRIntentNewAdapter(_BaseBVRIntentAdapter):
    """Online adapter for IntentRecognition/bvr_intent_new."""

    def __init__(self, weight_path: Optional[Path] = None, window_size: int = 32):
        env_path = os.getenv("BVR_INTENT_NEW_WEIGHT")
        default_weight = Path(env_path) if env_path else weight_path
        if default_weight is None:
            default_weight = (
                Path(__file__).resolve().parent.parent
                / "IntentRecognition"
                / "bvr_intent_new"
                / "runtime"
                / "intent_lstm_best.pth"
            )
        super().__init__("bvr_intent_new", default_weight)

        try:
            env_window = os.getenv("BVR_INTENT_NEW_WINDOW")
            self.window_size = int(env_window) if env_window else int(window_size)
        except Exception:
            self.window_size = int(window_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[IntentLSTM] = None
        self.use_status_embed = False
        self.status_vocab_size = 4
        self.status_emb_dim = 4
        self.label2id: Dict[str, int] = {}
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.cont_dim_scaled: Optional[int] = None
        self.has_wing_mask = True
        self.scale_wing_mask = False
        self._buffers: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=max(96, self.window_size * 3))
        )
        self._warmup_logged = set()
        self._model_missing_log_step: Dict[str, int] = {}
        self._feature_fail_log_step: Dict[str, int] = {}
        self._last_intent: Dict[str, str] = {}
        self._infer_log_count: Dict[str, int] = defaultdict(int)

    def _iter_meta_paths(self):
        env_meta = os.getenv("BVR_INTENT_NEW_META")
        if env_meta:
            yield Path(env_meta)

        if not self.default_weight:
            return

        runtime_dir = self.default_weight.parent
        project_dir = runtime_dir.parent

        yield runtime_dir / "meta.npy"
        for meta_path in sorted(project_dir.glob("npz_runtime_t*/meta.npy")):
            yield meta_path

    def _load_sidecar_meta(self) -> Dict[str, Any]:
        seen = set()
        for meta_path in self._iter_meta_paths():
            meta_path = Path(meta_path)
            if meta_path in seen or not meta_path.exists():
                continue
            seen.add(meta_path)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = np.load(meta_path, allow_pickle=True).item()
            except Exception as exc:
                LOG.warning("Failed to load bvr_intent_new meta %s: %s", meta_path, exc)
                continue

            meta: Dict[str, Any] = {
                "meta_path": str(meta_path),
                "label2id": dict(raw.get("label2id") or {}),
                "input_dim_num": int(raw.get("input_dim_num", 0) or 0),
                "use_status_embed": bool(raw.get("use_status_embed", False)),
                "has_wing_mask": bool(raw.get("has_wing_mask", True)),
                "scale_wing_mask": bool(raw.get("scale_wing_mask", False)),
            }

            cont_dim = raw.get("cont_dim_scaled")
            if cont_dim is not None:
                meta["cont_dim_scaled"] = int(cont_dim)

            scaler_mean = raw.get("scaler_mean")
            scaler_scale = raw.get("scaler_scale")
            scaler = raw.get("scaler")
            if scaler_mean is None and scaler is not None and hasattr(scaler, "mean_"):
                scaler_mean = scaler.mean_
            if scaler_scale is None and scaler is not None and hasattr(scaler, "scale_"):
                scaler_scale = scaler.scale_

            if scaler_mean is not None:
                meta["scaler_mean"] = np.asarray(scaler_mean, dtype=np.float32)
            if scaler_scale is not None:
                meta["scaler_scale"] = np.asarray(scaler_scale, dtype=np.float32)

            if meta["input_dim_num"] <= 0 and meta.get("scaler_mean") is not None:
                input_dim = int(len(meta["scaler_mean"]))
                if meta.get("has_wing_mask", True):
                    input_dim += 1
                meta["input_dim_num"] = input_dim

            return meta

        return {}

    def _infer_checkpoint_config(
        self,
        ckpt: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
        sidecar_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        label2id = dict(ckpt.get("label2id") or sidecar_meta.get("label2id") or {})

        ih0 = state_dict.get("lstm.weight_ih_l0")
        hh0 = state_dict.get("lstm.weight_hh_l0")
        fc_weight = state_dict.get("fc.weight")
        status_weight = state_dict.get("status_emb.weight")

        lstm_ih_keys = [key for key in state_dict if re.match(r"^lstm\.weight_ih_l\d+(?:_reverse)?$", key)]
        layer_indices = []
        for key in lstm_ih_keys:
            match = re.match(r"^lstm\.weight_ih_l(\d+)", key)
            if match:
                layer_indices.append(int(match.group(1)))

        bidirectional = bool(ckpt.get("bidirectional", any(key.endswith("_reverse") for key in lstm_ih_keys)))
        num_directions = 2 if bidirectional else 1

        hidden_dim = ckpt.get("hidden_dim")
        if hidden_dim is None:
            if hh0 is not None:
                hidden_dim = int(hh0.shape[1])
            elif fc_weight is not None:
                hidden_dim = int(fc_weight.shape[1] // num_directions)
            else:
                hidden_dim = 64

        use_status_embed = ckpt.get("use_status_embed")
        if use_status_embed is None:
            use_status_embed = bool(sidecar_meta.get("use_status_embed", False) or status_weight is not None)
        use_status_embed = bool(use_status_embed)

        status_vocab_size = ckpt.get("status_vocab_size")
        if status_vocab_size in (None, 0) and status_weight is not None:
            status_vocab_size = int(status_weight.shape[0])
        if status_vocab_size in (None, 0):
            status_vocab_size = 4 if use_status_embed else 0

        status_emb_dim = ckpt.get("status_emb_dim")
        if status_emb_dim in (None, 0) and status_weight is not None:
            status_emb_dim = int(status_weight.shape[1])
        if status_emb_dim in (None, 0):
            status_emb_dim = 4 if use_status_embed else 0

        input_dim = ckpt.get("input_dim_num")
        if input_dim in (None, 0):
            input_dim = sidecar_meta.get("input_dim_num")
        if input_dim in (None, 0) and ih0 is not None:
            input_dim = int(ih0.shape[1]) - (int(status_emb_dim) if use_status_embed else 0)
        input_dim = int(input_dim or 13)

        num_classes = ckpt.get("num_classes")
        if num_classes in (None, 0) and fc_weight is not None:
            num_classes = int(fc_weight.shape[0])
        if num_classes in (None, 0):
            num_classes = max(len(label2id), 3)
        num_classes = int(num_classes)

        num_layers = ckpt.get("num_layers")
        if num_layers in (None, 0):
            num_layers = max(layer_indices) + 1 if layer_indices else 1
        num_layers = int(num_layers)

        use_self_attn = ckpt.get("use_self_attn")
        if use_self_attn is None:
            use_self_attn = "attn_W.weight" in state_dict and "attn_u.weight" in state_dict
        use_self_attn = bool(use_self_attn)

        attn_dim = ckpt.get("attn_dim")
        if attn_dim is None and "attn_W.weight" in state_dict:
            attn_dim = int(state_dict["attn_W.weight"].shape[0])

        use_layernorm = ckpt.get("use_layernorm")
        if use_layernorm is None:
            use_layernorm = "ln.weight" in state_dict and "ln.bias" in state_dict
        use_layernorm = bool(use_layernorm)

        window_size = ckpt.get("window_size")
        if window_size in (None, 0):
            meta_path = str(sidecar_meta.get("meta_path", ""))
            match = re.search(r"npz_runtime_t(\d+)", meta_path)
            if match:
                window_size = int(match.group(1))
        window_size = int(window_size or self.window_size)

        scaler_mean = ckpt.get("scaler_mean")
        if scaler_mean is None:
            scaler_mean = sidecar_meta.get("scaler_mean")
        scaler_scale = ckpt.get("scaler_scale")
        if scaler_scale is None:
            scaler_scale = sidecar_meta.get("scaler_scale")

        cont_dim_scaled = ckpt.get("cont_dim_scaled")
        if cont_dim_scaled is None:
            cont_dim_scaled = sidecar_meta.get("cont_dim_scaled")

        return {
            "label2id": label2id,
            "input_dim_num": input_dim,
            "num_classes": num_classes,
            "hidden_dim": int(hidden_dim),
            "num_layers": num_layers,
            "use_status_embed": use_status_embed,
            "status_vocab_size": int(status_vocab_size),
            "status_emb_dim": int(status_emb_dim),
            "bidirectional": bidirectional,
            "use_self_attn": use_self_attn,
            "attn_dim": (None if attn_dim is None else int(attn_dim)),
            "dropout": float(ckpt.get("dropout", 0.0) or 0.0),
            "use_layernorm": use_layernorm,
            "window_size": window_size,
            "scaler_mean": (None if scaler_mean is None else np.asarray(scaler_mean, dtype=np.float32)),
            "scaler_scale": (None if scaler_scale is None else np.asarray(scaler_scale, dtype=np.float32)),
            "cont_dim_scaled": (None if cont_dim_scaled is None else int(cont_dim_scaled)),
            "has_wing_mask": bool(ckpt.get("has_wing_mask", sidecar_meta.get("has_wing_mask", True))),
            "scale_wing_mask": bool(ckpt.get("scale_wing_mask", sidecar_meta.get("scale_wing_mask", False))),
        }

    def _needs_checkpoint_repair(self, ckpt: Dict[str, Any]) -> bool:
        required = {
            "label2id",
            "input_dim_num",
            "hidden_dim",
            "num_classes",
            "num_layers",
            "window_size",
            "use_status_embed",
            "status_vocab_size",
            "status_emb_dim",
            "bidirectional",
            "use_self_attn",
            "dropout",
            "use_layernorm",
        }
        return any(key not in ckpt for key in required)

    def _rewrite_checkpoint(
        self,
        ckpt: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
        config: Dict[str, Any],
    ) -> None:
        if not self.default_weight:
            return

        repaired = {
            "model_state": state_dict,
            "label2id": dict(config["label2id"]),
            "input_dim_num": int(config["input_dim_num"]),
            "hidden_dim": int(config["hidden_dim"]),
            "num_classes": int(config["num_classes"]),
            "num_layers": int(config["num_layers"]),
            "window_size": int(config["window_size"]),
            "use_status_embed": bool(config["use_status_embed"]),
            "status_vocab_size": int(config["status_vocab_size"] if config["use_status_embed"] else 0),
            "status_emb_dim": int(config["status_emb_dim"] if config["use_status_embed"] else 0),
            "bidirectional": bool(config["bidirectional"]),
            "use_self_attn": bool(config["use_self_attn"]),
            "attn_dim": config["attn_dim"],
            "dropout": float(config["dropout"]),
            "use_layernorm": bool(config["use_layernorm"]),
            "has_wing_mask": bool(config["has_wing_mask"]),
            "scale_wing_mask": bool(config["scale_wing_mask"]),
        }
        if ckpt.get("best_val_macro_f1") is not None:
            repaired["best_val_macro_f1"] = float(ckpt["best_val_macro_f1"])
        if ckpt.get("epoch") is not None:
            repaired["epoch"] = int(ckpt["epoch"])
        if config.get("scaler_mean") is not None:
            repaired["scaler_mean"] = np.asarray(config["scaler_mean"], dtype=np.float32)
        if config.get("scaler_scale") is not None:
            repaired["scaler_scale"] = np.asarray(config["scaler_scale"], dtype=np.float32)
        if config.get("cont_dim_scaled") is not None:
            repaired["cont_dim_scaled"] = int(config["cont_dim_scaled"])

        try:
            torch.save(repaired, self.default_weight)
            LOG.info("Repaired legacy bvr_intent_new checkpoint metadata: %s", self.default_weight)
        except Exception as exc:
            LOG.warning("Failed to rewrite bvr_intent_new checkpoint %s: %s", self.default_weight, exc)

    def _load_model(self) -> bool:
        if self.model is not None:
            return True

        if not self.default_weight or not self.default_weight.exists():
            if not self._warned_missing:
                LOG.warning("bvr_intent_new weight missing: %s", self.default_weight)
                self._warned_missing = True
            return False

        try:
            ckpt = torch.load(self.default_weight, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state") or ckpt
            sidecar_meta = self._load_sidecar_meta()
            config = self._infer_checkpoint_config(ckpt, state_dict, sidecar_meta)

            self.label2id = dict(config["label2id"])
            input_dim = int(config["input_dim_num"])
            num_classes = int(config["num_classes"])
            hidden_dim = int(config["hidden_dim"])
            num_layers = int(config["num_layers"])
            self.use_status_embed = bool(config["use_status_embed"])
            self.status_vocab_size = int(config["status_vocab_size"])
            self.status_emb_dim = int(config["status_emb_dim"])
            bidirectional = bool(config["bidirectional"])
            use_self_attn = bool(config["use_self_attn"])
            attn_dim = config["attn_dim"]
            dropout = float(config["dropout"])
            use_layernorm = bool(config["use_layernorm"])
            self.window_size = int(config["window_size"])
            self.scaler_mean = config["scaler_mean"]
            self.scaler_scale = config["scaler_scale"]
            self.cont_dim_scaled = config["cont_dim_scaled"]
            self.has_wing_mask = bool(config["has_wing_mask"])
            self.scale_wing_mask = bool(config["scale_wing_mask"])

            self.model = IntentLSTM(
                input_dim_num=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                use_status_embed=self.use_status_embed,
                status_vocab_size=self.status_vocab_size,
                status_emb_dim=self.status_emb_dim,
                bidirectional=bidirectional,
                use_self_attn=use_self_attn,
                attn_dim=attn_dim,
                dropout=dropout,
                use_layernorm=use_layernorm,
            ).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            if self._needs_checkpoint_repair(ckpt):
                self._rewrite_checkpoint(ckpt, state_dict, config)
            LOG.info("Loaded bvr_intent_new checkpoint: %s", self.default_weight)
            return True
        except Exception as exc:  # pragma: no cover
            LOG.warning("Failed to load bvr_intent_new checkpoint %s: %s", self.default_weight, exc)
            return False

    def _resolve_enemy_wingman(self, enemy_aircraft: Any, env: Any) -> Optional[Any]:
        enemy_id = _resolve_agent_id(enemy_aircraft, env)
        wing_id = _pairmate_id(enemy_id)
        if not wing_id:
            return None
        wingman = (getattr(env, "agents", {}) or {}).get(wing_id)
        if wingman is None or not getattr(wingman, "is_alive", False):
            return None
        return wingman

    def _extract_features(
        self,
        enemy_aircraft: Any,
        my_aircraft: Any,
        env: Any,
    ) -> Optional[Tuple[np.ndarray, int]]:
        try:
            enemy_pos = _safe_get_pos(enemy_aircraft)
            my_pos = _safe_get_pos(my_aircraft)
            enemy_vel = _safe_get_vel(enemy_aircraft)
            my_vel = _safe_get_vel(my_aircraft)

            rel = enemy_pos - my_pos
            range_m = float(np.linalg.norm(rel) + 1e-6)
            rel_alt = float(rel[2])

            enemy_speed = float(np.linalg.norm(enemy_vel))
            own_speed = float(np.linalg.norm(my_vel))

            los = my_pos - enemy_pos
            los_norm = np.linalg.norm(los) + 1e-6
            los_unit = los / los_norm

            enemy_vel_h = enemy_vel.copy()
            enemy_vel_h[2] = 0.0
            los_h = los.copy()
            los_h[2] = 0.0
            ev_h_norm = np.linalg.norm(enemy_vel_h) + 1e-6
            los_h_norm = np.linalg.norm(los_h) + 1e-6
            cos_ang = np.clip(np.dot(enemy_vel_h, los_h) / (ev_h_norm * los_h_norm), -1.0, 1.0)
            enemy_enter_angle = float(np.degrees(np.arccos(cos_ang)))

            bearing = float(np.degrees(np.arctan2(-rel[1], -rel[0])) % 360.0)
            rel_vel = enemy_vel - my_vel
            closure = float(-np.dot(rel_vel, los_unit))

            wing_valid = 0.0
            dist_enemy_wing = 0.0
            distdiff_to_own = 0.0
            angle_enemy_wing_from_own = 0.0
            alt_diff_enemy_wing = 0.0

            wingman = self._resolve_enemy_wingman(enemy_aircraft, env)
            if wingman is not None:
                wing_pos = _safe_get_pos(wingman)
                if np.linalg.norm(wing_pos) > 0.0:
                    wing_valid = 1.0
                    enemy_to_wing = enemy_pos - wing_pos
                    dist_enemy_wing = float(np.linalg.norm(enemy_to_wing))

                    dist_enemy_to_own = float(np.linalg.norm(enemy_pos - my_pos))
                    dist_wing_to_own = float(np.linalg.norm(wing_pos - my_pos))
                    distdiff_to_own = dist_enemy_to_own - dist_wing_to_own

                    own_to_enemy = enemy_pos - my_pos
                    own_to_wing = wing_pos - my_pos
                    own_to_enemy_norm = np.linalg.norm(own_to_enemy) + 1e-6
                    own_to_wing_norm = np.linalg.norm(own_to_wing) + 1e-6
                    dot = float(np.dot(own_to_enemy, own_to_wing))
                    cos_enemy_wing = np.clip(dot / (own_to_enemy_norm * own_to_wing_norm), -1.0, 1.0)
                    angle_enemy_wing_from_own = float(np.degrees(np.arccos(cos_enemy_wing)))

                    alt_diff_enemy_wing = float(enemy_pos[2] - wing_pos[2])

            features = np.asarray(
                [
                    enemy_speed,
                    own_speed,
                    rel_alt,
                    range_m,
                    enemy_enter_angle,
                    bearing,
                    0.0,
                    closure,
                    dist_enemy_wing,
                    distdiff_to_own,
                    angle_enemy_wing_from_own,
                    alt_diff_enemy_wing,
                    wing_valid,
                ],
                dtype=np.float32,
            )
            status_code = _status_code_from_aircraft(enemy_aircraft)
            return features, status_code
        except Exception as exc:  # pragma: no cover
            LOG.debug("bvr_intent_new feature extraction failed: %s", exc)
            return None

    def _update_bearing_rate(self, key: str, features: np.ndarray, status_code: int):
        buffer = self._buffers[key]
        if buffer:
            previous_bearing = float(buffer[-1]["features"][5])
            delta = float(features[5] - previous_bearing)
            delta = (delta + 180.0) % 360.0 - 180.0
            features[6] = delta
        buffer.append({"features": features.copy(), "status_code": int(status_code)})

    def _normalise_window(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32).copy()
        cont_dim = self.cont_dim_scaled
        if cont_dim is None:
            cont_dim = arr.shape[-1]
            if self.has_wing_mask and not self.scale_wing_mask and cont_dim > 0:
                cont_dim -= 1
        cont_dim = max(0, min(int(cont_dim), arr.shape[-1]))

        if (
            cont_dim > 0
            and self.scaler_mean is not None
            and self.scaler_scale is not None
            and len(self.scaler_mean) == cont_dim
            and len(self.scaler_scale) == cont_dim
        ):
            scale = np.where(np.abs(self.scaler_scale) < 1e-6, 1.0, self.scaler_scale)
            arr[:, :cont_dim] = (arr[:, :cont_dim] - self.scaler_mean) / scale
        elif cont_dim > 0:
            mean = arr[:, :cont_dim].mean(axis=0, keepdims=True)
            std = arr[:, :cont_dim].std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            arr[:, :cont_dim] = (arr[:, :cont_dim] - mean) / std

        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _window_entries(self, key: str):
        buffer = self._buffers[key]
        if len(buffer) >= self.window_size:
            return list(buffer)[-self.window_size:]

        entries = list(buffer)
        while len(entries) < self.window_size and entries:
            tail = entries[-min(3, len(entries)) :]
            entries.extend(tail)
        return entries[: self.window_size]

    def _label_to_intent(self, label_txt: str) -> str:
        mapping = {
            "攻击": "ATTACK",
            "协同": "ATTACK",
            "探测": "NEUTRAL",
            "侦察": "NEUTRAL",
            "中立": "NEUTRAL",
            "防御": "NEUTRAL",
            "规避": "NEUTRAL",
            "其他": "NEUTRAL",
            "撤退": "RETREAT",
            "逃逸": "RETREAT",
        }
        return mapping.get(str(label_txt), "NEUTRAL")

    def _predict(self, enemy_aircraft: Any, my_aircraft: Any, env: Any) -> Optional[str]:
        key = _resolve_agent_id(enemy_aircraft, env)
        now_step = int(getattr(env, "current_step", 0) or 0)

        if not self._load_model():
            last_step = int(self._model_missing_log_step.get(key, -10**9))
            if now_step - last_step >= 300:
                LOG.warning("bvr_intent_new model not ready for %s", key)
                self._model_missing_log_step[key] = now_step
            self._last_infer_info[key] = {
                "algorithm": self.name,
                "used_model": False,
                "reason": "missing_weight",
                "window_size": int(self.window_size),
            }
            return self._last_intent.get(key, "NEUTRAL")

        extracted = self._extract_features(enemy_aircraft, my_aircraft, env)
        if extracted is None:
            last_step = int(self._feature_fail_log_step.get(key, -10**9))
            if now_step - last_step >= 300:
                LOG.warning("bvr_intent_new feature extraction failed for %s", key)
                self._feature_fail_log_step[key] = now_step
            self._last_infer_info[key] = {
                "algorithm": self.name,
                "used_model": False,
                "reason": "feature_extraction_failed",
                "buf_len": int(len(self._buffers[key])),
                "window_size": int(self.window_size),
            }
            return self._last_intent.get(key, "NEUTRAL")

        features, status_code = extracted
        self._update_bearing_rate(key, features, status_code)
        buffer = self._buffers[key]

        if len(buffer) < self.window_size:
            if key not in self._warmup_logged:
                LOG.info("bvr_intent_new warmup %s: %s/%s", key, len(buffer), self.window_size)
                self._warmup_logged.add(key)

        entries = self._window_entries(key)
        arr = np.stack([entry["features"] for entry in entries], axis=0)
        arr = self._normalise_window(arr)
        x_num = torch.tensor(arr[None, ...], dtype=torch.float32, device=self.device)
        x_status = None
        if self.use_status_embed:
            statuses = np.asarray([entry["status_code"] for entry in entries], dtype=np.int64)
            x_status = torch.tensor(statuses[None, ...], dtype=torch.long, device=self.device)

        try:
            with torch.no_grad():
                logits = self.model(x_num, x_status)
                pred_id = int(torch.argmax(logits, dim=1).item())
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().tolist()
        except Exception as exc:  # pragma: no cover
            LOG.error("bvr_intent_new inference failed for %s: %s", key, exc, exc_info=True)
            self._last_infer_info[key] = {
                "algorithm": self.name,
                "used_model": False,
                "reason": f"inference_exception: {exc}",
                "buf_len": int(len(buffer)),
                "window_size": int(self.window_size),
            }
            return self._last_intent.get(key, "NEUTRAL")

        id2label = {value: label for label, value in self.label2id.items()}
        label_txt = id2label.get(pred_id, "其他")
        intent = self._label_to_intent(label_txt)
        conf = float(max(probs)) if probs else None
        last_intent = self._last_intent.get(key)
        self._last_intent[key] = intent
        self._infer_log_count[key] += 1

        self._last_infer_info[key] = {
            "algorithm": self.name,
            "used_model": True,
            "reason": "model_infer",
            "buf_len": int(len(buffer)),
            "window_size": int(self.window_size),
            "padded_window": bool(len(buffer) < self.window_size),
            "pred_label": str(label_txt),
            "pred_id": int(pred_id),
            "confidence": conf,
            "intent": str(intent),
            "intent_last": str(last_intent) if last_intent is not None else None,
            "last_feat": {
                "enemy_speed": float(features[0]),
                "own_speed": float(features[1]),
                "rel_alt": float(features[2]),
                "range_m": float(features[3]),
                "enter_angle_deg": float(features[4]),
                "bearing_deg": float(features[5]),
                "bearing_rate_deg": float(features[6]),
                "closure_m_s": float(features[7]),
                "dist_enemy_wing_m": float(features[8]),
                "distdiff_to_own_m": float(features[9]),
                "angle_enemy_wing_from_own_deg": float(features[10]),
                "alt_diff_enemy_wing_m": float(features[11]),
                "wing_valid": float(features[12]),
                "status_code": int(status_code),
                "radar_mode": _normalise_radar_mode(getattr(enemy_aircraft, "radar_mode", None)),
            },
        }
        return intent
# Legacy import compatibility.
BVRIntentV1Adapter = BVRIntentNewAdapter
