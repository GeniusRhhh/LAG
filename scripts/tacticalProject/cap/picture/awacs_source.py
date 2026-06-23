"""AWACS data source used by CAP picture fusion.

This module keeps the public interface stable while allowing dedicated
cooperative-detection verification runs to override internal scan/loss
behavior for fair proposed-vs-baseline comparison.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .picture import Track, TrackSource

log = logging.getLogger(__name__)


@dataclass
class AwacsConfig:
    detection_range: float = 400.0
    position_error: float = 2.5
    altitude_error: float = 0.5
    heading_error: float = 10.0
    max_lost_duration: float = 20.0
    lost_probability: float = 0.02


class MockAwacsDataSource:
    """Mock AWACS source with optional deterministic behavior."""

    def __init__(self, config: Optional[AwacsConfig] = None, seed: Optional[int] = None):
        self.config = config or AwacsConfig()
        self._tracks: Dict[str, Track] = {}
        self._ground_truth: Dict[str, Dict] = {}
        self._lost_targets: Dict[str, float] = {}

        if seed is None:
            raw = os.environ.get("CAP_AWACS_SEED", "").strip()
            if raw:
                try:
                    seed = int(raw)
                except Exception:
                    seed = None
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        det = os.environ.get("CAP_AWACS_DETERMINISTIC", "").strip().lower()
        self._deterministic = det in ("1", "true", "yes", "y", "on")

        self._fixed_center = None
        raw_center = os.environ.get("CAP_AWACS_FIXED_CENTER_KM", "").strip()
        if raw_center:
            try:
                parts = [float(x) for x in raw_center.replace(";", ",").split(",") if x.strip()]
                if len(parts) >= 2:
                    x, y = parts[0], parts[1]
                    z = parts[2] if len(parts) >= 3 else 9.0
                    self._fixed_center = np.array([x, y, z], dtype=float)
            except Exception:
                self._fixed_center = None

        # Dedicated cooperative-detection validation can override these two
        # behaviors to guarantee fair, externally scripted AWACS availability.
        self._disable_internal_scan_cycle = False
        self._disable_random_loss = False

    def _u01(self, key: str) -> float:
        seed = 0 if self._seed is None else int(self._seed)
        data = f"{seed}|{key}".encode("utf-8", errors="ignore")
        digest = hashlib.blake2b(data, digest_size=8).digest()
        val = int.from_bytes(digest, "little", signed=False)
        return (val & ((1 << 64) - 1)) / float(1 << 64)

    def set_ground_truth(self, targets: List[Dict]):
        self._ground_truth = {t["id"]: t for t in targets}

    def update(self, current_time: float, friendly_positions: List[np.ndarray] = None) -> Dict[str, Track]:
        cfg = self.config
        tracks: Dict[str, Track] = {}

        if self._fixed_center is not None:
            friendly_center = self._fixed_center
        elif friendly_positions and len(friendly_positions) > 0:
            friendly_center = np.mean(friendly_positions, axis=0)
        else:
            friendly_center = np.array([100.0, 0.0, 9.0], dtype=float)

        for tid, gt in getattr(self, "_ground_truth", {}).items():
            pos = np.array(gt["position"], dtype=float)
            dist = float(np.linalg.norm(pos - friendly_center))
            if dist > cfg.detection_range:
                continue

            if self._disable_internal_scan_cycle:
                scan_slot = int(current_time * 5.0)
            else:
                last_scan_key = f"_last_scan_{tid}"
                last_scan_time = float(getattr(self, last_scan_key, -100.0))
                if current_time - last_scan_time < 10.0:
                    if tid in self._tracks:
                        tracks[tid] = self._tracks[tid]
                    continue
                setattr(self, last_scan_key, float(current_time))
                scan_slot = int(current_time // 10.0)

            target_alt_km = float(pos[2]) if len(pos) > 2 else 8.0
            if (not self._disable_random_loss) and self._handle_lost_with_altitude(
                tid, current_time, target_alt_km, scan_slot=scan_slot
            ):
                continue

            if self._deterministic:
                u1 = self._u01(f"posdist:{tid}:{scan_slot}")
                u2 = self._u01(f"postheta:{tid}:{scan_slot}")
                u3 = self._u01(f"alt:{tid}:{scan_slot}")
                u4 = self._u01(f"hdg:{tid}:{scan_slot}")
                offset_dist = 2.0 + u1 * 1.0
                theta = u2 * 2.0 * np.pi
                alt_noise = (u3 * 2.0 - 1.0) * cfg.altitude_error
                hdg_noise = (u4 * 2.0 - 1.0) * cfg.heading_error
            else:
                offset_dist = 2.0 + float(self._rng.random()) * 1.0
                theta = float(self._rng.random()) * 2.0 * np.pi
                alt_noise = float(self._rng.uniform(-cfg.altitude_error, cfg.altitude_error))
                hdg_noise = float(self._rng.uniform(-cfg.heading_error, cfg.heading_error))

            noisy_pos = pos.copy()
            noisy_pos[0] += offset_dist * np.cos(theta)
            noisy_pos[1] += offset_dist * np.sin(theta)
            noisy_pos[2] = max(0.0, float(pos[2]) + alt_noise)
            noisy_heading = (float(gt["heading"]) + hdg_noise) % 360.0

            tracks[tid] = Track(
                track_id=tid,
                source=TrackSource.AWACS,
                position=noisy_pos,
                velocity=np.array(gt["velocity"]),
                heading=noisy_heading,
                timestamp=float(current_time),
                confidence=0.7,
                is_hostile=True,
            )

            if not hasattr(self, "_ever_detected"):
                self._ever_detected = set()
            if tid not in self._ever_detected:
                self._ever_detected.add(tid)
                log.info("[AWACS] First detect %s (dist=%.1fkm)", tid, dist)

        self._tracks = tracks
        return tracks

    def _handle_lost(self, track_id: str, current_time: float, scan_slot: Optional[int] = None) -> bool:
        cfg = self.config
        if track_id in self._lost_targets:
            data = self._lost_targets[track_id]
            if isinstance(data, tuple):
                lost_time, duration = data
            else:
                lost_time, duration = data, cfg.max_lost_duration
            if current_time - float(lost_time) < float(duration):
                return True
            del self._lost_targets[track_id]
            return False

        if self._deterministic:
            slot = 0 if scan_slot is None else int(scan_slot)
            u_loss = self._u01(f"lost:{track_id}:{slot}")
            u_dur = self._u01(f"dur:{track_id}:{slot}")
            trigger = u_loss < cfg.lost_probability
            duration = 5.0 + u_dur * max(0.0, cfg.max_lost_duration - 5.0)
        else:
            trigger = float(self._rng.random()) < cfg.lost_probability
            duration = float(self._rng.uniform(5.0, cfg.max_lost_duration))

        if trigger:
            self._lost_targets[track_id] = (float(current_time), float(duration))
            log.info("[AWACS] Random loss %s (%.1fs)", track_id, duration)
            return True
        return False

    def _handle_lost_with_altitude(
        self,
        track_id: str,
        current_time: float,
        target_alt_km: float,
        scan_slot: Optional[int] = None,
    ) -> bool:
        cfg = self.config
        if track_id in self._lost_targets:
            data = self._lost_targets[track_id]
            if isinstance(data, tuple):
                lost_time, duration = data
            else:
                lost_time, duration = data, cfg.max_lost_duration
            if current_time - float(lost_time) < float(duration):
                return True
            del self._lost_targets[track_id]
            return False

        if target_alt_km > 7.0:
            loss_prob = cfg.lost_probability
        elif target_alt_km > 5.0:
            loss_prob = 0.05
        elif target_alt_km > 3.5:
            loss_prob = 0.30
        elif target_alt_km > 1.5:
            loss_prob = 0.70
        else:
            loss_prob = 0.95

        if self._deterministic:
            slot = 0 if scan_slot is None else int(scan_slot)
            u_loss = self._u01(f"lost_alt:{track_id}:{slot}")
            u_dur = self._u01(f"dur_alt:{track_id}:{slot}")
            trigger = u_loss < loss_prob
            duration = 5.0 + u_dur * 10.0
        else:
            trigger = float(self._rng.random()) < loss_prob
            duration = float(self._rng.uniform(5.0, 15.0))

        if trigger:
            if target_alt_km < 3.5:
                duration = min(duration * 1.8, 35.0)
            self._lost_targets[track_id] = (float(current_time), float(duration))
            return True
        return False

    def get_tracks(self) -> Dict[str, Track]:
        return self._tracks.copy()

    def is_target_lost(self, track_id: str) -> bool:
        return track_id in self._lost_targets
