"""Track fusion for AWACS and radar measurements."""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from .picture import FusedTrack, Track, TrackSource
from ..tactics.state_estimator import create_state_estimator

try:
    from ..config.algorithm_config import ALGO_CONFIG
except ImportError:
    class _FallbackConfig:
        use_federated_filter = False
        use_kalman_filter = False
        fusion_threshold_km = 30.0

    ALGO_CONFIG = _FallbackConfig()


class TrackFusion:
    """Fuse AWACS and radar tracks into a single picture."""

    def __init__(
        self,
        awacs_weight: float = 0.3,
        radar_weight: float = 0.7,
        awacs_only_max_age_s: float = 20.0,
        awacs_partner_max_age_s: float = 5.0,
        fresh_track_window_s: float = 5.0,
    ):
        self.awacs_weight = awacs_weight
        self.radar_weight = radar_weight
        self.awacs_only_max_age_s = float(awacs_only_max_age_s)
        self.awacs_partner_max_age_s = float(awacs_partner_max_age_s)
        self.fresh_track_window_s = float(fresh_track_window_s)
        self._fused_tracks: Dict[str, FusedTrack] = {}
        self._debug_print = os.environ.get("CAP_DEBUG_PRINT", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        use_kalman = getattr(ALGO_CONFIG, "use_kalman_filter", False)
        self.state_estimator = create_state_estimator(use_kalman=use_kalman)

    def fuse(
        self,
        awacs_tracks: Dict[str, Track],
        radar_tracks: Dict[str, Track],
        current_time: float,
    ) -> Dict[str, FusedTrack]:
        """Fuse current-source tracks and retain only recent fused state."""
        threshold_km = float(getattr(ALGO_CONFIG, "fusion_threshold_km", 30.0))
        associations = self._associate_tracks(
            awacs_tracks,
            radar_tracks,
            threshold_km=threshold_km,
            current_time=current_time,
        )
        used_awacs_ids = set()

        for rid, radar in radar_tracks.items():
            aid = associations.get(rid)
            if aid:
                awacs = awacs_tracks[aid]
                fused = self._fuse_tracks(awacs, radar, current_time)
                used_awacs_ids.add(aid)
                if self._debug_print and current_time % 60 < 0.2:
                    dist = awacs.distance_to(radar)
                    print(f"[FUSION] radar={rid} awacs={aid} dist={dist:.1f}km")
            else:
                fused = self._create_from_single(radar, current_time, radar_source=True)
            self._update_fused_track(fused, current_time)

        for aid, awacs in awacs_tracks.items():
            if aid in used_awacs_ids:
                continue
            awacs_age = self._track_age_seconds(awacs, current_time)
            if awacs_age is not None and awacs_age > self.awacs_only_max_age_s:
                continue
            fused = self._create_from_single(awacs, current_time, radar_source=False)
            self._update_fused_track(fused, current_time)
            if self._debug_print and current_time % 60 < 0.2:
                print(f"[FUSION] awacs_only={aid}")

        current_ids = {
            track.track_id
            for track in self._fused_tracks.values()
            if float(getattr(track, "timestamp", -1.0)) >= current_time - self.fresh_track_window_s
        }

        active_tracks: Dict[str, FusedTrack] = {}
        for tid, track in self._fused_tracks.items():
            if tid in current_ids:
                active_tracks[tid] = track
            elif not track.is_lost:
                track.lost_since = current_time
                active_tracks[tid] = track

        self._fused_tracks = active_tracks
        return self._fused_tracks.copy()

    def _associate_tracks(
        self,
        awacs_tracks: Dict[str, Track],
        radar_tracks: Dict[str, Track],
        threshold_km: float,
        current_time: Optional[float] = None,
    ) -> Dict[str, str]:
        """Associate radar tracks to sufficiently fresh AWACS tracks."""
        matches: Dict[str, str] = {}
        if not awacs_tracks or not radar_tracks:
            return matches

        available_awacs = {
            aid
            for aid, awacs in awacs_tracks.items()
            if self._track_age_seconds(awacs, current_time) is None
            or self._track_age_seconds(awacs, current_time) <= self.awacs_partner_max_age_s
        }
        remaining_radar = set(radar_tracks.keys())

        for rid in list(remaining_radar):
            if rid in available_awacs:
                matches[rid] = rid
                available_awacs.remove(rid)
                remaining_radar.remove(rid)

        candidates = []
        for rid in remaining_radar:
            r_track = radar_tracks[rid]
            for aid in available_awacs:
                dist = r_track.distance_to(awacs_tracks[aid])
                if dist < threshold_km:
                    candidates.append((dist, rid, aid))

        candidates.sort(key=lambda item: item[0])
        for _, rid, aid in candidates:
            if rid not in matches and aid in available_awacs:
                matches[rid] = aid
                available_awacs.remove(aid)

        return matches

    def _update_fused_track(self, new_track: FusedTrack, current_time: float):
        """Update smoothed fused state with measurement timestamp preserved."""
        tid = new_track.track_id
        state_timestamp = float(getattr(new_track, "timestamp", current_time))
        smoothed_state = self.state_estimator.update(
            target_id=tid,
            position=new_track.position,
            velocity=new_track.velocity,
            heading=new_track.heading,
            timestamp=state_timestamp,
        )

        new_track.position = smoothed_state.position
        new_track.velocity = smoothed_state.velocity

        if tid in self._fused_tracks:
            old = self._fused_tracks[tid]
            if old.is_lost:
                if state_timestamp >= current_time - self.fresh_track_window_s:
                    new_track.lost_since = None
                else:
                    new_track.lost_since = old.lost_since

        self._fused_tracks[tid] = new_track

    def clear(self):
        self._fused_tracks.clear()
        self.state_estimator.clear()

    def _fuse_tracks(self, awacs: Track, radar: Track, current_time: float) -> FusedTrack:
        """Fuse two matched tracks."""
        if getattr(ALGO_CONFIG, "use_federated_filter", False):
            awacs_cov = 2.5 ** 2
            radar_cov = 0.1 ** 2
            total_info = 1.0 / awacs_cov + 1.0 / radar_cov
            w_r = (1.0 / radar_cov) / total_info
            w_a = (1.0 / awacs_cov) / total_info
            fused_pos = w_a * awacs.position + w_r * radar.position
            fused_vel = w_a * awacs.velocity + w_r * radar.velocity
            fused_hdg = self._fuse_heading(awacs.heading, radar.heading, w_a, w_r)
            fused_conf = max(awacs.confidence, radar.confidence)
        else:
            w_a, w_r = self.awacs_weight, self.radar_weight
            fused_pos = w_a * awacs.position + w_r * radar.position
            fused_vel = w_a * awacs.velocity + w_r * radar.velocity
            fused_hdg = self._fuse_heading(awacs.heading, radar.heading, w_a, w_r)
            fused_conf = w_a * awacs.confidence + w_r * radar.confidence

        return FusedTrack(
            track_id=radar.track_id,
            source=TrackSource.FUSED,
            position=fused_pos,
            velocity=fused_vel,
            heading=fused_hdg,
            timestamp=max(
                float(getattr(awacs, "timestamp", current_time)),
                float(getattr(radar, "timestamp", current_time)),
            ),
            confidence=min(1.0, fused_conf),
            is_hostile=radar.is_hostile,
            awacs_track=awacs,
            radar_track=radar,
            lost_since=None,
        )

    def _create_from_single(self, track: Track, current_time: float, radar_source: bool) -> FusedTrack:
        return FusedTrack(
            track_id=track.track_id,
            source=TrackSource.FUSED,
            position=track.position.copy(),
            velocity=track.velocity.copy(),
            heading=track.heading,
            timestamp=float(getattr(track, "timestamp", current_time)),
            confidence=track.confidence,
            is_hostile=track.is_hostile,
            awacs_track=None if radar_source else track,
            radar_track=track if radar_source else None,
            lost_since=None,
        )

    def _fuse_heading(self, h1: float, h2: float, w1: float, w2: float) -> float:
        r1, r2 = np.radians(h1), np.radians(h2)
        x = w1 * np.cos(r1) + w2 * np.cos(r2)
        y = w1 * np.sin(r1) + w2 * np.sin(r2)
        return float(np.degrees(np.arctan2(y, x)) % 360)

    def get_fused_tracks(self) -> Dict[str, FusedTrack]:
        return self._fused_tracks.copy()

    def _track_age_seconds(self, track: Optional[Track], current_time: Optional[float]) -> Optional[float]:
        if track is None or current_time is None:
            return None
        timestamp = getattr(track, "timestamp", None)
        if timestamp is None:
            return None
        try:
            return max(0.0, float(current_time) - float(timestamp))
        except Exception:
            return None
