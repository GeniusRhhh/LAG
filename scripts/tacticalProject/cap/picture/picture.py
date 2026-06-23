"""
感知层数据结构 - Picture、Track、FusedTrack
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class TrackSource(Enum):
    """航迹来源"""

    AWACS = "awacs"
    RADAR = "radar"
    FUSED = "fused"


@dataclass
class Track:
    """单条航迹数据"""

    track_id: str
    source: TrackSource
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    timestamp: float
    confidence: float = 1.0
    is_hostile: bool = True

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    @property
    def altitude(self) -> float:
        return float(self.position[2])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def distance_to(self, other: "Track") -> float:
        """计算与另一条航迹的距离(km)"""
        return float(np.linalg.norm(self.position - other.position))


@dataclass
class FusedTrack(Track):
    """融合航迹"""

    awacs_track: Optional[Track] = None
    radar_track: Optional[Track] = None
    lost_since: Optional[float] = None

    @property
    def is_lost(self) -> bool:
        return self.lost_since is not None

    def time_since_lost(self, current_time: float) -> float:
        return current_time - self.lost_since if self.lost_since is not None else 0.0


class RiskZone(Enum):
    """风险区"""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    OUTSIDE = "OUTSIDE"


@dataclass
class Picture:
    """态势图 - 战场全局感知数据结构"""

    tracks: Dict[str, FusedTrack] = field(default_factory=dict)
    faor_length: float = 300.0

    @property
    def _zone_bounds(self) -> Tuple[float, float]:
        return self.faor_length / 3.0, self.faor_length * 2.0 / 3.0

    def update_track(self, track: FusedTrack):
        self.tracks[track.track_id] = track

    def remove_track(self, track_id: str):
        self.tracks.pop(track_id, None)

    def get_risk_zone(self, y: float) -> RiskZone:
        high_max, medium_max = self._zone_bounds
        if y < 0 or y > self.faor_length:
            return RiskZone.OUTSIDE
        if y < high_max:
            return RiskZone.HIGH
        if y < medium_max:
            return RiskZone.MEDIUM
        return RiskZone.LOW

    def _is_track_visible(
        self,
        track: FusedTrack,
        current_time: Optional[float] = None,
        max_track_age: Optional[float] = None,
        include_recent_lost: bool = True,
        max_lost_time: Optional[float] = None,
    ) -> bool:
        if not getattr(track, "is_hostile", False):
            return False
        if current_time is None:
            return True

        timestamp = getattr(track, "timestamp", None)
        if max_track_age is not None and timestamp is not None:
            if (float(current_time) - float(timestamp)) > float(max_track_age):
                return False

        if not track.is_lost:
            return True
        if not include_recent_lost:
            return False
        if max_lost_time is None:
            return True
        return track.time_since_lost(float(current_time)) <= float(max_lost_time)

    def get_threats_in_zone(
        self,
        zone: RiskZone,
        current_time: Optional[float] = None,
        max_track_age: Optional[float] = None,
        include_recent_lost: bool = True,
        max_lost_time: Optional[float] = None,
    ) -> List[FusedTrack]:
        return [
            track
            for track in self.tracks.values()
            if self._is_track_visible(
                track,
                current_time=current_time,
                max_track_age=max_track_age,
                include_recent_lost=include_recent_lost,
                max_lost_time=max_lost_time,
            )
            and self.get_risk_zone(track.y) == zone
        ]

    def get_all_threats(
        self,
        current_time: Optional[float] = None,
        max_track_age: Optional[float] = None,
        include_recent_lost: bool = True,
        max_lost_time: Optional[float] = None,
    ) -> List[FusedTrack]:
        return [
            track
            for track in self.tracks.values()
            if self._is_track_visible(
                track,
                current_time=current_time,
                max_track_age=max_track_age,
                include_recent_lost=include_recent_lost,
                max_lost_time=max_lost_time,
            )
        ]

    def get_nearest_threat(
        self,
        ref_pos: np.ndarray,
        current_time: Optional[float] = None,
        max_track_age: Optional[float] = None,
        include_recent_lost: bool = True,
        max_lost_time: Optional[float] = None,
    ) -> Optional[FusedTrack]:
        threats = self.get_all_threats(
            current_time=current_time,
            max_track_age=max_track_age,
            include_recent_lost=include_recent_lost,
            max_lost_time=max_lost_time,
        )
        if not threats:
            return None
        ref_track = Track("ref", TrackSource.RADAR, ref_pos, np.zeros(3), 0.0, 0.0)
        return min(threats, key=lambda threat: threat.distance_to(ref_track))

    def is_target_lost(self, track_id: str, current_time: float, max_lost_time: float = 20.0) -> bool:
        track = self.tracks.get(track_id)
        if not track:
            return True
        if not track.is_lost:
            return False
        return track.time_since_lost(current_time) > max_lost_time

    def get_high_zone_threat_count(
        self,
        current_time: Optional[float] = None,
        max_track_age: Optional[float] = None,
        include_recent_lost: bool = True,
        max_lost_time: Optional[float] = None,
    ) -> int:
        return len(
            self.get_threats_in_zone(
                RiskZone.HIGH,
                current_time=current_time,
                max_track_age=max_track_age,
                include_recent_lost=include_recent_lost,
                max_lost_time=max_lost_time,
            )
        )

    def get_medium_zone_threat_count(
        self,
        current_time: Optional[float] = None,
        max_track_age: Optional[float] = None,
        include_recent_lost: bool = True,
        max_lost_time: Optional[float] = None,
    ) -> int:
        return len(
            self.get_threats_in_zone(
                RiskZone.MEDIUM,
                current_time=current_time,
                max_track_age=max_track_age,
                include_recent_lost=include_recent_lost,
                max_lost_time=max_lost_time,
            )
        )


from .awacs_source import MockAwacsDataSource  # noqa: E402
from .track_fusion import TrackFusion  # noqa: E402
