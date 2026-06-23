"""Shared radar enums and data models."""

from dataclasses import dataclass
from enum import Enum


class RadarStatus(Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    LOCK = "LOCK"
    STANDBY = "STANDBY"
    JAMMING = "JAMMING"
    MAINTENANCE = "MAINTENANCE"


class ECMType(Enum):
    NOISE_JAMMING = "NOISE_JAMMING"
    DECEPTION_JAMMING = "DECEPTION_JAMMING"
    CHAFF = "CHAFF"
    FLARE = "FLARE"
    FREQUENCY_AGILITY = "FREQUENCY_AGILITY"
    SIDELOBE_BLANKING = "SIDELOBE_BLANKING"


@dataclass
class RadarTarget:
    target_id: str
    distance: float
    bearing: float
    elevation: float
    velocity: float
    rcs: float = 5.0
    detection_probability: float = 0.0
    track_quality: float = 0.0
    lock_time: float = 0.0
    last_update: float = 0.0
    doppler_shift: float = 0.0
    snr: float = 0.0
    multipath_factor: float = 1.0
    atmospheric_loss: float = 0.0
    aspect_angle: float = 0.0
    radial_velocity: float = 0.0
    in_notch: bool = False
    clutter_factor: float = 1.0
    rwr_threat_level: int = 0


@dataclass
class APG68RadarModel:
    max_detection_range: float = 200000
    max_detection_range_large: float = 240000
    max_track_range: float = 180000
    max_lock_range: float = 130000
    max_simultaneous_tracks: int = 10
    max_simultaneous_engagement: int = 2
    search_beam_width: float = 120.0
    search_elevation_coverage: float = 60.0
    track_beam_width: float = 2.5
    lock_beam_width: float = 0.8
    scan_period: float = 2.0
    lock_update_rate: float = 0.04
    track_update_rate: float = 0.3
    detection_probability_base: float = 0.90
    track_loss_probability: float = 0.015
    lock_loss_probability: float = 0.008
    jamming_resistance: float = 0.75
    eccm_capability: float = 0.82
    frequency_agility: bool = True
    sidelobe_suppression: float = 0.90
    lpi_capability: float = 0.65
    has_look_down_shoot_down: bool = True
    has_synthetic_aperture: bool = True
    has_ground_moving_target: bool = True
    has_sea_surface_search: bool = True
    weather_degradation: float = 0.08
    terrain_masking_threshold: float = 300.0
    atmospheric_absorption: float = 0.0008
    range_resolution: float = 30.0
    angular_resolution: float = 1.5
    velocity_resolution: float = 5.0


@dataclass
class N001VERadarModel:
    max_detection_range: float = 200000
    max_track_range: float = 170000
    max_lock_range: float = 120000
    max_simultaneous_tracks: int = 10
    max_simultaneous_engagement: int = 2
    search_beam_width: float = 70.0
    track_beam_width: float = 3.0
    lock_beam_width: float = 1.0
    scan_period: float = 3.5
    lock_update_rate: float = 0.05
    track_update_rate: float = 0.5
    detection_probability_base: float = 0.85
    track_loss_probability: float = 0.02
    lock_loss_probability: float = 0.01
    jamming_resistance: float = 0.6
    eccm_capability: float = 0.7
    frequency_agility: bool = True
    sidelobe_suppression: float = 0.85
    weather_degradation: float = 0.1
    terrain_masking_threshold: float = 500.0
    atmospheric_absorption: float = 0.001
    has_look_down_shoot_down: bool = False
