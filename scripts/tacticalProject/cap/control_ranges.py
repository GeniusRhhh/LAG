"""
Control-range configuration for Chapter 6 CAP tactics.

This module now supports dynamic, situation-driven control-range generation.
The old proportional compression logic has been retired because it compressed
all nodes with a single ratio and could not reflect different detection,
engagement, and survivability constraints.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ControlRanges:
    """
    Tactical control ranges in km.

    Base sequence:
    NLT(180) -> MELD(150) -> MTR(120) -> LR(100) -> TR(80) -> DOR(60) -> DR(50) -> MAR(35)

    Second attack sequence:
    MTR'(45) -> LR'(42) -> TR'(38)
    """

    NLT: float = 180.0
    MELD: float = 150.0
    MTR: float = 120.0
    LR: float = 100.0
    TR: float = 80.0
    DOR: float = 60.0
    DR: float = 50.0
    MTR_PRIME: float = 45.0
    LR_PRIME: float = 42.0
    TR_PRIME: float = 38.0
    MAR: float = 35.0
    COMPRESSION_BASE: float = 180.0

    def get_all(self) -> Dict[str, float]:
        return {
            key: getattr(self, key)
            for key in (
                "NLT",
                "MELD",
                "MTR",
                "LR",
                "TR",
                "DOR",
                "DR",
                "MAR",
                "MTR_PRIME",
                "LR_PRIME",
                "TR_PRIME",
            )
        }

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _ensure_monotonic(self, ranges: Dict[str, float]) -> Dict[str, float]:
        ordered = (
            ("NLT", 12.0),
            ("MELD", 10.0),
            ("MTR", 8.0),
            ("LR", 7.0),
            ("TR", 6.0),
            ("DOR", 5.0),
            ("DR", 4.0),
            ("MAR", 0.0),
        )
        previous = float("inf")
        for index, (key, min_gap) in enumerate(ordered):
            value = float(ranges.get(key, getattr(self, key)))
            if index == 0:
                ranges[key] = self._clamp(value, self.NLT * 0.45, self.NLT * 1.10)
                previous = float(ranges[key])
                continue

            lower = self.MAR * 0.85 if key == "MAR" else self.MAR + 1.0
            upper = max(lower, previous - min_gap)
            if key == "MAR":
                ranges[key] = self._clamp(value, self.MAR * 0.85, min(self.MAR * 1.25, upper))
            else:
                ranges[key] = self._clamp(value, lower, upper)
            previous = float(ranges[key])

        ranges["MTR_PRIME"] = self._clamp(
            float(ranges.get("MTR_PRIME", self.MTR_PRIME)),
            float(ranges["MAR"]) + 6.0,
            float(ranges["DR"]) - 1.0,
        )
        ranges["LR_PRIME"] = self._clamp(
            float(ranges.get("LR_PRIME", self.LR_PRIME)),
            float(ranges["MAR"]) + 4.0,
            float(ranges["MTR_PRIME"]) - 1.0,
        )
        ranges["TR_PRIME"] = self._clamp(
            float(ranges.get("TR_PRIME", self.TR_PRIME)),
            float(ranges["MAR"]) + 2.0,
            float(ranges["LR_PRIME"]) - 1.0,
        )
        return ranges

    @staticmethod
    def _zone_priority(zone: str) -> int:
        zone_text = str(zone or "").strip().upper()
        if zone_text == "HIGH":
            return 3
        if zone_text == "MEDIUM":
            return 2
        if zone_text == "LOW":
            return 1
        return 0

    def _derive_sensor_envelope(
        self,
        *,
        target_zone: str,
        has_awacs: bool,
        radar_track_count: int,
        radar_threat: float,
    ) -> tuple[float, float]:
        zone_priority = self._zone_priority(target_zone)
        nlt_km = 170.0 if has_awacs else 154.0
        meld_km = 146.0 if has_awacs else 132.0

        if zone_priority >= 3:
            nlt_km -= 10.0
            meld_km -= 10.0
        elif zone_priority == 2:
            nlt_km -= 5.0
            meld_km -= 6.0

        nlt_km += min(max(radar_track_count, 0), 4) * 1.5
        meld_km += min(max(radar_track_count, 0), 4) * 1.2
        nlt_km -= float(radar_threat) * 6.0
        meld_km -= float(radar_threat) * 5.0

        nlt_km = self._clamp(nlt_km, 138.0, 186.0)
        meld_km = self._clamp(meld_km, 116.0, nlt_km - 12.0)
        return float(nlt_km), float(meld_km)

    def _derive_first_attack_window(
        self,
        *,
        target_zone: str,
        has_awacs: bool,
        stable_ready_count: int,
        gate_pressure: float,
        closure_rate_mps: float,
        time_to_merge_s: float,
    ) -> tuple[float, float]:
        zone = str(target_zone or "").strip().upper()
        low_km = 70.0
        high_km = 84.0
        if zone == "MEDIUM":
            low_km = 68.0
            high_km = 85.0
        elif zone == "HIGH":
            low_km = 66.0
            high_km = 86.0

        if stable_ready_count > 0:
            low_km -= 1.0
            high_km += 1.0
        if not has_awacs:
            high_km -= 2.0
        if gate_pressure >= 0.60:
            low_km += 1.0
            high_km -= 2.0

        closure_term = self._clamp(float(closure_rate_mps) / 90.0, 0.0, 4.5)
        merge_term = 0.0
        if np.isfinite(float(time_to_merge_s)):
            merge_term = self._clamp(max(0.0, 180.0 - float(time_to_merge_s)) / 45.0, 0.0, 4.5)
        pressure_shift = min(6.0, closure_term + merge_term)
        low_km += pressure_shift * 0.35
        high_km += pressure_shift * 0.45

        low_km = self._clamp(low_km, 60.0, 76.0)
        high_km = self._clamp(high_km, low_km + 10.0, 90.0)
        return float(low_km), float(high_km)

    def _derive_second_attack_window(
        self,
        *,
        target_zone: str,
        active_relay: bool,
        missile_incoming: bool,
        closure_rate_mps: float,
        time_to_merge_s: float,
    ) -> tuple[float, float]:
        zone = str(target_zone or "").strip().upper()
        low_km = 36.0
        high_km = 54.0
        if zone in {"MEDIUM", "HIGH"}:
            low_km = 35.0
            high_km = 56.0

        if active_relay:
            low_km -= 1.0
            high_km += 1.0
        if missile_incoming:
            high_km -= 2.0

        closure_term = self._clamp(float(closure_rate_mps) / 110.0, 0.0, 3.0)
        merge_term = 0.0
        if np.isfinite(float(time_to_merge_s)):
            merge_term = self._clamp(max(0.0, 150.0 - float(time_to_merge_s)) / 60.0, 0.0, 3.0)
        pressure_shift = min(4.0, closure_term + merge_term)
        low_km += pressure_shift * 0.25
        high_km += pressure_shift * 0.35

        low_km = self._clamp(low_km, 34.0, 42.0)
        high_km = self._clamp(high_km, low_km + 12.0, 58.0)
        return float(low_km), float(high_km)

    def get_dynamic_ranges(
        self,
        actual_distance: float,
        context: Optional[Mapping[str, object]] = None,
        previous_ranges: Optional[Mapping[str, float]] = None,
        env: Optional[object] = None,
    ) -> Dict[str, float]:
        """
        Generate phase nodes from the current situation instead of applying a
        single compression ratio to all nodes.

        Layering:
        - NLT/MELD: detection and perception envelope
        - MTR/LR/TR: stable tracking, release gate, relay envelope
        - DOR/DR/MAR: survivability and breakaway envelope
        """
        _ = env
        if actual_distance < self.MAR * 0.75:
            return {}

        ctx = dict(context or {})
        nearest = float(actual_distance)
        has_awacs = bool(ctx.get("has_awacs_info", False))
        has_high = bool(ctx.get("has_hostile_in_high_zone", False))
        has_medium = bool(ctx.get("has_hostile_in_medium_zone", False))
        missile_incoming = bool(ctx.get("is_missile_incoming", False))
        rwr_level = int(ctx.get("rwr_level", 0) or 0)
        ecm_active = bool(ctx.get("ecm_active", False))
        radar_threat = float(ctx.get("radar_threat_level", 0.0) or 0.0)
        stable_ready_count = int(ctx.get("stable_ready_count", 0) or 0)
        radar_track_count = int(ctx.get("radar_track_count", 0) or 0)
        active_relay = bool(ctx.get("active_relay", False))
        active_guided_peak = int(ctx.get("active_guided_missile_peak", 0) or 0)
        closure_rate_mps = float(ctx.get("closure_rate_mps", 0.0) or 0.0)
        time_to_merge_s = float(ctx.get("time_to_merge_s", float("inf")) or float("inf"))
        nearest_altitude_delta_km = abs(float(ctx.get("nearest_altitude_delta_km", 0.0) or 0.0))
        nearest_target_speed_mps = float(ctx.get("nearest_target_speed_mps", 0.0) or 0.0)
        gate_pass_count = int(ctx.get("gate_pass_count", 0) or 0)
        gate_block_count = int(ctx.get("gate_block_count", 0) or 0)
        gate_pressure = (
            float(gate_block_count) / float(gate_pass_count + gate_block_count)
            if (gate_pass_count + gate_block_count) > 0
            else 0.0
        )
        target_zone = str(ctx.get("nearest_target_zone", "") or "").strip().upper()
        if not target_zone:
            if has_high:
                target_zone = "HIGH"
            elif has_medium:
                target_zone = "MEDIUM"
            else:
                target_zone = "LOW"

        required_seconds = float(ctx.get("stable_tracking_required_s", 5.0) or 5.0)
        relay_entry_km = float(ctx.get("relay_entry_km", 0.0) or 0.0)

        sensor_nlt, sensor_meld = self._derive_sensor_envelope(
            target_zone=target_zone,
            has_awacs=has_awacs,
            radar_track_count=radar_track_count,
            radar_threat=radar_threat,
        )
        first_low, first_high = self._derive_first_attack_window(
            target_zone=target_zone,
            has_awacs=has_awacs,
            stable_ready_count=stable_ready_count,
            gate_pressure=gate_pressure,
            closure_rate_mps=closure_rate_mps,
            time_to_merge_s=time_to_merge_s,
        )
        second_low, second_high = self._derive_second_attack_window(
            target_zone=target_zone,
            active_relay=active_relay,
            missile_incoming=missile_incoming,
            closure_rate_mps=closure_rate_mps,
            time_to_merge_s=time_to_merge_s,
        )

        prelaunch_buffer = 5.5 * required_seconds
        if not has_awacs:
            prelaunch_buffer += 4.0
        if radar_track_count < 2:
            prelaunch_buffer += 4.0
        if stable_ready_count > 0:
            prelaunch_buffer -= 2.0
        if has_high:
            prelaunch_buffer -= 2.0
        if np.isfinite(time_to_merge_s):
            if time_to_merge_s <= 90.0:
                prelaunch_buffer += 4.0
            elif time_to_merge_s <= 150.0:
                prelaunch_buffer += 2.0
        prelaunch_buffer += self._clamp(closure_rate_mps / 120.0, 0.0, 4.0)
        mtr = first_high + self._clamp(prelaunch_buffer, 16.0, 40.0)

        meld = max(mtr + 20.0, sensor_meld)
        nlt = max(meld + 20.0, sensor_nlt)

        threat_push = 0.0
        if has_high:
            threat_push += 5.0
        elif has_medium:
            threat_push += 2.5
        threat_push += max(rwr_level - 2, 0) * 1.6
        threat_push += radar_threat * 5.5
        if missile_incoming:
            threat_push += 7.0
        if ecm_active:
            threat_push += 2.0
        if active_guided_peak > 0:
            threat_push += min(active_guided_peak, 4) * 0.5
        if active_relay:
            threat_push -= 1.5
        if closure_rate_mps > 0.0:
            threat_push += self._clamp(closure_rate_mps / 45.0, 0.0, 6.0)
        if np.isfinite(time_to_merge_s):
            threat_push += self._clamp(max(0.0, 180.0 - time_to_merge_s) / 30.0, 0.0, 6.0)
        threat_push += self._clamp(nearest_altitude_delta_km / 0.8, 0.0, 3.0)
        threat_push += self._clamp(nearest_target_speed_mps / 500.0, 0.0, 2.0)

        mar = self.MAR + self._clamp(threat_push * 0.45, 0.0, 10.0)
        relay_entry = max(
            relay_entry_km,
            mar + 10.0,
            45.0,
            38.0 + self._clamp(closure_rate_mps / 12.0, 0.0, 10.0),
        )
        dor = max(second_high + 4.0, relay_entry + 9.0 + max(threat_push * 0.15, 0.0))
        dr = max(second_high, relay_entry + 2.0 + max(threat_push * 0.10, 0.0), mar + 8.0)

        if active_relay:
            dor -= 2.0
            dr -= 1.5
        if missile_incoming:
            dor += 3.0
            dr += 2.0

        if nearest <= 170.0:
            nlt = min(nlt, nearest + 26.0)
        if nearest <= 150.0:
            meld = min(meld, nearest + 20.0)
        if nearest <= 118.0:
            mtr = min(mtr, nearest + 16.0)
        if nearest <= 95.0:
            first_high = min(first_high, nearest + 10.0)
        if nearest <= 78.0:
            first_low = min(first_low, nearest + 6.0)
        if nearest <= 68.0:
            dor = min(dor, nearest - 4.0 if nearest > 10.0 else dor)
        if nearest <= 58.0:
            dr = min(dr, nearest - 6.0 if nearest > 12.0 else dr)

        ranges = {
            "NLT": nlt,
            "MELD": meld,
            "MTR": mtr,
            "LR": first_high,
            "TR": first_low,
            "DOR": dor,
            "DR": dr,
            "MAR": mar,
            "MTR_PRIME": min(max(second_high + 5.0, mar + 6.0), dr - 1.0),
            "LR_PRIME": min(max(second_high, mar + 4.0), dr - 3.0),
            "TR_PRIME": min(max(second_low, mar + 2.0), second_high - 1.0),
        }
        ranges = self._ensure_monotonic(ranges)

        if previous_ranges:
            alpha = 0.38
            smoothed: Dict[str, float] = {}
            for key, value in ranges.items():
                previous = previous_ranges.get(key)
                if previous is None:
                    smoothed[key] = float(value)
                else:
                    smoothed[key] = float(previous) * (1.0 - alpha) + float(value) * alpha
            ranges = self._ensure_monotonic(smoothed)

        return ranges

    def get_compressed_ranges(self, actual_distance: float) -> Dict[str, float]:
        """
        Compatibility wrapper.

        The legacy proportional compression has been replaced by dynamic
        range generation, but many call sites still use the old method name.
        """
        if actual_distance < self.MAR:
            return {}
        return self.get_dynamic_ranges(actual_distance)

    def get_current_node(self, distance: float, ranges_dict: Dict[str, float] = None) -> str:
        def get_val(key: str) -> float:
            return ranges_dict[key] if ranges_dict and key in ranges_dict else getattr(self, key)

        if distance > get_val("NLT"):
            return "BEYOND_NLT"
        if distance > get_val("MELD"):
            return "NLT_MELD"
        if distance > get_val("MTR"):
            return "MELD_MTR"
        if distance > get_val("LR"):
            return "MTR_LR"
        if distance > get_val("TR"):
            return "LR_TR"
        if distance > get_val("DOR"):
            return "TR_DOR"
        if distance > get_val("DR"):
            return "DOR_DR"
        if distance > get_val("MAR"):
            return "DR_MAR"
        return "BELOW_MAR"

    def is_second_attack_zone(self, distance: float) -> bool:
        return self.TR_PRIME <= distance <= (self.DR + 2.0)


DEFAULT_RANGES = ControlRanges()
