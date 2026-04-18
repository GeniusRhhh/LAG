"""Extracted AWACS/picture/context helpers for CAPTask."""

import logging
import os

import numpy as np

from envs.JSBSim.core.catalog import Catalog as c

from .cap_state_machine import CAPState, StateContext
from .picture import RiskZone as PictureRiskZone, Track, TrackSource


log = logging.getLogger(__name__)
_CAP_DEBUG_PRINT = os.environ.get("CAP_DEBUG_PRINT") == "1"


def _should_emit_picture_log(self, time_attr: str, sig_attr: str, current_time: float, interval_s: float, signature):
    last_time = float(getattr(self, time_attr, -9999.0))
    last_sig = getattr(self, sig_attr, None)
    if (current_time - last_time) >= float(interval_s) or signature != last_sig:
        setattr(self, time_attr, float(current_time))
        setattr(self, sig_attr, signature)
        return True
    return False


def _update_awacs_data(self, env, current_time: float):
    """Update AWACS truth data and detect whether a new AWACS frame arrived."""
    if not hasattr(self, "awacs") or self.awacs is None:
        if self.step_count <= 5:
            log.warning("AWACS对象未初始化")
        return

    targets = []
    for aid in env.agents:
        if aid.startswith("B") and env.agents[aid].is_alive:
            try:
                ac = env.agents[aid]
                pos = self._get_battlefield_pos(env, aid)
                _, _, alt_m = ac.get_geodetic()
                alt_km = alt_m * 0.001
                heading = self._get_heading(env, aid)
                vel = ac.get_velocity()
                targets.append(
                    {
                        "id": aid,
                        "position": [pos[0], pos[1], alt_km],
                        "velocity": list(vel),
                        "heading": heading,
                    }
                )
            except Exception as exc:
                if self.step_count <= 5:
                    log.warning(f"[WARN] 获取敌机{aid}数据失败: {exc}")

    friendly_positions = []
    for aid in ["A0100", "A0200", "A0300", "A0400"]:
        if aid in env.agents and env.agents[aid].is_alive:
            try:
                pos = self._get_battlefield_pos(env, aid)
                _, _, alt_m = env.agents[aid].get_geodetic()
                friendly_positions.append(np.array([pos[0], pos[1], alt_m * 0.001]))
            except Exception:
                pass

    self._awacs_updated_this_step = False
    if not targets:
        if hasattr(self, "_clear_awacs_tracks"):
            try:
                self._clear_awacs_tracks(env, current_time=current_time)
            except Exception:
                self._awacs_updated_this_step = False
        elif hasattr(self.awacs, "_tracks"):
            try:
                self.awacs._tracks = {}
            except Exception:
                pass
    if targets:
        self.awacs.set_ground_truth(targets)
        self.awacs.update(current_time, friendly_positions=friendly_positions)

        tracks = self.awacs.get_tracks()
        stamps = [getattr(t, "timestamp", None) for t in tracks.values()]
        stamps = [s for s in stamps if s is not None]
        awacs_stamp = max(stamps) if stamps else None
        last_stamp = getattr(self, "_awacs_last_stamp", None)
        if awacs_stamp is not None and (last_stamp is None or awacs_stamp != last_stamp):
            self._awacs_updated_this_step = True
            self._awacs_last_stamp = awacs_stamp

        track_ids = tuple(sorted(tracks.keys()))
        awacs_signature = (
            len(targets),
            len(tracks),
            track_ids[:4],
            bool(self._awacs_updated_this_step),
        )
        if _should_emit_picture_log(
            self,
            "_last_awacs_data_log_time",
            "_last_awacs_data_log_signature",
            current_time,
            12.0,
            awacs_signature,
        ):
            log.info(
                "[AWACS数据] 目标数=%d | 探测航迹数=%d | 我方位置数=%d",
                len(targets),
                len(tracks),
                len(friendly_positions),
            )
            for track_id in track_ids[:2]:
                track = tracks.get(track_id)
                pos = getattr(track, "position", None)
                if pos is not None:
                    log.info("  %s: 位置=(%.1f, %.1f)km", track_id, pos[0], pos[1])
    elif _should_emit_picture_log(
        self,
        "_last_awacs_empty_log_time",
        "_last_awacs_empty_log_signature",
        current_time,
        30.0,
        ("empty",),
    ):
        log.warning("⚠️ [AWACS数据] 无敌机目标")


def _calculate_cap_context(self, env, current_time: float) -> StateContext:
    """Build the CAP state context using precise geodetic distance."""
    import pymap3d

    my_geos = []
    my_positions = []
    for aid in env.agents:
        if aid.startswith("A") and env.agents[aid].is_alive:
            my_geos.append(env.agents[aid].get_geodetic())
            my_positions.append(self._get_battlefield_pos(env, aid))

    if my_geos:
        geo_center = np.mean(my_geos, axis=0)
        formation_center_y = np.mean(my_positions, axis=0)[1]
    else:
        geo_center = np.array([120.0, 60.0, 8000.0])
        formation_center_y = 50.0

    min_distance = float("inf")
    distance_source = "none"
    has_awacs_info = False
    use_awacs_context = self.cap_state_machine.state in (CAPState.PATROL, CAPState.INTERCEPT)
    awacs_track_count = 0

    alive_enemy_ids = set()
    if hasattr(self, "_get_alive_enemy_ids"):
        try:
            alive_enemy_ids = set(self._get_alive_enemy_ids(env))
        except Exception:
            alive_enemy_ids = set()

    if use_awacs_context and hasattr(self, "_get_alive_awacs_tracks"):
        awacs_tracks = self._get_alive_awacs_tracks(env)
    elif use_awacs_context and hasattr(self, "awacs") and self.awacs is not None:
        awacs_tracks = self.awacs.get_tracks()
        if alive_enemy_ids:
            awacs_tracks = {
                str(tid): track for tid, track in awacs_tracks.items()
                if str(tid) in alive_enemy_ids
            }
        else:
            awacs_tracks = {}
    else:
        awacs_tracks = {}
    if use_awacs_context:
        awacs_track_count = len(awacs_tracks)
        if awacs_tracks:
            has_awacs_info = True
            self._last_awacs_info_time = current_time
            for _, track in awacs_tracks.items():
                if hasattr(track, "position") and track.position is not None:
                    t_lon, t_lat = self.coord_sys.battlefield_to_geodetic(track.position[0], track.position[1])
                    t_alt_m = track.position[2] * 1000.0
                    n, e, d = pymap3d.geodetic2ned(
                        t_lat,
                        t_lon,
                        t_alt_m,
                        geo_center[1],
                        geo_center[0],
                        geo_center[2],
                    )
                    dist = np.sqrt(n**2 + e**2 + d**2) / 1000.0
                    if dist < min_distance:
                        min_distance = dist
                        distance_source = "awacs"

        else:
            grace_s = float(getattr(self, "_awacs_grace_seconds", 15.0))
            last_t = getattr(self, "_last_awacs_info_time", None)
            if alive_enemy_ids and last_t is not None and (current_time - float(last_t)) <= grace_s:
                has_awacs_info = True

    picture_min_distance = float("inf")
    picture_threats = self._get_picture_threats(current_time=current_time, purpose="decision")
    search_threats = self._get_picture_threats(current_time=current_time, purpose="search")
    raw_threats = self._get_picture_threats(current_time=current_time, purpose="raw")

    if min_distance == float("inf") and picture_threats:
        for track in picture_threats:
            t_lon, t_lat = self.coord_sys.battlefield_to_geodetic(track.x, track.y)
            t_alt_m = track.altitude * 1000.0
            n, e, d = pymap3d.geodetic2ned(
                t_lat,
                t_lon,
                t_alt_m,
                geo_center[1],
                geo_center[0],
                geo_center[2],
            )
            dist = np.sqrt(n**2 + e**2 + d**2) / 1000.0
            picture_min_distance = min(picture_min_distance, dist)
            if dist < min_distance:
                min_distance = dist
                distance_source = "picture"

        if not has_awacs_info:
            grace_s = float(getattr(self, "_awacs_grace_seconds", 15.0))
            last_t = getattr(self, "_last_awacs_info_time", None)
            if alive_enemy_ids and last_t is not None and (current_time - float(last_t)) <= grace_s:
                has_awacs_info = True

    live_env_min_distance = float("inf")
    if min_distance == float("inf"):
        for aid in env.agents:
            if aid.startswith("B") and env.agents[aid].is_alive:
                e_geo = env.agents[aid].get_geodetic()
                n, e, d = pymap3d.geodetic2ned(
                    e_geo[1],
                    e_geo[0],
                    e_geo[2],
                    geo_center[1],
                    geo_center[0],
                    geo_center[2],
                )
                dist = np.sqrt(n**2 + e**2 + d**2) / 1000.0
                live_env_min_distance = min(live_env_min_distance, dist)
                if dist < min_distance:
                    min_distance = dist
                    distance_source = "live"
                if alive_enemy_ids and dist < 400:
                    has_awacs_info = True
    else:
        for aid in env.agents:
            if aid.startswith("B") and env.agents[aid].is_alive:
                e_geo = env.agents[aid].get_geodetic()
                n, e, d = pymap3d.geodetic2ned(
                    e_geo[1],
                    e_geo[0],
                    e_geo[2],
                    geo_center[1],
                    geo_center[0],
                    geo_center[2],
                )
                dist = np.sqrt(n**2 + e**2 + d**2) / 1000.0
                live_env_min_distance = min(live_env_min_distance, dist)

    if picture_min_distance == float("inf"):
        picture_min_distance = -1.0
    if live_env_min_distance == float("inf"):
        live_env_min_distance = -1.0

    recent_lost_count = sum(1 for threat in search_threats if getattr(threat, "is_lost", False))
    search_ids = {threat.track_id for threat in search_threats}
    stale_filtered_count = sum(1 for threat in raw_threats if threat.track_id not in search_ids)
    freshness_signature = (
        len(raw_threats),
        len(picture_threats),
        len(search_threats),
        recent_lost_count,
        stale_filtered_count,
        int(picture_min_distance // 5.0) if picture_min_distance >= 0.0 else -1,
    )
    if _should_emit_picture_log(
        self,
        "_last_picture_freshness_log_time",
        "_last_picture_freshness_log_signature",
        current_time,
        20.0,
        freshness_signature,
    ):
        log.info(
            "[PICTURE_FRESHNESS] raw=%d decision=%d search=%d recent_lost=%d stale_filtered=%d nearest_picture=%s",
            len(raw_threats),
            len(picture_threats),
            len(search_threats),
            recent_lost_count,
            stale_filtered_count,
            "-" if picture_min_distance < 0.0 else f"{picture_min_distance:.1f}km",
        )

    distance_delta = (
        abs(picture_min_distance - live_env_min_distance)
        if picture_min_distance >= 0.0 and live_env_min_distance >= 0.0
        else 0.0
    )
    mismatch_signature = (
        int(picture_min_distance // 5.0) if picture_min_distance >= 0.0 else -1,
        int(live_env_min_distance // 5.0) if live_env_min_distance >= 0.0 else -1,
        int(distance_delta // 5.0),
        len(picture_threats),
        recent_lost_count,
    )
    if (
        distance_delta >= float(getattr(self, "_picture_diag_distance_delta_km", 18.0))
        and _should_emit_picture_log(
            self,
            "_last_picture_diag_log_time",
            "_last_picture_diag_log_signature",
            current_time,
            18.0,
            mismatch_signature,
        )
    ):
        log.warning(
            "[PICTURE_DIAG] picture=%.1fkm live=%.1fkm delta=%.1fkm awacs=%s decision=%d search=%d recent_lost=%d stale_filtered=%d",
            picture_min_distance,
            live_env_min_distance,
            distance_delta,
            "Y" if has_awacs_info else "N",
            len(picture_threats),
            len(search_threats),
            recent_lost_count,
            stale_filtered_count,
        )

    has_high_zone = self._get_picture_zone_count(PictureRiskZone.HIGH, current_time=current_time, purpose="decision") > 0
    has_medium_zone = self._get_picture_zone_count(PictureRiskZone.MEDIUM, current_time=current_time, purpose="decision") > 0
    hostile_alive_count = sum(
        1
        for aid, aircraft in getattr(env, "agents", {}).items()
        if str(aid).startswith("B") and getattr(aircraft, "is_alive", False)
    )

    if hostile_alive_count <= 0:
        min_distance = float("inf")
        has_awacs_info = False
        has_high_zone = False
        has_medium_zone = False
        awacs_track_count = 0

    distance_bucket = -1 if min_distance == float("inf") else int(min_distance // 5.0)
    context_signature = (awacs_track_count, bool(has_awacs_info), distance_bucket, distance_source)
    if _should_emit_picture_log(
        self,
        "_last_awacs_context_log_time",
        "_last_awacs_context_log_signature",
        current_time,
        12.0,
        context_signature,
    ):
        distance_text = "inf" if min_distance == float("inf") else f"{min_distance:.1f}km"
        log.info(
            "[状态上下文] AWACS航迹数=%d | has_awacs_info=%s | 最近距离=%s",
            awacs_track_count,
            has_awacs_info,
            distance_text,
        )

    return StateContext(
        min_threat_distance=min_distance,
        formation_center_y=formation_center_y,
        has_awacs_info=has_awacs_info,
        has_hostile_in_high_zone=has_high_zone,
        has_hostile_in_medium_zone=has_medium_zone,
        is_missile_incoming=self._is_missile_incoming(env),
        fuel_critical=self._is_fuel_critical(env),
        mission_time=current_time,
        hostile_alive_count=hostile_alive_count,
    )


def _update_picture(self, env, current_time: float):
    """Fuse AWACS and radar tracks into the picture and coop-detection cache."""
    enemies = []
    for aid in env.agents:
        if aid.startswith("B") and env.agents[aid].is_alive:
            ac = env.agents[aid]
            x_km, y_km = self._get_battlefield_pos(env, aid)
            _, _, alt_m = ac.get_geodetic()
            alt_km = alt_m * 0.001
            hdg = np.degrees(ac.get_property_value(c.attitude_heading_true_rad))
            vel = ac.get_property_values([c.velocities_u_mps, c.velocities_v_mps, c.velocities_w_mps])
            enemies.append({"id": aid, "position": [x_km, y_km, alt_km], "velocity": list(vel), "heading": hdg})
    alive_enemy_ids = {enemy["id"] for enemy in enemies}
    if hasattr(self, "_clear_enemy_contact_state"):
        try:
            self._clear_enemy_contact_state(
                env,
                clear_picture=not alive_enemy_ids,
                reason="picture_sync",
            )
        except Exception:
            pass

    if hasattr(self, "_get_alive_awacs_tracks"):
        awacs_raw_tracks = self._get_alive_awacs_tracks(env)
    else:
        awacs_raw_tracks = self.awacs.get_tracks() if hasattr(self, "awacs") and self.awacs is not None else {}
        if alive_enemy_ids:
            awacs_raw_tracks = {tid: track for tid, track in awacs_raw_tracks.items() if tid in alive_enemy_ids}
        else:
            awacs_raw_tracks = {}
    awacs_only_max_age_s = float(getattr(self, "_picture_awacs_only_max_age_s", 20.0))
    awacs_partner_max_age_s = float(getattr(self, "_picture_awacs_partner_max_age_s", 5.0))
    awacs_tracks = {}
    awacs_age_samples = []
    awacs_partner_fresh = 0
    awacs_stale_dropped = 0
    for tid, track in awacs_raw_tracks.items():
        timestamp = getattr(track, "timestamp", None)
        if timestamp is None:
            awacs_tracks[tid] = track
            awacs_partner_fresh += 1
            continue
        try:
            age_s = max(0.0, current_time - float(timestamp))
        except Exception:
            age_s = None
        if age_s is None:
            awacs_tracks[tid] = track
            awacs_partner_fresh += 1
            continue
        awacs_age_samples.append(age_s)
        if age_s <= awacs_only_max_age_s:
            awacs_tracks[tid] = track
        else:
            awacs_stale_dropped += 1
        if age_s <= awacs_partner_max_age_s:
            awacs_partner_fresh += 1

    radar_tracks = {}
    radar_track_max_age_s = float(getattr(self, "_picture_radar_track_max_age_s", 1.0))
    enemy_truth_by_id = {enemy["id"]: enemy for enemy in enemies}
    for aid, tracks in self.cap_radar._radar_tracks.items():
        for tid, track in tracks.items():
            if tid not in alive_enemy_ids:
                continue
            last_upd = float(getattr(track, "last_update", 0.0))
            if last_upd <= 0.0 or (current_time - last_upd) > radar_track_max_age_s:
                continue
            if tid in enemy_truth_by_id:
                gt = enemy_truth_by_id[tid]
                pos = np.array(gt["position"])
                vel = np.array(gt["velocity"])
                pos += np.random.normal(0, 0.1, 3)
                radar_tracks[tid] = Track(
                    track_id=tid,
                    source=TrackSource.RADAR,
                    position=pos,
                    velocity=vel,
                    heading=gt["heading"],
                    timestamp=last_upd,
                    confidence=track.detection_prob,
                    is_hostile=True,
                )

    awacs_fusion_tracks = awacs_tracks if bool(getattr(self, "_awacs_updated_this_step", False)) else {}
    fused = self.track_fusion.fuse(awacs_fusion_tracks, radar_tracks, current_time)
    for tid, ft in fused.items():
        self.picture.update_track(ft)
        if hasattr(self, "coop_detection") and self.coop_detection is not None:
            meas_time = None
            try:
                aw = getattr(ft, "awacs_track", None)
                if aw is not None:
                    ts = getattr(aw, "timestamp", None)
                    if ts is not None:
                        meas_time = float(ts)
                rd = getattr(ft, "radar_track", None)
                if rd is not None:
                    ts = getattr(rd, "timestamp", None)
                    if ts is not None:
                        meas_time = float(ts) if meas_time is None else max(meas_time, float(ts))
            except Exception:
                meas_time = None
            if meas_time is None:
                meas_time = current_time

            self.coop_detection.update_target_info(
                tid,
                (ft.position[0], ft.position[1]),
                (ft.velocity[0] / 1000.0, ft.velocity[1] / 1000.0),
                ft.heading,
                meas_time,
            )

    picture_raw = self._get_picture_threats(current_time=current_time, purpose="raw")
    picture_decision = self._get_picture_threats(current_time=current_time, purpose="decision")
    picture_search = self._get_picture_threats(current_time=current_time, purpose="search")
    dual_source_count = sum(1 for track in fused.values() if getattr(track, "awacs_track", None) is not None and getattr(track, "radar_track", None) is not None)
    radar_only_count = sum(1 for track in fused.values() if getattr(track, "radar_track", None) is not None and getattr(track, "awacs_track", None) is None)
    awacs_only_count = sum(1 for track in fused.values() if getattr(track, "awacs_track", None) is not None and getattr(track, "radar_track", None) is None)
    lost_count = sum(1 for track in fused.values() if getattr(track, "is_lost", False))
    awacs_newest_age = min(awacs_age_samples) if awacs_age_samples else -1.0
    awacs_oldest_age = max(awacs_age_samples) if awacs_age_samples else -1.0
    picture_signature = (
        len(awacs_raw_tracks),
        len(awacs_tracks),
        len(awacs_fusion_tracks),
        awacs_partner_fresh,
        awacs_stale_dropped,
        len(radar_tracks),
        dual_source_count,
        radar_only_count,
        awacs_only_count,
        len(fused),
        len(picture_decision),
        len(picture_search),
        bool(getattr(self, "_awacs_updated_this_step", False)),
    )
    if _should_emit_picture_log(
        self,
        "_last_picture_update_log_time",
        "_last_picture_update_log_signature",
        current_time,
        20.0,
        picture_signature,
    ):
        log.info(
            "[PICTURE_UPDATE] awacs_raw=%d usable=%d in_fusion=%d partner=%d stale=%d radar=%d fused=%d src=%d/%d/%d lost=%d raw=%d decision=%d search=%d awacs_updated=%s awacs_age=%.1f/%.1fs",
            len(awacs_raw_tracks),
            len(awacs_tracks),
            len(awacs_fusion_tracks),
            awacs_partner_fresh,
            awacs_stale_dropped,
            len(radar_tracks),
            len(fused),
            dual_source_count,
            radar_only_count,
            awacs_only_count,
            lost_count,
            len(picture_raw),
            len(picture_decision),
            len(picture_search),
            "Y" if getattr(self, "_awacs_updated_this_step", False) else "N",
            awacs_newest_age,
            awacs_oldest_age,
        )


def _build_state_context(self, env, current_time: float) -> StateContext:
    """Build a simple formation-center context from the current fused picture."""
    my_positions = []
    for aid in env.agents:
        if aid.startswith("A") and env.agents[aid].is_alive:
            my_positions.append(self._get_battlefield_pos(env, aid))

    if my_positions:
        center_x = np.mean([pos[0] for pos in my_positions])
        center_y = np.mean([pos[1] for pos in my_positions])
    else:
        center_x, center_y = 100.0, 50.0

    min_dist = float("inf")
    center_pos = np.array([center_x, center_y, 8.0])
    nearest = self._get_picture_nearest_threat(center_pos, current_time=current_time, purpose="decision")
    if nearest:
        min_dist = np.linalg.norm(np.array([center_x, center_y]) - np.array([nearest.x, nearest.y]))
    hostile_alive_count = sum(
        1
        for aid, aircraft in getattr(env, "agents", {}).items()
        if str(aid).startswith("B") and getattr(aircraft, "is_alive", False)
    )

    return StateContext(
        min_threat_distance=min_dist,
        formation_center_y=center_y,
        has_hostile_in_high_zone=self._get_picture_zone_count(PictureRiskZone.HIGH, current_time=current_time, purpose="decision") > 0,
        has_hostile_in_medium_zone=self._get_picture_zone_count(PictureRiskZone.MEDIUM, current_time=current_time, purpose="decision") > 0,
        is_missile_incoming=self._is_missile_incoming(env),
        fuel_critical=self._is_fuel_critical(env),
        mission_time=current_time - self.mission_start_time,
        hostile_alive_count=hostile_alive_count,
    )
