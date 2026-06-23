"""CAP root-cause and safety-state helpers."""

from envs.JSBSim.core.catalog import Catalog as c


def _is_missile_incoming(self, env) -> bool:
    """Detect whether an incoming missile is targeting our side."""
    missiles = getattr(env, 'missiles', {}) or {}
    for _, missile in missiles.items():
        try:
            if hasattr(missile, 'is_done') and bool(getattr(missile, 'is_done')):
                continue
            if hasattr(missile, 'is_alive') and not bool(getattr(missile, 'is_alive')):
                continue
            target = getattr(missile, 'target_aircraft', None)
            if target is None or not getattr(target, 'is_alive', False):
                continue
            target_id = getattr(target, 'uid', '')
            if not str(target_id).startswith('A'):
                continue
            if hasattr(missile, 'target_distance'):
                dist = float(getattr(missile, 'target_distance'))
                # Global CAP state should only enter EVADE for genuinely urgent
                # inbound threats; individual aircraft still have finer-grained
                # missile evasion logic in the tactical layer.
                if dist <= 40000.0:
                    return True
            else:
                return True
        except Exception:
            continue
    return False


def _is_fuel_critical(self, env) -> bool:
    """Detect whether our fuel state has reached the RTB threshold."""
    fuel_props = []
    for attr in ('propulsion_total_fuel_lbs', 'propulsion_tank0_contents_lbs'):
        try:
            prop = getattr(c, attr)
            if prop is not None:
                fuel_props.append(prop)
        except Exception:
            continue

    critical_count = 0
    alive_count = 0
    for aid in env.agents:
        if not aid.startswith('A') or not env.agents[aid].is_alive:
            continue
        alive_count += 1
        ac = env.agents[aid]
        fuel_lbs = None
        for prop in fuel_props:
            try:
                fuel_lbs = float(ac.get_property_value(prop))
                if fuel_lbs >= 0:
                    break
            except Exception:
                continue
        if fuel_lbs is None:
            continue

        if aid not in self._initial_fuel_lbs or self._initial_fuel_lbs[aid] <= 0:
            self._initial_fuel_lbs[aid] = max(1.0, fuel_lbs)

        threshold = self._initial_fuel_lbs[aid] * 0.5
        if fuel_lbs <= threshold:
            critical_count += 1

    if alive_count == 0:
        return False
    return critical_count >= max(1, alive_count // 2)
