"""Target assignment helpers shared by the tactical system."""


def get_fixed_target(my_agent_id: str) -> str:
    """Return the legacy fixed opponent mapping."""
    target_map = {
        'A0100': 'B0100',
        'A0200': 'B0200',
        'A0300': 'B0300',
        'A0400': 'B0400',
        'B0100': 'A0100',
        'B0200': 'A0200',
        'B0300': 'A0300',
        'B0400': 'A0400',
    }

    if my_agent_id in target_map:
        return target_map[my_agent_id]
    return 'B0100' if my_agent_id.startswith('A') else 'A0100'


def get_backup_target(my_agent_id: str, primary_target_alive: bool) -> str:
    """Return a backup target when the fixed target is unavailable."""
    if primary_target_alive:
        return get_fixed_target(my_agent_id)

    primary = get_fixed_target(my_agent_id)
    for candidate in get_enemy_team(my_agent_id):
        if candidate != primary:
            return candidate
    return primary


def _get_alive_enemy_ids(my_agent_id: str, env) -> list:
    return [
        agent_id
        for agent_id, aircraft in env.agents.items()
        if (not str(agent_id).startswith(my_agent_id[0])) and getattr(aircraft, 'is_alive', False)
    ]


def get_target_with_fallback(my_agent_id: str, env) -> str:
    """
    Return a dynamically selected target.

    Priority:
    1. My own closest live enemy within 65 km.
    2. Any teammate's closest live enemy within 55 km.
    3. Fixed target if still valid and not much worse than the nearest enemy.
    4. Global nearest live enemy.
    """
    try:
        from tactical_utils import TacticalUtils

        same_side_ids = [
            aid for aid, aircraft in env.agents.items()
            if str(aid).startswith(my_agent_id[0]) and getattr(aircraft, 'is_alive', False)
        ]
        enemy_ids = _get_alive_enemy_ids(my_agent_id, env)

        my_close_enemy = None
        my_close_distance = float('inf')
        team_close_enemy = None
        team_close_distance = float('inf')

        for enemy_id in enemy_ids:
            enemy = env.agents[enemy_id]
            try:
                distance_to_me = TacticalUtils.calculate_distance_between(env.agents[my_agent_id], enemy)
                if distance_to_me < my_close_distance:
                    my_close_distance = distance_to_me
                    my_close_enemy = enemy_id
            except Exception:
                pass

            for friendly_id in same_side_ids:
                try:
                    distance = TacticalUtils.calculate_distance_between(env.agents[friendly_id], enemy)
                except Exception:
                    continue
                if distance < team_close_distance:
                    team_close_distance = distance
                    team_close_enemy = enemy_id

        if my_close_enemy is not None and my_close_distance <= 65000:
            return my_close_enemy
        if team_close_enemy is not None and team_close_distance <= 55000:
            return team_close_enemy
    except Exception:
        pass

    primary_target = get_fixed_target(my_agent_id)

    if primary_target in env.agents and getattr(env.agents[primary_target], 'is_alive', False):
        try:
            from tactical_utils import TacticalUtils

            my_aircraft = env.agents[my_agent_id]
            primary_aircraft = env.agents[primary_target]
            primary_distance = TacticalUtils.calculate_distance_between(my_aircraft, primary_aircraft)

            min_distance = float('inf')
            best_target = primary_target
            for enemy_id in _get_alive_enemy_ids(my_agent_id, env):
                enemy_aircraft = env.agents[enemy_id]
                distance = TacticalUtils.calculate_distance_between(my_aircraft, enemy_aircraft)
                if distance < min_distance:
                    min_distance = distance
                    best_target = enemy_id

            if primary_distance > 85000 or (
                best_target != primary_target and min_distance <= 70000 and (primary_distance - min_distance) >= 10000
            ):
                return best_target

            return primary_target
        except Exception:
            return primary_target

    try:
        from tactical_utils import TacticalUtils

        my_aircraft = env.agents[my_agent_id]
        min_distance = float('inf')
        best_target = None

        for enemy_id in _get_alive_enemy_ids(my_agent_id, env):
            enemy_aircraft = env.agents[enemy_id]
            distance = TacticalUtils.calculate_distance_between(my_aircraft, enemy_aircraft)
            if distance < min_distance:
                min_distance = distance
                best_target = enemy_id

        if best_target:
            return best_target
    except Exception:
        pass

    return primary_target


def is_my_team(agent_id: str) -> bool:
    """Return True when the aircraft belongs to side A."""
    return agent_id.startswith('A')


def get_team_members(agent_id: str) -> list:
    """Return the nominal two-ship team for the aircraft."""
    if agent_id in ('A0100', 'A0200'):
        return ['A0100', 'A0200']
    if agent_id in ('A0300', 'A0400'):
        return ['A0300', 'A0400']
    if agent_id in ('B0100', 'B0200'):
        return ['B0100', 'B0200']
    if agent_id in ('B0300', 'B0400'):
        return ['B0300', 'B0400']
    if agent_id.startswith('A'):
        return ['A0100', 'A0200', 'A0300', 'A0400']
    return ['B0100', 'B0200', 'B0300', 'B0400']


def get_enemy_team(agent_id: str) -> list:
    """Return the nominal opponent team for the aircraft."""
    if agent_id in ('A0100', 'A0200'):
        return ['B0100', 'B0200']
    if agent_id in ('A0300', 'A0400'):
        return ['B0300', 'B0400']
    if agent_id in ('B0100', 'B0200'):
        return ['A0100', 'A0200']
    if agent_id in ('B0300', 'B0400'):
        return ['A0300', 'A0400']
    if agent_id.startswith('A'):
        return ['B0100', 'B0200', 'B0300', 'B0400']
    return ['A0100', 'A0200', 'A0300', 'A0400']
