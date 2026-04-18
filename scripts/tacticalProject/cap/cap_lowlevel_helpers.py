"""CAP low-level control helpers extracted from CAPTask.

The goal is structural refactoring only. Control behavior and thresholds are
kept identical to the previous in-class implementation.
"""

from __future__ import annotations

def _speed_error_to_spd_cmd(self, speed_error_mps: float) -> int:
    if speed_error_mps >= 60.0:
        return 6
    if speed_error_mps >= 30.0:
        return 5
    if speed_error_mps >= 5.0:
        return 4
    if speed_error_mps <= -60.0:
        return 0
    if speed_error_mps <= -30.0:
        return 1
    if speed_error_mps <= -5.0:
        return 2
    return 3


def _select_lowlevel_model_for_agent(self, agent_id: str):
    is_enemy = agent_id.startswith('B')
    preferred_type = self.enemy_lowlevel_type if is_enemy else self.friend_lowlevel_type

    if preferred_type == 'SU27' and self.su27_model is not None:
        return 'su27_legacy', self.su27_model

    if preferred_type == 'F16':
        if is_enemy and getattr(self, 'enemy_use_f16_native', False) and self.f16_cap_model is not None:
            semantics = str(getattr(self, 'enemy_lowlevel_semantics', '') or '').strip().lower()
            if 'legacy' in semantics:
                return 'f16_legacy', self.f16_cap_model
            if 'residual' in semantics:
                return 'f16_native_residual', self.f16_cap_model
            return 'f16_native_direct', self.f16_cap_model
        if self.f16_model is not None:
            return 'f16_legacy', self.f16_model

    if self.su27_model is not None:
        return 'su27_legacy', self.su27_model
    if getattr(self, 'enemy_use_f16_native', False) and self.f16_cap_model is not None:
        semantics = str(getattr(self, 'enemy_lowlevel_semantics', '') or '').strip().lower()
        if 'legacy' in semantics:
            return 'f16_legacy', self.f16_cap_model
        if 'residual' in semantics:
            return 'f16_native_residual', self.f16_cap_model
        return 'f16_native_direct', self.f16_cap_model
    if self.f16_model is not None:
        return 'f16_legacy', self.f16_model
    return 'none', None
