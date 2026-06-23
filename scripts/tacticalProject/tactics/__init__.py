from .base_tactic import BaseTactic
from .drag_shoot import DragShootTactic, DragShootDecisionManager
from .pincer_attack import PincerAttackTactic, PincerAttackDecisionManager
from .high_low_attack import HighLowAttackTactic, HighLowAttackDecisionManager
from .side_by_side import SideBySideTactic, SideBySideDecisionManager
from .tactical_evasion import TacticalEvasion
from .tactical_turn import TacticalTurn

# 前后攻击使用sequential_attack
from .sequential_attack import SequentialAttackTactic as FrontBackAttackTactic

__all__ = [
    'BaseTactic',
    'DragShootTactic', 'DragShootDecisionManager',
    'PincerAttackTactic', 'PincerAttackDecisionManager',
    'HighLowAttackTactic', 'HighLowAttackDecisionManager',
    'SequentialAttackTactic', 'FrontBackAttackTactic',
    'SideBySideTactic', 'SideBySideDecisionManager',
    'TacticalEvasion',
    'TacticalTurn',
]
