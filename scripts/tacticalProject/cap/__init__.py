"""CAP战术任务模块"""
from .cap_task import CAPTask
from .patrol_task import SimplePatrolTask
from .coordinate_system import CoordinateSystem, BattlefieldConfig
from .config_manager import ConfigManager, CAPConfig, AircraftPosition
from .faor_manager import FAORManager, RiskZone, FAORBoundary
from .formation_manager import FormationManager, PatrolPhase, FormationRole, FormationMember
from .patrol_state_machine import PatrolStateMachine, PatrolState, PatrolBox
