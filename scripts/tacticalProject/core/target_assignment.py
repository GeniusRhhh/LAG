"""
目标分配模块 - 固定1v1对应关系
确保整个系统中目标分配的一致性
"""

def get_fixed_target(my_agent_id: str) -> str:
    """
    获取固定的目标敌机ID
    
    固定对应关系：
    - A0100 → B0100 (我方长机 vs 敌方长机)
    - A0200 → B0200 (我方僚机 vs 敌方僚机)
    - B0100 → A0100 (敌方长机 vs 我方长机)
    - B0200 → A0200 (敌方僚机 vs 我方僚机)
    
    Args:
        my_agent_id: 我方飞机ID
        
    Returns:
        str: 目标敌机ID
    """
    target_map = {
        'A0100': 'B0100',  # 我方长机 → 敌方长机
        'A0200': 'B0200',  # 我方僚机 → 敌方僚机
        'B0100': 'A0100',  # 敌方长机 → 我方长机
        'B0200': 'A0200',  # 敌方僚机 → 我方僚机
    }
    
    return target_map.get(my_agent_id, 'B0100')  # 默认返回B0100


def get_backup_target(my_agent_id: str, primary_target_alive: bool) -> str:
    """
    获取备用目标（当主目标被击落时）
    
    Args:
        my_agent_id: 我方飞机ID
        primary_target_alive: 主目标是否存活
        
    Returns:
        str: 备用目标ID
    """
    if primary_target_alive:
        return get_fixed_target(my_agent_id)
    
    # 主目标被击落，选择另一个敌机
    if my_agent_id.startswith('A'):
        # 我方飞机，选择另一个敌机
        primary = get_fixed_target(my_agent_id)
        return 'B0200' if primary == 'B0100' else 'B0100'
    else:
        # 敌方飞机，选择另一个我方飞机
        primary = get_fixed_target(my_agent_id)
        return 'A0200' if primary == 'A0100' else 'A0100'


def get_target_with_fallback(my_agent_id: str, env) -> str:
    """
    获取目标（带回退机制）
    
    优先选择固定目标，如果固定目标不存活则选择备用目标
    
    Args:
        my_agent_id: 我方飞机ID
        env: 环境对象
        
    Returns:
        str: 目标ID
    """
    # 获取固定目标
    primary_target = get_fixed_target(my_agent_id)
    
    # 检查主目标是否存活
    if primary_target in env.agents and env.agents[primary_target].is_alive:
        return primary_target
    
    # 主目标不存活，获取备用目标
    backup_target = get_backup_target(my_agent_id, False)
    
    # 检查备用目标是否存活
    if backup_target in env.agents and env.agents[backup_target].is_alive:
        return backup_target
    
    # 都不存活，返回主目标（让调用者处理）
    return primary_target


def is_my_team(agent_id: str) -> bool:
    """
    判断是否是我方飞机
    
    Args:
        agent_id: 飞机ID
        
    Returns:
        bool: True表示我方，False表示敌方
    """
    return agent_id.startswith('A')


def get_team_members(agent_id: str) -> list:
    """
    获取队友列表
    
    Args:
        agent_id: 飞机ID
        
    Returns:
        list: 队友ID列表（包括自己）
    """
    if agent_id.startswith('A'):
        return ['A0100', 'A0200']
    else:
        return ['B0100', 'B0200']


def get_enemy_team(agent_id: str) -> list:
    """
    获取敌方队伍列表
    
    Args:
        agent_id: 飞机ID
        
    Returns:
        list: 敌方ID列表
    """
    if agent_id.startswith('A'):
        return ['B0100', 'B0200']
    else:
        return ['A0100', 'A0200']

