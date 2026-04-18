"""
任务分配算法模块
- 简单版：贪婪分配（按距离）
- 专业版：匈牙利算法（最优分配）
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Assignment:
    """分配结果"""
    agent_id: str
    target_id: str
    priority: float  # 优先级（距离、威胁等级综合）
    distance: float  # 距离(km)


class SimpleTaskAllocator:
    """简单任务分配器 - 贪婪最近邻"""
    
    def allocate(self, 
                 agents: Dict[str, Tuple[float, float]],  # {id: (x, y)}
                 targets: Dict[str, Tuple[float, float]],  # {id: (x, y)}
                 max_per_agent: int = 2) -> List[Assignment]:
        """贪婪分配：每个目标分配给最近的飞机
        
        Args:
            agents: 我方飞机位置
            targets: 敌方目标位置
            max_per_agent: 每机最大分配目标数
            
        Returns:
            分配结果列表
        """
        if not agents or not targets:
            return []
        
        assignments = []
        agent_counts = {aid: 0 for aid in agents}
        assigned_targets = set()
        
        # 计算所有距离对
        candidates = []
        for tid, t_pos in targets.items():
            for aid, a_pos in agents.items():
                dist = np.sqrt((a_pos[0] - t_pos[0])**2 + (a_pos[1] - t_pos[1])**2)
                candidates.append((dist, aid, tid))
        
        # 按距离排序
        candidates.sort(key=lambda x: x[0])
        
        # 贪婪分配
        for dist, aid, tid in candidates:
            if tid in assigned_targets:
                continue
            if agent_counts[aid] >= max_per_agent:
                continue
            
            assignments.append(Assignment(
                agent_id=aid,
                target_id=tid,
                priority=1.0 / (dist + 1),
                distance=dist
            ))
            assigned_targets.add(tid)
            agent_counts[aid] += 1
            
            if len(assigned_targets) == len(targets):
                break
        
        return assignments


class HungarianTaskAllocator:
    """匈牙利算法任务分配器 - 最优分配
    
    使用Kuhn-Munkres算法求解最小成本分配问题
    """
    
    def allocate(self,
                 agents: Dict[str, Tuple[float, float]],
                 targets: Dict[str, Tuple[float, float]],
                 max_per_agent: int = 2) -> List[Assignment]:
        """匈牙利算法分配
        
        成本函数: 距离
        目标: 最小化总距离
        """
        if not agents or not targets:
            return []
        
        agent_ids = list(agents.keys())
        target_ids = list(targets.keys())
        n_agents = len(agent_ids)
        n_targets = len(target_ids)
        
        # 构建成本矩阵 (考虑每机可分配多目标)
        n_slots = n_agents * max_per_agent
        cost_matrix = np.full((n_slots, n_targets), np.inf)
        
        for i, aid in enumerate(agent_ids):
            a_pos = agents[aid]
            for j, tid in enumerate(target_ids):
                t_pos = targets[tid]
                dist = np.sqrt((a_pos[0] - t_pos[0])**2 + (a_pos[1] - t_pos[1])**2)
                # 每个agent有max_per_agent个槽位
                for k in range(max_per_agent):
                    slot_idx = i * max_per_agent + k
                    cost_matrix[slot_idx, j] = dist
        
        # 使用scipy的线性分配求解器（如果可用）
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            assignments = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < np.inf:
                    aid = agent_ids[r // max_per_agent]
                    tid = target_ids[c]
                    dist = cost_matrix[r, c]
                    assignments.append(Assignment(
                        agent_id=aid,
                        target_id=tid,
                        priority=1.0 / (dist + 1),
                        distance=dist
                    ))
            return assignments
        except ImportError:
            # fallback到贪婪算法
            simple = SimpleTaskAllocator()
            return simple.allocate(agents, targets, max_per_agent)


def create_task_allocator(use_hungarian: bool = False) -> SimpleTaskAllocator:
    """工厂函数：创建任务分配器"""
    if use_hungarian:
        return HungarianTaskAllocator()
    else:
        return SimpleTaskAllocator()
