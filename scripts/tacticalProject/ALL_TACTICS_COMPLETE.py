# 所有5个战术的完整实现 - 严格按照原战术模板
# 这个文件包含所有战术的完整代码，可以直接复制到tactical_task.py中

# ==================== 战术1：拖曳射击 ====================
def _execute_drag_shoot_COMPLETE(self, env, agent_id: str) -> tuple:
    """
    战术1: 拖曳射击 - 严格按照drag_shoot_tactical_task.py
    """
    is_lead = agent_id.endswith('100')
    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
    current_time = env.current_step * env.time_interval
    
    # 检查Short Skate状态
    if agent_id in self.short_skate_states:
        return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 长机动作
    if is_lead:
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            return self._maintain_heading_precise(env, agent_id, 0.0)
        elif self.current_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, 0.0)
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 僚机动作（独立阶段判断）
    else:
        leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
        if leader_blue and leader_blue.is_alive:
            distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
            wingman_phase = self._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self.current_phase
        
        if wingman_phase == TacticalPhase.NLT_MELD:
            return self._maintain_heading_precise(env, agent_id, 68.0)
        elif wingman_phase == TacticalPhase.MELD_MTR:
            return self._maintain_heading_precise(env, agent_id, 0.0)
        elif wingman_phase in [TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            return self._maintain_heading_precise(env, agent_id, 0.0)
        elif wingman_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, 350.0)
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)


# ==================== 战术2：钳形攻势 ====================  
def _execute_pincer_attack_COMPLETE(self, env, agent_id: str) -> tuple:
    """
    战术2: 钳形攻势 - 严格按照pincer_attack_tactical_task_complete.py
    """
    is_lead = agent_id.endswith('100')
    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
    current_time = env.current_step * env.time_interval
    
    # 检查Short Skate状态
    if agent_id in self.short_skate_states:
        return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 长机动作（左侧Crank）
    if is_lead:
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            # 长机左侧Crank -45°
            return self._maintain_heading_precise(env, agent_id, -45.0)
        elif self.current_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, -45.0)
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 僚机动作（右侧Crank，独立阶段判断）
    else:
        leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
        if leader_blue and leader_blue.is_alive:
            distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
            wingman_phase = self._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self.current_phase
        
        if wingman_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            # 僚机右侧Crank +45°
            return self._maintain_heading_precise(env, agent_id, 45.0)
        elif wingman_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, 45.0)
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)


# ==================== 战术3：上下夹击 ====================
def _execute_high_low_attack_COMPLETE(self, env, agent_id: str) -> tuple:
    """
    战术3: 上下夹击 - 严格按照high_low_attack_fixed.py
    """
    is_lead = agent_id.endswith('100')
    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
    current_alt = env.agents[agent_id].get_position()[2]
    current_time = env.current_step * env.time_interval
    
    # 检查Short Skate状态
    if agent_id in self.short_skate_states:
        return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 长机动作（保持低空）
    if is_lead:
        target_altitude = 6096.0  # 保持低空6000m
        
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            # 保持低空0°航向
            alt_diff = target_altitude - current_alt
            if abs(alt_diff) > 100:
                alt_cmd = 11 if alt_diff > 0 else 3  # 爬升或下降
            else:
                alt_cmd = 7  # 保持高度
            return alt_cmd, 8, 3
        elif self.current_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return 7, 8, 3
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 僚机动作（爬升高空，独立阶段判断）
    else:
        leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
        if leader_blue and leader_blue.is_alive:
            distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
            wingman_phase = self._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self.current_phase
        
        target_altitude = 8000.0  # 目标高空8000m
        
        if wingman_phase == TacticalPhase.NLT_MELD:
            # 保持0°航向
            return 7, 8, 3
        elif wingman_phase == TacticalPhase.MELD_MTR:
            # 爬升到高空
            alt_diff = target_altitude - current_alt
            if abs(alt_diff) > 100:
                alt_cmd = 14  # 爬升+1500m
            else:
                alt_cmd = 7  # 保持高度
            return alt_cmd, 8, 3
        elif wingman_phase in [TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            # 保持高空
            return 7, 8, 3
        elif wingman_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                # 俯冲攻击
                return 5, 8, 4  # 下降-150m + 加速
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)


# ==================== 战术4：前后攻击 ====================
def _execute_sequential_attack_COMPLETE(self, env, agent_id: str) -> tuple:
    """
    战术4: 前后攻击 - 严格按照front_back_attack_final_task.py
    """
    is_lead = agent_id.endswith('100')
    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
    current_time = env.current_step * env.time_interval
    
    # 检查Short Skate状态
    if agent_id in self.short_skate_states:
        return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 长机动作
    if is_lead:
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            return self._maintain_heading_precise(env, agent_id, 0.0)
        elif self.current_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, 0.0)
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 僚机动作（藏在长机后方，独立阶段判断）
    else:
        leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
        if leader_blue and leader_blue.is_alive:
            distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
            wingman_phase = self._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self.current_phase
        
        # 僚机保持0°航向（藏在长机后方3海里）
        if wingman_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            return self._maintain_heading_precise(env, agent_id, 0.0)
        elif wingman_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, 0.0)
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)


# ==================== 战术5：并排射击 ====================
def _execute_side_by_side_COMPLETE(self, env, agent_id: str) -> tuple:
    """
    战术5: 并排射击 - 严格按照side_by_side_shooting_tactical_task.py
    """
    is_lead = agent_id.endswith('100')
    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
    current_time = env.current_step * env.time_interval
    
    # 检查Short Skate状态
    if agent_id in self.short_skate_states:
        return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 长机动作
    if is_lead:
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            return self._maintain_heading_precise(env, agent_id, 0.0)
        elif self.current_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, 0.0)
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)
    
    # 僚机动作（保持编队间距，独立阶段判断）
    else:
        leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
        if leader_blue and leader_blue.is_alive:
            distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
            wingman_phase = self._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self.current_phase
        
        if wingman_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            return self._maintain_heading_precise(env, agent_id, 0.0)
        elif wingman_phase == TacticalPhase.TR_DOR:
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, 0.0)
        else:  # DOR_DR及之后
            return self._execute_short_skate_precise(env, agent_id, current_time)
