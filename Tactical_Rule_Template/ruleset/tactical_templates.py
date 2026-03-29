import numpy as np
from envs.JSBSim.core.simulatior import AircraftSimulator

def crank_strategy_3d(env, agent_id, radar):
    ego = env.agents[agent_id]
    shoot=0
    if not any(sim.uid.startswith(agent_id) for sim in env._tempsims.values()):
        # 尚未发射导弹，执行接敌
        shoot=1
        return [1, 2, 0,shoot]  # 保持高度、正前方、加速

    # 获取自身位置/速度
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_heading = ego_vel / (np.linalg.norm(ego_vel) + 1e-6)

    best_enemy = None
    min_R = float("inf")
    for enemy in ego.enemies:
        R = np.linalg.norm(np.array(enemy.get_position()) - ego_pos)
        if R < min_R:
            min_R = R
            best_enemy = enemy

    if best_enemy is None:
        return [1, 2, 1,0]  # 无敌人时保持飞行

    # 计算敌人相对方向向量（从自己指向敌人），并归一化为单位向量
    enemy_pos = np.array(best_enemy.get_position())  # 敌人坐标
    relative_vec = enemy_pos - ego_pos  # 敌人相对我方的位置向量
    rel_dir = relative_vec / (np.linalg.norm(relative_vec) + 1e-6)  # 单位化后的方向向量（避免除以0）

    # ===== 计算水平偏角（水平投影夹角） =====
    ego_heading_xy = ego_heading.copy()
    ego_heading_xy[2] = 0  # 自身速度方向在水平面的投影
    rel_dir_xy = rel_dir.copy()
    rel_dir_xy[2] = 0  # 敌人方向在水平面的投影

    # 如果水平投影接近0向量，设水平偏角为0
    if np.linalg.norm(ego_heading_xy) < 1e-3 or np.linalg.norm(rel_dir_xy) < 1e-3:
        theta_h = 0
    else:
        # 计算两向量角度
        theta_h = np.rad2deg(np.arccos(np.clip(
            np.dot(ego_heading_xy, rel_dir_xy) /
            (np.linalg.norm(ego_heading_xy) * np.linalg.norm(rel_dir_xy) + 1e-6),
            -1, 1)))

        # 用右手法则判断夹角方向（左负右正）
        cross_h = np.cross(ego_heading_xy, rel_dir_xy)
        theta_h *= -np.sign(cross_h[2])  # cross 的z分量决定是左侧还是右侧

    # ===== 计算垂直偏角（高度方向上的偏差） =====
    ego_heading_xz = ego_heading.copy()
    ego_heading_xz[1] = 0  # 投影到 XZ 平面，忽略 Y 分量
    rel_dir_xz = rel_dir.copy()
    rel_dir_xz[1] = 0  # 同样投影

    if np.linalg.norm(ego_heading_xz) < 1e-3 or np.linalg.norm(rel_dir_xz) < 1e-3:
        theta_v = 0
    else:
        theta_v = np.rad2deg(np.arccos(np.clip(
            np.dot(ego_heading_xz, rel_dir_xz) /
            (np.linalg.norm(ego_heading_xz) * np.linalg.norm(rel_dir_xz) + 1e-6),
            -1, 1)))

        # 用叉乘确定俯仰方向（Z 轴在上，Y 被忽略 → 用 Y 分量判断上下）
        cross_v = np.cross(ego_heading_xz, rel_dir_xz)
        theta_v *= -np.sign(cross_v[1])  # 右手法则：上为正，下为负

    # === Crank 控制逻辑 ===
    crank_angle_deg = np.rad2deg(0.8 * radar.fai_width)
    max_pitch_offset = np.rad2deg(0.8 * radar.theta_height)

    # 水平 crank：控制 heading_cmd
    if theta_h >= 0:
        if theta_h < crank_angle_deg:
            heading_cmd = 3  # 右转
        elif theta_h > crank_angle_deg:
            heading_cmd = 1  # 轻左转，纠偏
        else:
            heading_cmd = 2  # 保持
    else:
        if theta_h > -crank_angle_deg:
            heading_cmd = 1  # 左转
        elif theta_h < -crank_angle_deg:
            heading_cmd = 3  # 右转
        else:
            heading_cmd = 2  # 保持

    # 垂直 crank：控制 alt_cmd
    if theta_v >= 0:
        if theta_v < max_pitch_offset:
            alt_cmd = 0  # 向上（目标方向内）
        elif theta_v > max_pitch_offset:
            alt_cmd = 2  # 向下修正
        else:
            alt_cmd = 1  # 保持
    else:
        if theta_v > -max_pitch_offset:
            alt_cmd = 2  # 向下（目标方向内）
        elif theta_v < -max_pitch_offset:
            alt_cmd = 0  # 向上修正
        else:
            alt_cmd = 1  # 保持

    vel_cmd = 1  # 保持中速，防止过快闭合或过慢被追击
    return [alt_cmd, heading_cmd, vel_cmd,shoot]


def beam_strategy_3d(env, agent_id):
    ego = env.agents[agent_id]
    # 获取自身位置/速度
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_heading = ego_vel / (np.linalg.norm(ego_vel) + 1e-6)

    # 寻找最近的敌人
    best_enemy = None
    min_R = float("inf")
    for enemy in ego.enemies:
        R = np.linalg.norm(np.array(enemy.get_position()) - ego_pos)
        if R < min_R:
            min_R = R
            best_enemy = enemy

    # for enemy in ego.enemies:
    #     if not enemy.is_alive:
    #         print("被击毁")

    shoot = 0  # 默认不发射
    if best_enemy is None:
        return [1, 2, 1,shoot]  # 没有敌人，保持当前飞行状态

    # 计算相对位置和单位方向
    enemy_pos = np.array(best_enemy.get_position())
    relative_vec = enemy_pos - ego_pos
    rel_dir = relative_vec / (np.linalg.norm(relative_vec) + 1e-6)

    rel_vel = best_enemy.get_velocity() - ego_vel
    rel_vel_unit = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)
    # 计算水平面投影的方向
    rel_vel_xy = rel_vel_unit.copy()

    rel_vel_xy[2] = 0
    los_xy = rel_dir.copy()
    los_xy[2] = 0

    if np.linalg.norm(rel_vel_xy) < 1e-3 or np.linalg.norm(los_xy) < 1e-3:
        theta_h = 0
    else:
        theta_h = np.rad2deg(np.arccos(np.clip(
            np.dot(rel_vel_xy, los_xy) /
            (np.linalg.norm(rel_vel_xy) * np.linalg.norm(los_xy) + 1e-6),
            -1, 1)))
        cross_h = np.cross(rel_vel_xy, los_xy)
        theta_h *= np.sign(cross_h[2])  # 右为正，左为负

    # Beam 理想角度是 90 度（即相对方向与航向垂直）
    target_theta = 90.0
    tolerance = 5.0  # 容差

    # 判断是朝左还是朝右转以逼近 90 度
    if theta_h >= 0:
        # 相对速度在 LOS 右侧，目标夹角是 +90°
        if theta_h < target_theta - tolerance:
            heading_cmd = 3  # 右转
        elif theta_h > target_theta + tolerance:
            heading_cmd = 1  # 左转回来
        else:
            heading_cmd = 2  # 保持
    else:
        # 相对速度在 LOS 左侧，目标夹角是 -90°
        if theta_h > -target_theta + tolerance:
            heading_cmd = 1  # 左转靠近 -90°
        elif theta_h < -target_theta - tolerance:
            heading_cmd = 3  # 右转回来
        else:
            heading_cmd = 2  # 保持

    # 仅控制水平偏航，不控制高度
    alt_cmd = 1  # 保持高度
    vel_cmd = 1  # 保持中速（避免太快/太慢）

    #print(f"[BEAM] θ_h = {theta_h:.2f}°, heading_cmd = {heading_cmd}")
    return [alt_cmd, heading_cmd, vel_cmd,shoot]


def ortesc_strategy_3d(env, agent_id):
    """
    垂直机动
    OrtEsc（Orthogonal Escape）机动策略：
    保持自身速度方向垂直于敌方导弹 LOS，最大化横向机动，消耗导弹能量。
    """
    ego = env.agents[agent_id]
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_speed = np.linalg.norm(ego_vel)
    ego_normvec = ego_vel / (ego_speed + 1e-6)

    # 查找所有导向ego的敌方导弹
    threat_missiles = [
        m for m in ego.under_missiles
        if m.is_alive and m.target_aircraft and m.target_aircraft.uid == ego.uid
    ]

    shoot = 0  # 默认不发射
    if len(threat_missiles) == 0:
        return [1, 2, 1,shoot]  # 无导弹威胁，保持平飞中速

    # 获取目标 LOS 方向（共线或不共线）
    if len(threat_missiles) == 1:
        # 单导弹：与 LOS 垂直即可
        los_vec = np.array(threat_missiles[0].get_position()) - ego_pos
        los_unit = los_vec / (np.linalg.norm(los_vec) + 1e-6)
    else:
        # 多导弹（>=2）：构建张成平面
        los1 = np.array(threat_missiles[0].get_position()) - ego_pos
        los2 = np.array(threat_missiles[1].get_position()) - ego_pos
        # 计算每个导弹的距离
        distance1 = np.linalg.norm(los1)
        distance2 = np.linalg.norm(los2)
        # 选择距离最近的导弹
        if distance1 < distance2:
            los_unit = los1 / (np.linalg.norm(los1) + 1e-6)
        else:
            los_unit = los2 / (np.linalg.norm(los2) + 1e-6)

    # 计算速度方向与 los_unit 的夹角
    if ego_speed < 1e-3 or np.linalg.norm(los_unit) < 1e-3:
        return [1, 2, 1]  # 停滞或方向异常，保持当前

    cos_theta = np.clip(np.dot(ego_normvec, los_unit), -1.0, 1.0)
    theta = np.rad2deg(np.arccos(cos_theta))

    # 控制目标夹角为 90°（即速度与 LOS 垂直）
    target_theta = 90.0
    tolerance = 5.0  # 容差范围
    cross = np.cross(ego_normvec, los_unit)
    sign = np.sign(cross[2])  # 判断左右转（右为正，左为负）
    #print(theta)
    # 判断是朝左还是朝右转以逼近 90 度
    if sign >= 0:
        # 敌人在右侧，目标夹角是 90°
        if theta < target_theta - tolerance:
            heading_cmd = 1
        elif theta > target_theta + tolerance:
            heading_cmd = 3
        else:
            heading_cmd = 2  # 保持
    else:
        # 敌人在左侧，目标夹角是 90°
        if theta < target_theta - tolerance:
            heading_cmd = 3
        elif theta > target_theta + tolerance:
            heading_cmd = 1
        else:
            heading_cmd = 2  # 保持

    alt_cmd = 1  # 保持高度
    vel_cmd = 2  # 加速

    # print(f"[OrtEsc] θ = {theta:.2f}°, heading_cmd = {heading_cmd}")
    return [alt_cmd, heading_cmd, vel_cmd,shoot]

def skate_strategy_3d(env, agent_id, radar):
    """
    Skate 机动策略（完整流程版）：
    发射导弹 → Crank 脱离 → Turn Cold 背离 → 回头接敌 → 二次开火 → 最终脱离
    每阶段根据相对距离和角度动态推进，调用 crank_strategy_3d 实现精确脱离
    """
    ego = env.agents[agent_id]

    # 初始化阶段变量（只执行一次）
    if not hasattr(ego, "skate_phase"):
        ego.skate_phase = 0                # 当前阶段标志
        ego.skate_flag_shot = False        # 是否已发射第一枚导弹

    # 选择最近敌机
    best_enemy = None
    min_R = float("inf")
    for enemy in ego.enemies:
        R = np.linalg.norm(np.array(enemy.get_position()) - np.array(ego.get_position()))
        if R < min_R:
            min_R = R
            best_enemy = enemy

    # 若无敌人，维持飞行状态
    if best_enemy is None:
        return [1, 2, 1, 0]

    # 获取敌我相对方向
    ego_pos = np.array(ego.get_position())
    tar_pos = np.array(best_enemy.get_position())
    rel_vec = tar_pos - ego_pos
    rel_dist = np.linalg.norm(rel_vec)  # 计算敌我距离（单位：米）
    rel_dir = rel_vec / (rel_dist + 1e-6)  # 将相对位置向量单位化（归一化为长度为1的方向向量）

    # 相对角度（用于判断回头阶段）
    ego_vel = np.array(ego.get_velocity())
    ego_dir = ego_vel / (np.linalg.norm(ego_vel) + 1e-6)
    cos_theta = np.clip(np.dot(ego_dir, rel_dir), -1.0, 1.0)
    rel_angle = np.rad2deg(np.arccos(cos_theta))

    # 获取姿态角，用于判断是否雷达锁定
    roll, pitch, yaw = ego.get_rpy() * 180 / np.pi
    is_locked, _ = radar.detect(
        ego_pos=ego_pos,
        ego_vel=ego_vel,
        ego_yaw=yaw,
        ego_pitch=pitch,
        tar_pos=tar_pos,
        tar_vel=np.array(best_enemy.get_velocity())
    )

    # 阈值设定
    MAR = 20000        # Missile Avoidance Range（最小脱离距离）
    safe_sep = 22000   # 安全接敌距离

    # 默认动作
    alt_cmd = 1       # 保持高度
    heading_cmd = 2   # 正前飞
    vel_cmd = 1       # 中速
    shoot = 0         # 默认不开火

    # -------- Skate 战术阶段控制 --------
    print("阶段：", ego.skate_phase)
    # Phase 0：首次开火前，保持正前飞，锁定后开火
    if ego.skate_phase == 0:
        if is_locked:
            shoot = 1
            ego.skate_flag_shot = True
            ego.skate_phase = 1  # 转入 Crank 阶段

    # Phase 1：Crank 脱离，调用 crank 策略
    elif ego.skate_phase == 1:
        alt_cmd, heading_cmd, vel_cmd,shoot = crank_strategy_3d(env, agent_id, radar)
        if rel_dist < MAR:
            ego.skate_phase = 2  # 转入 Cold 背离

    # Phase 2：Turn Cold 背离
    elif ego.skate_phase == 2:
        # 如果夹角小于150°，说明未真正背离 → 调整航向
        if rel_angle < 150:
            heading_cmd = 1
            # cross = np.cross(ego_dir, rel_dir)
            # heading_cmd = 1 if cross[2] < 0 else 3  # 判断向左还是向右转，直到背离。敌人在右边，就左转远离；敌人在左边，就右转远离
        else:
            heading_cmd = 2  # 已背离，保持航向
        if rel_dist > safe_sep or len(best_enemy.launch_missiles) >= 1:
            ego.skate_phase = 3  # 进入下一阶段：转回接敌
        print(
            f"[DEBUG] Phase 2 - rel_angle: {rel_angle:.1f}, rel_dist: {rel_dist / 1000:.1f} km, heading_cmd: {heading_cmd}")


    # Phase 3：转回接敌
    elif ego.skate_phase == 3:
        cross = np.cross(ego_dir, rel_dir)
        heading_cmd = 1 if cross[2] < 0 else 3
        if rel_angle < 30:  # 面向敌方接近正向
            ego.skate_phase = 4

    # Phase 4：二次接敌，若锁定则开火
    elif ego.skate_phase == 4:
        heading_cmd = 2
        if is_locked and rel_dist < MAR:
            shoot = 1
            ego.skate_phase = 5  # 最终脱离

    # Phase 5：最终 Cold 脱离
    elif ego.skate_phase == 5:
        if rel_angle < 150:
            cross = np.cross(ego_dir, rel_dir)
            heading_cmd = 1 if cross[2] < 0 else 3
        else:
            heading_cmd = 2  # 已背离，保持航向
        shoot = 0

    return [alt_cmd, heading_cmd, vel_cmd, shoot]

def short_skate_strategy_3d(env, agent_id, radar):
    """
    Short Skate 机动策略：
    发射导弹 → Turn Cold 背离 → 最终脱离
    取消了第二次接敌和第二轮导弹发射步骤
    """
    ego = env.agents[agent_id]

    # 初始化阶段变量（只执行一次）
    if not hasattr(ego, "skate_phase"):
        ego.skate_phase = 0                # 当前阶段标志
        ego.skate_flag_shot = False        # 是否已发射第一枚导弹

    # 选择最近敌机
    best_enemy = None
    min_R = float("inf")
    for enemy in ego.enemies:
        R = np.linalg.norm(np.array(enemy.get_position()) - np.array(ego.get_position()))
        if R < min_R:
            min_R = R
            best_enemy = enemy

    # 若无敌人，维持飞行状态
    if best_enemy is None:
        return [1, 2, 1, 0]

    # 获取敌我相对方向
    ego_pos = np.array(ego.get_position())
    tar_pos = np.array(best_enemy.get_position())
    rel_vec = tar_pos - ego_pos
    rel_dist = np.linalg.norm(rel_vec)  # 计算敌我距离（单位：米）
    rel_dir = rel_vec / (rel_dist + 1e-6)  # 将相对位置向量单位化（归一化为长度为1的方向向量）

    # 相对角度（用于判断背离阶段）
    ego_vel = np.array(ego.get_velocity())
    ego_dir = ego_vel / (np.linalg.norm(ego_vel) + 1e-6)
    cos_theta = np.clip(np.dot(ego_dir, rel_dir), -1.0, 1.0)
    rel_angle = np.rad2deg(np.arccos(cos_theta))

    # 获取姿态角，用于判断是否雷达锁定
    roll, pitch, yaw = ego.get_rpy() * 180 / np.pi
    is_locked, _ = radar.detect(
        ego_pos=ego_pos,
        ego_vel=ego_vel,
        ego_yaw=yaw,
        ego_pitch=pitch,
        tar_pos=tar_pos,
        tar_vel=np.array(best_enemy.get_velocity())
    )

    # 阈值设定
    MAR = 20000        # Missile Avoidance Range（最小脱离距离）
    safe_sep = 25000   # 安全接敌距离

    # 默认动作
    alt_cmd = 1       # 保持高度
    heading_cmd = 2   # 正前飞
    vel_cmd = 1       # 中速
    shoot = 0         # 默认不开火

    # -------- Short Skate 战术阶段控制 --------
    #print("阶段：", ego.skate_phase)

    # Phase 0：首次开火前，保持正前飞，锁定后开火
    if ego.skate_phase == 0:
        if is_locked:
            shoot = 1
            ego.skate_flag_shot = True
            ego.skate_phase = 1  # 转入 Turn Cold 阶段

    # Phase 1：Turn Cold 背离
    elif ego.skate_phase == 1:
        # 如果夹角小于150°，说明未真正背离 → 调整航向
        if rel_angle < 150:
            heading_cmd = 1
            # cross = np.cross(ego_dir, rel_dir)
            # heading_cmd = 1 if cross[2] < 0 else 3  # 判断向左还是向右转，直到背离。敌在右边，就左转远离；敌在左边，就右转远离
        else:
            ego.skate_phase = 2  # 进入最终脱离阶段

    # Phase 2：最终 Cold 脱离
    elif ego.skate_phase == 2:
        heading_cmd = 2  # 背离飞行，不再交战
        shoot = 0  # 确保不再发射导弹

    return [alt_cmd, heading_cmd, vel_cmd, shoot]

def max_g_break_strategy_3d(env, agent_id):
    """
    Max-G Break（末段大过载急转）
    目标：把导弹 LOS 压到我机 3/9 线（≈±90°），持续同向大转弯。
    仅在已有来袭导弹时由 RL 选择本模板；不处理“无导弹”分支。
    返回: [alt_cmd, heading_cmd, vel_cmd, shoot]
    """
    ego = env.agents[agent_id]
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())

    # 筛选所有正在攻击我机的存活导弹
    threatening_missiles = [
        mm for mm in ego.under_missiles
        if mm.is_alive
           and mm.target_aircraft
           and mm.target_aircraft.uid == ego.uid
    ]
    # 如果没有威胁导弹 → 返回“平飞”动作，防止报错
    if not threatening_missiles:
        alt_cmd = 1  # 保持高度
        heading_cmd = 2  # 保持航向（不转弯）
        vel_cmd = 1  # 保持速度
        shoot = 0  # 不发射
        return [alt_cmd, heading_cmd, vel_cmd, shoot]
    # 选择最近的一枚导弹
    m = min(
        threatening_missiles,
        key=lambda mm: np.linalg.norm(np.array(mm.get_position()) - ego_pos)
    )

    # NEU 平面几何：只用水平投影
    ego_xy = ego_vel.copy(); ego_xy[2] = 0
    ego_xy /= (np.linalg.norm(ego_xy) + 1e-6)

    los = np.array(m.get_position()) - ego_pos
    los_xy = los.copy(); los_xy[2] = 0
    los_xy /= (np.linalg.norm(los_xy) + 1e-6)

    # 与 LOS 的夹角及左右关系
    cos_th = np.clip(np.dot(ego_xy, los_xy), -1.0, 1.0)
    theta = np.rad2deg(np.arccos(cos_th))          # [0,180]
    cross_z = np.cross(ego_xy, los_xy)[2]          # NEU：符号用于判左右

    # 使用 ego 的自定义属性记录当前是否正在执行大转及其方向
    if not hasattr(ego, '_max_g_break_turn_dir'):
        ego._max_g_break_turn_dir = None  # 初始化：未启动
    if not hasattr(ego, 'turn'):
        ego.turn = 0  # 初始化：未启动

    if abs(theta) < 120 and ego.turn==0:
        # -------------------------------
        # 需要大过载急转：启动或维持方向
        # -------------------------------
        if ego._max_g_break_turn_dir is None:
            # 第一次进入 → 根据 cross_z 决定转向方向
            if cross_z >= 0:
                ego._max_g_break_turn_dir = 0
                #print("左转", agent_id)
            else:
                ego._max_g_break_turn_dir = 4
                #print("右转", agent_id)
        # 无论 cross_z 如何变化，都保持初始方向
        heading_cmd = ego._max_g_break_turn_dir
    else:
        ego.turn=1
        heading_cmd = 2  # 无锁定时保持

    alt_cmd = 1                   # 保持高度（水平大过载为主）
    vel_cmd = 2                   # 加速度
    shoot   = 0                   # 防御机动不发射

    return [alt_cmd, heading_cmd, vel_cmd, shoot]
