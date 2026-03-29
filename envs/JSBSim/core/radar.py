import numpy as np

class RadarModel:
    def __init__(self,
                 max_range=80000,             # 雷达最大探测距离（单位：米）
                 fai_width_deg=60,             # 水平方向主波束半锥角（度）
                 theta_height_deg=50,          # 垂直方向主波束半锥角（度）
                 doppler_threshold=100):        # 多普勒盲区门限
        self.max_range = max_range
        self.fai_width = np.radians(fai_width_deg)         # 转换为弧度
        self.theta_height = np.radians(theta_height_deg)
        self.doppler_threshold = doppler_threshold

    def detect(self, ego_pos, ego_vel, ego_yaw, ego_pitch,
                     tar_pos, tar_vel):
        # === Step 1: 相对位置和距离 ===
        rel_vec = tar_pos - ego_pos     # 目标相对于雷达的相对向量
        x, y, z = rel_vec
        dist = np.linalg.norm(rel_vec)  # 欧几里得距离
        is_dis_detect = dist <= self.max_range  # 是否在雷达最大探测范围内

        # === Step 2: 水平方向（方位角）判断 ===
        ra_tar_ang_fw = np.arctan2(y, x)    # 目标在水平面上的相对角度（雷达正前为0）
        ra_ang_fw = ego_yaw                 # 雷达当前朝向（yaw）
        fw_cha = abs(ra_tar_ang_fw - ra_ang_fw)  # 方位角偏差
        if fw_cha > np.pi:                       # 修正成最小角差（考虑π到−π的环绕）
            fw_cha = 2 * np.pi - fw_cha
        is_fw_detect = fw_cha <= self.fai_width  # 是否在主波束水平角范围内

        # === Step 3: 垂直方向（仰角）判断 ===
        horizontal_dist = np.hypot(x, y)              # 水平距离 √(x² + y²)
        ra_tar_ang_fy = np.arctan2(z, horizontal_dist)  # 目标仰角
        ra_ang_fy = ego_pitch                         # 雷达当前俯仰角
        is_fy_detect = (ra_tar_ang_fy >= ra_ang_fy - self.theta_height and
                        ra_tar_ang_fy <= ra_ang_fy + self.theta_height)  # 是否在俯仰角波束范围内

        # === Step 4: 多普勒盲区判断 ===
        los = rel_vec / (dist + 1e-6)           # 单位向量：雷达指向目标的视线（Line of Sight）
        Vr = np.dot(tar_vel - ego_vel, los)    # 相对径向速度（目标速度减雷达速度，投影到LOS方向）
        is_in_doppler_blind = abs(Vr) < self.doppler_threshold  # Vr过小，落入盲区

        # === Step 5: 最终判断 ===
        is_beam_detect = is_dis_detect and is_fw_detect and is_fy_detect     # 在主波束内
        is_locked = is_beam_detect and (not is_in_doppler_blind)             # 成功锁定的条件 = 主波束内 + 不在盲区

        # === 返回锁定状态 + 详细信息 ===
        return is_locked, {
            "is_locked": is_locked,                      #  是否被锁定
            "in_beam": is_beam_detect,                   #  是否在波束范围内
            "in_doppler_blind": is_in_doppler_blind,     #  是否在 Doppler 盲区
            "distance": dist,                            #  距离（米）
            "Vr": Vr,                                    #  相对径向速度（单位：m/s）
            "az_angle": ra_tar_ang_fw,                   #  水平角：目标方向角（rad）
            "az_diff": fw_cha,                           #  方位角误差：目标与雷达航向差
            "el_angle": ra_tar_ang_fy,                   #  仰角：目标在雷达视线方向的角度（rad）
            "el_center": ra_ang_fy,                      #  雷达自身当前仰角中心（rad）
        }
