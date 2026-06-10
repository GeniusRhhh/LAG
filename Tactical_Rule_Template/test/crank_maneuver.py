import numpy as np

class PIDController:
    """简单PID控制器"""
    def __init__(self, Kp, Ki, Kd, output_limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.last_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        self.last_error = error
        return output


class CrankManeuver:
    """
    Crank机动动作模板（基于离散舵面控制）：
    控制飞机滚转到一定角度后，向一侧偏航达到目标Crank角度，并保持一定时间。
    """
    def __init__(self, crank_angle_deg=30, crank_duration=5.0, target_bank_angle_deg=15,
                 throttle_index=20, control_dt=0.05):
        """
        初始化Crank机动动作参数
        :param crank_angle_deg: 目标Crank角度（航向变化量），单位：度
        :param crank_duration: Crank动作持续时间，单位：秒
        :param target_bank_angle_deg: 期望滚转角度（帮助偏航），单位：度
        :param throttle_index: 目标油门离散档位
        :param control_dt: 仿真控制步长，单位：秒
        """
        self.crank_angle_rad = np.radians(crank_angle_deg)  # 目标Crank角弧度值
        self.crank_duration = crank_duration  # Crank动作持续总时间
        self.target_bank_angle_rad = np.radians(target_bank_angle_deg)  # 期望滚转角弧度值
        self.throttle_index = throttle_index  # 固定油门档位
        self.control_dt = control_dt  # 仿真步长

        self.elapsed_time = 0.0  # 已经执行的时间累计
        self.active = True  # Crank动作是否还在进行中
        self.initial_heading = None  # 动作起始航向，用于计算目标航向

        # 初始化两个独立的PID控制器
        self.roll_pid = PIDController(Kp=10.0, Ki=0.0, Kd=2.0, output_limit=20.0)  # 控制副翼档位变化
        self.yaw_pid = PIDController(Kp=10.0, Ki=0.0, Kd=2.0, output_limit=20.0)   # 控制方向舵档位变化

    def generate_control(self, current_state):
        """
        输入当前飞机状态，输出本时刻应执行的离散舵面动作。
        :param current_state: dict，包含至少 'heading'（航向），'roll'（滚转角）
        :return: np.array([aileron_index, elevator_index, rudder_index, throttle_index])
        """
        if not self.active:
            return None

        # 更新已执行时间
        self.elapsed_time += self.control_dt

        # 如果持续时间已到，标记动作结束
        if self.elapsed_time >= self.crank_duration:
            self.active = False
            return None

        # 提取当前飞机状态
        current_heading = current_state.get('heading', 0.0)  # 当前航向（弧度）
        current_roll = current_state.get('roll', 0.0)         # 当前滚转角（弧度）

        # 第一次调用时记录初始航向
        if self.initial_heading is None:
            self.initial_heading = current_heading

        # 计算目标航向 = 初始航向 + Crank角
        target_heading = self.initial_heading + self.crank_angle_rad

        # 计算航向误差（自动处理±π跨越问题）
        heading_error = np.arctan2(np.sin(target_heading - current_heading),
                                   np.cos(target_heading - current_heading))

        # 计算滚转角误差
        bank_error = self.target_bank_angle_rad - current_roll

        # 由PID控制器输出档位调整量
        delta_aileron = self.roll_pid.control(bank_error, self.control_dt)
        delta_rudder = self.yaw_pid.control(heading_error, self.control_dt)

        # 将PID输出映射到离散档位（副翼、方向舵）
        # 中性位置是20，偏移后夹到合法范围[0,40]
        aileron_index = int(np.clip(20 + delta_aileron, 0, 40))
        elevator_index = 20  # Crank期间保持中性升降舵
        rudder_index = int(np.clip(20 + delta_rudder, 0, 40))
        throttle_index = self.throttle_index  # 固定油门档位

        return np.array([aileron_index, elevator_index, rudder_index, throttle_index])