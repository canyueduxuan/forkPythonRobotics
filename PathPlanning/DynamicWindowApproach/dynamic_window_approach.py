"""动态窗口法（Dynamic Window Approach，DWA）局部路径规划示例。

这个文件解决的问题可以用一句话概括：

    机器人下一小步应该采用多大的前进速度 ``v`` 和转向角速度 ``omega``？

DWA 并不一次规划出从起点到终点的完整路线。它会在每一个控制周期中重复：

1. 根据机器人的速度、加速度限制，计算“下一时刻实际能达到”的速度范围；
2. 从这个范围中采样许多组 ``[v, omega]``；
3. 假设每组速度在未来几秒内保持不变，预测对应的运动轨迹；
4. 分别评价轨迹是否朝向目标、速度是否足够快、是否会撞障碍物；
5. 选总代价最小的一组速度，只执行一个很短的控制周期；
6. 获得新状态后重新规划，因此能够随机器人运动不断修正决策。

重要变量约定：

``x = [px, py, yaw, v, omega]``
    机器人状态：二维位置、朝向角、线速度、角速度。
``u = [v, omega]``
    控制输入：线速度和角速度。
``goal = [gx, gy]``
    目标点坐标。
``ob = [[ox1, oy1], ...]``
    障碍物点坐标集合。

角度和角速度统一使用弧度（rad），距离统一使用米（m），时间使用秒（s）。

原作者：Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

# 是否显示 Matplotlib 动画。测试或无图形界面运行时可改为 False。
show_animation = True


def dwa_control(x, config, goal, ob):
    """执行一次 DWA 决策，返回本周期控制量和对应的预测轨迹。

    注意：这里只“做决策”，并不会直接修改真实机器人状态。调用者随后
    使用 :func:`motion` 执行一个 ``dt``，再用新状态开始下一轮决策。

    ``x`` 是 ``[px, py, yaw, v, omega]``；``goal`` 是 ``[gx, gy]``；
    ``ob`` 是 N 行 2 列的障碍物坐标。返回的 ``u`` 是 ``[v, omega]``。
    """
    # 第一步：只保留“既不超机械极限、下一周期又确实能达到”的速度范围。
    dw = calc_dynamic_window(x, config)

    # 第二步：遍历动态窗口中的候选速度，预测轨迹、打分并选出最优解。
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


class RobotType(Enum):
    """碰撞检测所使用的机器人外形。"""

    circle = 0
    rectangle = 1


class Config:
    """集中保存机器人约束、DWA 参数、外形尺寸和示例障碍物。"""

    def __init__(self):
        # ------------------------- 机器人运动能力 -------------------------
        # 线速度的全局上下限。负值表示机器人允许倒车。
        self.max_speed = 1.0  # [m/s] 最大前进速度
        self.min_speed = -0.5  # [m/s] 最大倒车速度
        # 最大角速度为 40 度/秒；程序内部把角度统一换算为弧度。
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        # 每秒最多能让线速度改变多少，即 |dv/dt| 的上限。
        self.max_accel = 0.2  # [m/s^2]
        # 每秒最多能让角速度改变多少，即 |d(omega)/dt| 的上限。
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]

        # --------------------------- 采样精度 -----------------------------
        # 步长越小，候选控制量越多、搜索越细，但计算也越慢。
        self.v_resolution = 0.01  # [m/s] 线速度采样间隔
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]

        # --------------------------- 预测参数 -----------------------------
        # 每隔 dt 秒根据运动模型推演一次；控制主循环也只执行这么长时间。
        self.dt = 0.1  # [s]
        # 对每个候选速度向未来预测 3 秒。太短会“目光短浅”，太长更耗时。
        self.predict_time = 3.0  # [s]

        # --------------------------- 代价权重 -----------------------------
        # 最终代价 = 朝向目标代价 + 速度代价 + 障碍物代价。
        # 增大某项权重，就等于告诉机器人“这一项更重要”。
        self.to_goal_cost_gain = 0.15      # 朝向目标的重要程度
        self.speed_cost_gain = 1.0         # 快速行驶的重要程度
        self.obstacle_cost_gain = 1.0      # 远离障碍物的重要程度
        # 当机器人和候选速度都几乎为 0 时，强制转动以避免原地不动。
        self.robot_stuck_flag_cons = 0.001
        self.robot_type = RobotType.circle

        # 圆形机器人用半径检查碰撞；两种外形都用它作为“到达目标”的阈值。
        self.robot_radius = 1.0  # [m]

        # 矩形机器人碰撞检测所用的宽和长。
        self.robot_width = 0.5  # [m]
        self.robot_length = 1.2  # [m]

        # 示例地图中的障碍物，每一行都是一个点 [x, y]，单位为米。
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])

    @property
    def robot_type(self):
        """返回当前机器人外形。"""
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        """设置机器人外形，并尽早拦截拼写错误或错误类型。"""
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


# 示例程序共用的配置对象。实际项目也可以为每台机器人分别创建 Config。
config = Config()


def motion(x, u, dt):
    """用简化的单轨/独轮车运动模型，把状态向前推进 ``dt`` 秒。

    控制量 ``u = [v, omega]`` 在这一个小时间段内视为常数。先更新朝向，
    再沿新朝向移动。这里会原地修改传入的 ``x``，并返回同一个数组。

    公式为：``yaw += omega*dt``，``px += v*cos(yaw)*dt``，
    ``py += v*sin(yaw)*dt``。
    """
    # omega * dt 是本时间片内转过的角度。
    x[2] += u[1] * dt
    # 把前进距离 v * dt 分解到世界坐标系的 x、y 两个方向。
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    # 状态的最后两项记录刚执行的线速度和角速度，供下一轮动态窗口使用。
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """计算本控制周期允许搜索的速度窗口 ``[v_min,v_max,w_min,w_max]``。

    “动态”的关键在这里：搜索范围不只是机器人铭牌上的最大速度，还会根据
    当前速度和加速度限制随时变化。例如当前静止、``dt=0.1``、最大加速度
    ``0.2 m/s²``，那么下一周期最多只能达到 ``0.02 m/s``，不能瞬间跳到
    ``1.0 m/s``。
    """
    # Vs：机器人结构/电机允许的绝对速度范围，任何时候都不能越过。
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Vd：从当前速度出发，考虑一个 dt 内最大加速度后“够得着”的范围。
    # x[3] 是当前线速度 v，x[4] 是当前角速度 omega。
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    # 求 Vs 与 Vd 的交集：下限取较大者，上限取较小者。
    # 得到 [线速度下限, 线速度上限, 角速度下限, 角速度上限]。
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """预测固定控制量 ``[v, y]`` 在未来会形成的轨迹。

    参数名 ``y`` 在原实现中表示角速度 omega，并不是 y 坐标。预测期间假设
    线速度和角速度保持不变。返回数组的每一行都是
    ``[px, py, yaw, v, omega]``，首行是尚未运动的初始状态。
    """
    # 必须复制状态，否则试算一条候选轨迹就会污染真实状态和其他候选轨迹。
    x = np.array(x_init)
    # 先保存 t=0 的状态；后续每走一个 dt 就在底部追加一行。
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """穷举动态窗口中的控制量，返回总代价最小的控制量及其轨迹。

    这是 DWA 的核心搜索器。可以把它想成让许多“虚拟机器人”同时试跑：
    每个虚拟机器人拿一组不同的 ``[v, omega]`` 向未来跑几秒，然后裁判用
    同一套规则打分。碰撞轨迹为无穷大分，其余轨迹中分数最低者胜出。
    """
    # 保存本轮共同的起点。x[:] 是浅复制；这里 x 是一维数值数组，已足够。
    x_init = x[:]
    # 任何有限代价都小于正无穷，因此遇到第一个合法候选时一定会更新。
    min_cost = float("inf")
    # 如果没有更好结果时使用的安全初值：停止前进、停止旋转。
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # 外层枚举线速度 v；np.arange 的终点通常不包含 dw[1]。
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        # 内层枚举角速度 y（即 omega）。两层循环组成速度空间中的采样网格。
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            # 让当前候选控制量从相同的 x_init 出发，模拟 predict_time 秒。
            trajectory = predict_trajectory(x_init, v, y, config)

            # 三项代价都遵循“越小越好”，各自乘权重后再相加：
            # 1) 末端朝向与目标方向的夹角，越对准目标越小；
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            # 2) 距离最高速度的差，速度越快越小；这里鼓励向前行驶；
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            # 3) 与最近障碍物距离的倒数，离障碍越远越小；碰撞则为 inf。
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            # 加权和就是这个候选轨迹的最终成绩。
            final_cost = to_goal_cost + speed_cost + ob_cost

            # <= 表示分数相同时保留后遍历到的候选解。
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory

                # 特殊“脱困”规则：若机器人原本静止，最优解又要求继续静止，
                # 就给它一个负方向角速度，让机器人先转起来，再在下一轮重算。
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate

    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """计算轨迹的避障代价；发生碰撞返回正无穷，否则返回最近距离的倒数。

    ``r[i, j]`` 表示第 j 个障碍物到轨迹第 i 个位置的欧氏距离。NumPy 的
    广播一次算完“所有轨迹点 × 所有障碍物”的距离，避免 Python 双重循环。
    圆形和矩形机器人使用不同的碰撞判定，但非碰撞时都用点间最近距离打分。
    """
    # 分别取出所有障碍物的 x 坐标与 y 坐标。
    ox = ob[:, 0]
    oy = ob[:, 1]
    # trajectory[:, 0] 形状为 (T,)，ox[:, None] 形状为 (N, 1)。
    # 广播后 dx、dy 的形状为 (N, T)：每个障碍物到每个预测位置的坐标差。
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    # hypot(dx, dy) 等价于 sqrt(dx**2 + dy**2)，但数值上更稳健。
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        # 矩形碰撞检测不能只看中心距离，因为机器人朝向会改变占用区域。
        yaw = trajectory[:, 2]
        # 为轨迹中的每个 yaw 构造一个 2×2 旋转矩阵。
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        # 先把障碍物位置平移成“相对机器人中心”的坐标。
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        # 再旋转到机器人的局部坐标系：机器人在这个坐标系里始终摆正。
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        # 点同时落在前、后、左、右四条边以内，就位于矩形机器人内部。
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            # 无穷大保证碰撞轨迹不会赢过任何有限代价的安全轨迹。
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        # 对圆形机器人，任意中心距离 <= 半径就表示障碍点进入机器人内部。
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    # 没碰撞时，只关心整条预测轨迹上离障碍物最近的那一次。
    min_r = np.min(r)
    # 用倒数把“距离越大越好”转换成代价函数要求的“数值越小越好”。
    return 1.0 / min_r


def calc_to_goal_cost(trajectory, goal):
    """返回预测轨迹末端朝向与“指向目标方向”之间的最小夹角。

    返回值范围是 ``[0, pi]``：0 表示正对目标，pi 表示完全背对目标。
    这里只评价方向，不直接评价末端到目标的直线距离。
    """
    # 从预测轨迹的最后一个位置指向目标的向量。
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    # atan2 同时利用 dx、dy，能正确区分目标所在象限，结果在 [-pi, pi]。
    error_angle = math.atan2(dy, dx)
    # 目标方向减去预测末端朝向，得到尚需转过的角度。
    cost_angle = error_angle - trajectory[-1, 2]
    # atan2(sin(a), cos(a)) 把任意角度归一化到 [-pi, pi]，避免 2*pi
    # 周期造成的假象。例如 179° 与 -179° 实际只相差 2°，不是 358°。
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    """在机器人中心画一支箭头，直观显示当前朝向。"""
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    """按照配置的圆形或矩形外形，在 Matplotlib 中画出机器人。"""
    if config.robot_type == RobotType.rectangle:
        # 先在机器人局部坐标系中给出矩形闭合轮廓的 5 个顶点。
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        # 把局部轮廓旋转 yaw，再平移到世界坐标 (x, y)。
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        # 圆形本身没有朝向，所以额外画一条从圆心到圆周的黑线表示 yaw。
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    """运行完整示例：反复规划一个 dt，直到机器人进入目标半径。

    绿色线是“当前轮选中的未来预测轨迹”，红色线是机器人真正走过的历史
    轨迹。两者不同是正常的，因为 DWA 每走 0.1 秒就会重新规划。
    """
    print(__file__ + " start!!")
    # 初始状态：[x位置(m), y位置(m), 朝向(rad), 线速度(m/s), 角速度(rad/s)]。
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # 目标只有位置，不要求机器人到达时保持某个特定朝向。
    goal = np.array([gx, gy])

    # 选择碰撞检测外形，并读取配置中的示例障碍物。
    config.robot_type = robot_type
    # trajectory 保存真实状态历史；初始只有一行。
    trajectory = np.array(x)
    ob = config.ob

    # “感知/规划/执行”闭环：每轮只执行一个 dt，然后立刻重新规划。
    while True:
        # 规划：根据当前状态选出控制量，并拿到用于展示的最佳预测轨迹。
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        # 执行：示例用运动模型模拟真实机器人只走 0.1 秒。
        x = motion(x, u, config.dt)
        # 记录实际走到的新状态，最后用于绘制红色完整路径。
        trajectory = np.vstack((trajectory, x))

        if show_animation:
            # 清除上一帧，重新绘制当前状态。
            plt.cla()
            # 按 Esc 可提前停止仿真。
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            # 绿线：本轮最佳预测；红叉：机器人；蓝叉：目标；黑点：障碍物。
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # 用欧氏距离判断是否到达；进入 robot_radius 范围就算成功。
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    # 直接运行本文件时默认演示矩形机器人；可切换为下面的圆形版本。
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)
