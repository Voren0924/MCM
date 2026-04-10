import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# ============================================================
# 模型参数（将所有常数集中放在一个地方）
# ============================================================

@dataclass
class ModelParams:
    # 能量池几何/尺度参数
    V: float            # [体积单位] 能量池的等效体积

    # 通道几何参数
    S: float            # [面积] 能量传输通道的截面积

    # 通道耗散强度
    mu: float           # [1/(速度)] 等效摩擦系数，功率损耗模型：P_loss = mu * rho * S * v^2

    # 热学参数
    C_th: float         # [J/K] 等效热容
    h: float            # [W/K] 与环境之间的换热系数
    T_a: float          # [摄氏度] 环境温度

    # 低温载流能力参数（alpha(T)，方案 A）
    alpha_min: float    # 当 T → -∞ 时 alpha(T) 的下界
    beta: float         # [摄氏度] 低温敏感尺度（指数衰减系数）

    # alpha(T)=1 的参考温度
    T_ref: float = 25.0 # [摄氏度] 在该温度及以上，alpha 取 1


# ============================================================
# alpha(T)：低温载流能力系数（方案 A）
# ============================================================

def alpha_of_T(T: float, p: ModelParams) -> float:
    """
    低温载流能力系数 alpha(T)

    方案 A：
      - 当 T >= T_ref 时，alpha(T) = 1
      - 当 T < T_ref 时：
        alpha(T) = alpha_min + (1 - alpha_min) * exp(-(T_ref - T) / beta)
    """
    if T >= p.T_ref:
        return 1.0

    # 防止 beta 为 0（在参数合法的情况下不应发生）
    if p.beta <= 0:
        raise ValueError("beta 必须大于 0")

    return p.alpha_min + (1.0 - p.alpha_min) * math.exp(-(p.T_ref - T) / p.beta)


# ============================================================
# 单步求解器：在给定 E_n、T_n、P(t_n) 的情况下，
# 先解代数变量 v_n，再用显式 Euler 推进 E 和 T
# ============================================================

@dataclass
class StepResult:
    # 更新后的状态
    E_next: float
    T_next: float

    # 时刻 t_n 的诊断量
    rho: float
    alpha: float
    v: float
    P_out: float
    P_loss: float
    P_use: float
    discriminant: float

    # 事件标志
    feasible: bool
    cutoff_reason: Optional[str] = None


def step_euler_dae(
    E_n: float,
    T_n: float,
    t_n: float,
    dt: float,
    P_func: Callable[[float, float, float], float],
    p: ModelParams,
    *,
    choose_small_root: bool = True,
    clamp_E_nonnegative: bool = True,
) -> StepResult:
    """
    执行 DAE 系统的一个时间步

    输入
    ----
    E_n, T_n : 当前时刻 t_n 的状态
    t_n      : 当前时间
    dt       : 时间步长
    P_func   : 外部负载功率需求模型，返回 P(t) [W]
               函数签名：P_func(t, E, T) -> P
               这里允许 P_func 使用 E、T，便于扩展
    p        : 模型参数
    choose_small_root : 若为 True，则选择二次方程的较小根
                        （更节能、发热更小的运行点）
    clamp_E_nonnegative : 若为 True，则将 E_next 限制为非负

    返回
    ----
    StepResult，包含更新后的状态与诊断信息
    """

    # ------------------------------------------------------------
    # 1) 计算当前状态下的 rho 和 alpha
    # ------------------------------------------------------------
    if p.V <= 0:
        raise ValueError("V 必须大于 0")
    if p.S <= 0:
        raise ValueError("S 必须大于 0")
    if p.mu <= 0:
        raise ValueError("mu 必须大于 0")
    if p.C_th <= 0:
        raise ValueError("C_th 必须大于 0")
    if p.h < 0:
        raise ValueError("h 必须大于等于 0")

    rho_n = E_n / p.V
    alpha_n = alpha_of_T(T_n, p)

    # 如果 rho <= 0，说明能量已经耗尽，无法继续供能
    if rho_n <= 0:
        return StepResult(
            E_next=max(E_n, 0.0) if clamp_E_nonnegative else E_n,
            T_next=T_n,
            rho=rho_n,
            alpha=alpha_n,
            v=0.0,
            P_out=0.0,
            P_loss=0.0,
            P_use=0.0,
            discriminant=float("nan"),
            feasible=False,
            cutoff_reason="能量耗尽（rho <= 0）",
        )

    # ------------------------------------------------------------
    # 2) 计算时刻 t_n 的外部功率需求
    # ------------------------------------------------------------
    P_dem = float(P_func(t_n, E_n, T_n))

    # 安全处理：需求功率不允许为负
    if P_dem < 0:
        P_dem = 0.0

    # 特殊情况：无功率需求，仅进行散热
    if P_dem == 0.0:
        P_out = 0.0
        P_loss = 0.0
        P_use = 0.0

        # 热方程的 Euler 更新
        dTdt = (P_loss - p.h * (T_n - p.T_a)) / p.C_th
        T_next = T_n + dTdt * dt

        # 无输出功率时能量不减少
        E_next = E_n

        return StepResult(
            E_next=E_next,
            T_next=T_next,
            rho=rho_n,
            alpha=alpha_n,
            v=0.0,
            P_out=P_out,
            P_loss=P_loss,
            P_use=P_use,
            discriminant=alpha_n**2,
            feasible=True,
            cutoff_reason=None,
        )

    # ------------------------------------------------------------
    # 3) 求解代数约束以得到 v_n：
    #
    #   alpha * rho * v * S - mu * rho * S * v^2 = P_dem
    #
    # 两边同时除以 (rho * S) > 0：
    #
    #   alpha * v - mu * v^2 = P_dem / (rho * S)
    #
    # 整理为二次方程：
    #
    #   mu * v^2 - alpha * v + P_dem / (rho * S) = 0
    # ------------------------------------------------------------
    rhs = P_dem / (rho_n * p.S)

    a = p.mu
    b = -alpha_n
    c = rhs

    # 判别式：alpha^2 - 4 * mu * rhs
    disc = alpha_n * alpha_n - 4.0 * p.mu * rhs

    # ------------------------------------------------------------
    # 4) 可行性 / 截断判断
    # ------------------------------------------------------------
    if disc < 0:
        # 无实根，说明在当前 (E, T) 下无法满足功率需求
        return StepResult(
            E_next=E_n,
            T_next=T_n,
            rho=rho_n,
            alpha=alpha_n,
            v=0.0,
            P_out=0.0,
            P_loss=0.0,
            P_use=0.0,
            discriminant=disc,
            feasible=False,
            cutoff_reason="能量侧截断：判别式 < 0（无可行 v）",
        )

    sqrt_disc = math.sqrt(disc)

    # 两个解
    v_plus = (alpha_n + sqrt_disc) / (2.0 * p.mu)
    v_minus = (alpha_n - sqrt_disc) / (2.0 * p.mu)

    # ------------------------------------------------------------
    # 5) 选择运行点（根的选择）
    #
    # 默认选择较小的根，以降低功率和发热
    # ------------------------------------------------------------
    if choose_small_root:
        v_n = v_minus
    else:
        v_n = v_plus

    # 数值安全：极小的负值视为 0
    if v_n < 0 and abs(v_n) < 1e-12:
        v_n = 0.0

    if v_n < 0:
        return StepResult(
            E_next=E_n,
            T_next=T_n,
            rho=rho_n,
            alpha=alpha_n,
            v=v_n,
            P_out=0.0,
            P_loss=0.0,
            P_use=0.0,
            discriminant=disc,
            feasible=False,
            cutoff_reason="数值异常：选取的 v 为负值",
        )

    # ------------------------------------------------------------
    # 6) 计算功率分解
    # ------------------------------------------------------------
    P_out = alpha_n * rho_n * v_n * p.S
    P_loss = p.mu * rho_n * p.S * (v_n ** 2)
    P_use = P_out - P_loss  # 理论上应等于 P_dem

    # ------------------------------------------------------------
    # 7) 用 Euler 法更新 E 和 T
    #
    # 能量方程：dE/dt = -P_out
    # 热方程：C_th dT/dt = P_loss - h(T - T_a)
    # ------------------------------------------------------------
    dEdt = -P_out
    dTdt = (P_loss - p.h * (T_n - p.T_a)) / p.C_th

    E_next = E_n + dEdt * dt
    T_next = T_n + dTdt * dt

    if clamp_E_nonnegative and E_next < 0:
        E_next = 0.0

    return StepResult(
        E_next=E_next,
        T_next=T_next,
        rho=rho_n,
        alpha=alpha_n,
        v=v_n,
        P_out=P_out,
        P_loss=P_loss,
        P_use=P_use,
        discriminant=disc,
        feasible=True,
        cutoff_reason=None,
    )


# ============================================================
# 完整仿真驱动函数
# ============================================================

@dataclass
class SimResult:
    t: List[float]
    E: List[float]
    T: List[float]
    rho: List[float]
    alpha: List[float]
    v: List[float]
    P_dem: List[float]
    P_out: List[float]
    P_loss: List[float]
    P_use: List[float]
    disc: List[float]
    cutoff_index: Optional[int]
    cutoff_reason: Optional[str]




def simulate(
    E0: float,
    T0: float,
    t_end: float,
    dt: float,
    P_func: Callable[[float, float, float], float],
    p: ModelParams,
) -> SimResult:
    """
    从 t=0 仿真到 t_end，时间步长为 dt。
    如果发生能量侧截断（判别式 disc < 0）或能量耗尽，则提前终止仿真。
    """

    if dt <= 0:
        raise ValueError("dt 必须大于 0")
    if t_end <= 0:
        raise ValueError("t_end 必须大于 0")

    n_steps = int(math.ceil(t_end / dt))

    # 用于存储仿真结果的数组
    t_list: List[float] = []
    E_list: List[float] = []
    T_list: List[float] = []
    rho_list: List[float] = []
    alpha_list: List[float] = []
    v_list: List[float] = []
    P_dem_list: List[float] = []
    P_out_list: List[float] = []
    P_loss_list: List[float] = []
    P_use_list: List[float] = []
    disc_list: List[float] = []

    cutoff_index: Optional[int] = None
    cutoff_reason: Optional[str] = None

    E_n, T_n = float(E0), float(T0)

    for n in range(n_steps + 1):
        t_n = n * dt

        # 单独记录当前时刻的功率需求（用于绘图或诊断）
        P_dem = float(P_func(t_n, E_n, T_n))
        if P_dem < 0:
            P_dem = 0.0

        # 执行一步计算（内部会求解 v、功率分解以及下一步状态）
        sr = step_euler_dae(E_n, T_n, t_n, dt, P_func, p)

        # 保存当前时刻（更新前）的状态与诊断量
        t_list.append(t_n)
        E_list.append(E_n)
        T_list.append(T_n)
        rho_list.append(sr.rho)
        alpha_list.append(sr.alpha)
        v_list.append(sr.v)
        P_dem_list.append(P_dem)
        P_out_list.append(sr.P_out)
        P_loss_list.append(sr.P_loss)
        P_use_list.append(sr.P_use)
        disc_list.append(sr.discriminant)

        # 若当前状态不可行，则提前终止
        if not sr.feasible:
            cutoff_index = n
            cutoff_reason = sr.cutoff_reason
            break

        # 推进到下一时刻
        E_n, T_n = sr.E_next, sr.T_next

        # 若更新后能量耗尽，则终止
        if E_n <= 0:
            cutoff_index = n + 1
            cutoff_reason = "更新后能量耗尽（E <= 0）"
            break

        if t_n >= t_end:
            break

    return SimResult(
        t=t_list,
        E=E_list,
        T=T_list,
        rho=rho_list,
        alpha=alpha_list,
        v=v_list,
        P_dem=P_dem_list,
        P_out=P_out_list,
        P_loss=P_loss_list,
        P_use=P_use_list,
        disc=disc_list,
        cutoff_index=cutoff_index,
        cutoff_reason=cutoff_reason,
    )


# ============================================================
# 示例用法（将 P_func 替换为你自己的设备使用模型）
# ============================================================

def example_P_func(t: float, E: float, T: float) -> float:
    """
    示例功率需求模型。
    在实际使用中，请用你自己的“设备使用模型”输出替换本函数。

    此处采用分段常值功率：
      - 0~200 秒：2 W
      - 200~600 秒：6 W
      - 600~1000 秒：3 W
      - 其余时间：1 W
    """
    if t < 200:
        return 20
    elif t < 600:
        return 39
    elif t < 1000:
        return 1234
    else:
        return 232


if __name__ == "__main__":
    # 设置模型参数（以下数值仅为占位符，需通过数据进行标定）
    params = ModelParams(
        V=1.0,           # 等效体积
        S=1.0,           # 通道截面积
        mu=0.2,          # 摩擦/耗散系数
        C_th=200.0,      # 等效热容
        h=1.0,           # 换热系数
        T_a=25.0,        # 环境温度
        alpha_min=0.2,   # 极低温下的最小 alpha
        beta=10.0,       # 低温敏感尺度
    )

    # 初始条件
    E0 = 10000.0  # 初始总能量（单位与 W·s = J 一致）
    T0 = 100.0    # 初始温度（摄氏度）

    # 仿真设置
    dt = 0.5
    t_end = 2000.0

    result = simulate(E0, T0, t_end, dt, example_P_func, params)

    # 输出简要仿真结果
    print(f"仿真步数：{len(result.t)}")
    if result.cutoff_index is not None:
        print(f"在第 {result.cutoff_index} 步发生截断，对应时间约 {result.t[result.cutoff_index]:.3f} 秒")
        print(f"截断原因：{result.cutoff_reason}")
        print(f"截断时能量 E：{result.E[result.cutoff_index]:.6g}")
        print(f"截断时温度 T：{result.T[result.cutoff_index]:.6g}")
        print(f"截断时 alpha：{result.alpha[result.cutoff_index]:.6g}")
        print(f"截断时判别式 disc：{result.disc[result.cutoff_index]:.6g}")
    else:
        print("在仿真时间范围内未发生截断。")

    plt.figure()
    plt.plot(result.t, result.E)
    plt.xlabel("time t (s)")
    plt.ylabel("energy E(t) (J)")
    plt.title("change")
    plt.grid(True)
    plt.tight_layout()

    # 可选：标记截断时刻
    if result.cutoff_index is not None:
        t_cut = result.t[result.cutoff_index]
        plt.axvline(t_cut, linestyle="--")

    plt.show()

    plt.figure()
    plt.plot(result.t, result.E)
    plt.xlabel("time t (s)")
    plt.ylabel("energy E(t) (J)")
    plt.title("change")
    plt.grid(True)
    plt.tight_layout()
    # 1. 获取初始时刻的能量和输出功率
    E_start = result.E[0]
    t_start = result.t[0]
    # 注意：P_out_list 存储的是每个时刻解出来的实际输出功率
    p_out_start = result.P_out[0]

    # 2. 计算切线。为了美观，我们让切线延伸一段距离（例如延伸到仿真结束或直到碰到 X 轴）
    # 切线方程：E_tangent = E_start - p_out_start * (t - t_start)
    t_tangent = result.t
    e_tangent = [E_start - p_out_start * (ti - t_start) for ti in t_tangent]

    # 3. 绘制切线（通常用不同的颜色或虚线，并限制显示范围防止拉得太长）
    plt.plot(t_tangent, e_tangent, 'g:', label='Initial Tangent', alpha=0.8)

    # 限制 y 轴范围，防止切线画到负数区域太远影响观感
    plt.ylim(bottom=min(result.E) * 0.9)

    plt.legend()

    # 可选：标记截断时刻
    if result.cutoff_index is not None:
        t_cut = result.t[result.cutoff_index]
        plt.axvline(t_cut, linestyle="--")

    plt.show()

    # ============================================================
    # 新增：等效电压 U = P_out / v 的曲线图
    # ============================================================

    # 计算 U = P_out / v (注意处理 v=0 的情况)
    # 根据公式 P_out = alpha * rho * v * S， U 实际上就是 alpha * rho * S
    u_list = [
        (res_p_out / res_v) if res_v > 1e-9 else (res_alpha * res_rho * params.S)
        for res_p_out, res_v, res_alpha, res_rho in zip(result.P_out, result.v, result.alpha, result.rho)
    ]

    plt.figure(figsize=(10, 6))

    # 创建双轴：左轴画 U，右轴画 v (对比看负载变化)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    line1, = ax1.plot(result.t, u_list, color='blue', label='Equivalent Voltage $U$ ($P_{out}/v$)')
    line2, = ax2.plot(result.t, result.v, color='red', linestyle='--', alpha=0.6, label='Velocity $v$')

    ax1.set_xlabel("Time $t$ (s)")
    ax1.set_ylabel("Voltage $U$ (Energy/Distance)")
    ax2.set_ylabel("Velocity $v$ (m/s)")

    plt.title("Equivalent Voltage $U$ and Velocity $v$ Over Time")

    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    if result.cutoff_index is not None:
        ax1.axvline(result.t[result.cutoff_index], color='orange', linestyle=':', label='Cutoff')

    plt.tight_layout()
    plt.show()
    # 如有需要，可在此添加更多绘图（matplotlib），
    # 但在正式报告中通常导出数据后再统一绘制。
