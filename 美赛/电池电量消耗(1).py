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

    # 新增：用于 rho 的能量衰减系数
    E_max: float  # [J] 满电能量（用于归一化）
    P_min: float  # [W] 最小可接受输出功率（用于定义 rho_min）
    gamma: float  # [-] 非线性指数（>=0，越大低电量越“掉能力”）

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

def alpha_of_T(T: float) -> float:
    """
    低温载流能力系数 alpha(T)
    """
    if T >= 10:
        return 1.0
    elif T >= -30.0:
        return max(0.1, math.log10((9/40) * (T + 30) + 1))
    else:
        return 0.1


# ============================================================
# 单步求解器：在给定 E_n、T_n、P(t_n) 的情况下，
# 先解代数变量 v_n，再用显式 Euler 推进 E 和 T
# ============================================================

@dataclass
class StepResult:
    # 更新后的状态
    E_next: float
    T_next: float

    # 池内/回路能力
    rho_raw: float      # E/V * (E/Emax)^gamma（如果你已经放进来了）
    rho_loop: float     # 回路等效能力（含温度）

    # 诊断量
    alpha: float
    v: float
    P_out: float
    P_loss: float
    P_use: float
    discriminant: float

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

    # rho = (E/V) * (E/E_max)^gamma
    if p.E_max <= 0:
        raise ValueError("E_max 必须大于 0")
    if p.gamma < 0:
        raise ValueError("gamma 必须大于等于 0")

    E_eff = max(E_n, 0.0)
    soc = min(max(E_eff / p.E_max, 0.0), 1.0)  # 归一化到 [0,1]
    rho_raw = (E_eff / p.V)
    alpha_n = alpha_of_T(T_n)
    rho_n = rho_raw * (soc ** p.gamma) * alpha_n

    # 如果 rho <= 0，说明能量已经耗尽，无法继续供能
    if rho_n <= 0:
        return StepResult(
            E_next=max(E_n, 0.0) if clamp_E_nonnegative else E_n,
            T_next=T_n,
            rho_raw=rho_raw,
            rho_loop=rho_n,
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
            rho_raw=rho_raw,
            rho_loop=rho_n,
            alpha=alpha_n,
            v=0.0,
            P_out=P_out,
            P_loss=P_loss,
            P_use=P_use,
            discriminant=1.0,
            feasible=True,
            cutoff_reason=None,
        )

    # ------------------------------------------------------------
    # 3) 求解代数约束以得到 v_n：
    #
    #   rho_loop * v * S - mu * rho_loop * S * v^2 = P_dem
    #
    # 两边同时除以 (rho_loop * S) > 0：
    #
    #   v - mu * v^2 = P_dem / (rho_loop * S)
    #
    # 整理为二次方程：
    #
    #   mu * v^2 - 1 * v + P_dem / (rho_loop * S) = 0
    # ------------------------------------------------------------
    rhs = P_dem / (rho_n * p.S)  # rho_n 在你新口径里就是 rho_loop

    a = p.mu
    b = -1.0
    c = rhs

    # 判别式：1 - 4 * mu * rhs
    disc = 1.0 - 4.0 * p.mu * rhs

    # ------------------------------------------------------------
    # 4) 可行性 / 截断判断
    # ------------------------------------------------------------
    if disc < 0:
        # 无实根，说明在当前 (E, T) 下无法满足功率需求
        return StepResult(
            E_next=E_n,
            T_next=T_n,
            rho_raw=rho_raw,
            rho_loop=rho_n,
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

    v_plus = (1.0 + sqrt_disc) / (2.0 * p.mu)
    v_minus = (1.0 - sqrt_disc) / (2.0 * p.mu)
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
            rho_raw=rho_raw,
            rho_loop=rho_n,
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
    P_out = rho_n * v_n * p.S
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
        rho_raw=rho_raw,
        rho_loop=rho_n,
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
    rho_raw: List[float]
    rho_loop: List[float]
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
    choose_small_root: bool = True,
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
    rho_raw_list: List[float] = []
    rho_loop_list: List[float] = []
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
        sr = step_euler_dae(E_n, T_n, t_n, dt, P_func, p, choose_small_root=choose_small_root)

        # 保存当前时刻（更新前）的状态与诊断量
        t_list.append(t_n)
        E_list.append(E_n)
        T_list.append(T_n)
        rho_raw_list.append(sr.rho_raw)
        rho_loop_list.append(sr.rho_loop)
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
            cutoff_index = n
            cutoff_reason = "更新后能量耗尽（E <= 0）"
            break

        if t_n >= t_end:
            break

    return SimResult(
        t=t_list,
        E=E_list,
        T=T_list,
        rho_raw=rho_raw_list,
        rho_loop=rho_loop_list,
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
    return 4.2


if __name__ == "__main__":
    # 设置模型参数（以下数值仅为占位符，需通过数据进行标定）
    params = ModelParams(
        V=500.0,
        S=1.0,
        E_max=50000.0,  # 你满电能量（通常和 E0 一致）
        P_min=4.2,
        gamma=0.3,
        mu=0.3,
        C_th=200.0,
        h=1.0,
        T_a=25.0,
        alpha_min=0.5,
        beta=100.0
    )

    # 初始条件
    E0 = params.E_max  # 初始总能量（单位与 W·s = J 一致）
    T0 = params.T_a    # 初始温度（摄氏度）

    # 仿真设置
    dt = 0.2
    t_end = 50000.0

    result = simulate(E0, T0, t_end, dt, example_P_func, params, choose_small_root=True)

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
    # ================== 后处理（不再重算物理） ==================

    # 回路等效能力（来自 step_euler_dae）
    rho_loop = result.rho_loop

    # 回路侧能力阈值（与判别式一致，alpha_ref = 1）
    rho_loop_max = rho_loop[0]
    # rho_loop_min = 4.0 * params.mu * params.P_min / params.S

    rho_loop_min = 4.0 * params.mu * params.P_min / params.S
    rho_loop_min_list = [rho_loop_min for _ in result.t]

    remaining_frac = [
        max(min((rho_l - rho_min) / (rho_loop[0] - rho_min), 1.0), 0.0)
        for rho_l, rho_min in zip(rho_loop, rho_loop_min_list)
    ]


    k = result.cutoff_index
    print("rho_loop(t):", rho_loop[k])
    print("rho_loop_min(t):", rho_loop_min_list[k])
    print("rho_loop(0):", rho_loop[0])
    print("rho_loop_min(0):", rho_loop_min_list[0])
    print("numerator:", rho_loop[k] - rho_loop_min_list[k])
    print("denominator:", rho_loop[0] - rho_loop_min_list[k])

    # ================== 能量随时间 ==================
    plt.figure()
    plt.plot(result.t, result.E)
    plt.xlabel("Time t (s)")
    plt.ylabel("Energy E(t) (J)")
    plt.title("Energy vs Time")
    plt.grid(True)

    if result.cutoff_index is not None:
        plt.axvline(result.t[result.cutoff_index], linestyle="--")

    plt.tight_layout()
    plt.show()

    # ================== 剩余可用电量随时间 ==================
    plt.figure()
    plt.plot(result.t, remaining_frac, linewidth=2)
    plt.xlabel("Time t (s)")
    plt.ylabel("Remaining Usable Energy")
    plt.title("State of Charge(Remaining Usable Energy)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ================== 温度随时间 ==================
    plt.figure()
    plt.plot(result.t, result.T, color="orange")
    plt.xlabel("Time t (s)")
    plt.ylabel("Temperature T(t) (°C)")
    plt.title("Temperature Evolution")
    plt.grid(True)

    if result.cutoff_index is not None:
        plt.axvline(result.t[result.cutoff_index], linestyle="--")

    plt.tight_layout()
    plt.show()

    # ================== 电压 & 速度 vs 剩余可用电量 ==================

    # 负载端电压（闭式）
    u_load = [
        rho_l * params.S - params.mu * rho_l * params.S * v
        for rho_l, v in zip(rho_loop, result.v)
    ]

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    line1, = ax1.plot(
        remaining_frac,
        u_load,
        linewidth=2,
        label="Load Voltage $U_{load}$"
    )

    line2, = ax2.plot(
        remaining_frac,
        result.v,
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Operating Velocity $v$"
    )

    ax1.set_xlabel("Remaining Usable Energy")
    ax1.set_ylabel("Load Voltage $U_{load}$")
    ax2.set_ylabel("Velocity $v$")

    ax1.invert_xaxis()
    ax1.grid(True, linestyle="--", alpha=0.5)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Load Voltage and Operating Velocity vs Remaining Usable Energy")
    plt.tight_layout()
    plt.show()



