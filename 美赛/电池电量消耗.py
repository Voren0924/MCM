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
    alpha_n = alpha_of_T(T_n, p)
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
        sr = step_euler_dae(E_n, T_n, t_n, dt, P_func, p)

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
            cutoff_index = n + 1
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
    if t < 200:
        return 1
    elif t < 600:
        return 1.0
    elif t < 1000:
        return 1.0
    else:
        return 1.0


# ============================================================
# 【新增模块】科研绘图专用函数 (无物理逻辑修改)
# ============================================================

def setup_publication_style():
    """配置 Matplotlib 以符合一般科研论文要求 (Times New Roman, 适中字号)"""
    import matplotlib as mpl

    # 字体配置 (优先使用 Times New Roman，如果没有则退回 DejaVu Serif)
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    mpl.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体与 Times 搭配较好

    # 字号配置
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12

    # 线条与刻度
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.4
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    # 输出设置
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'


def plot_for_publication(
        result: SimResult,
        remaining_frac: List[float],
        u_load: List[float],
        output_prefix: str = "fig_simulation"
):
    """
    生成并保存符合论文格式的图表。

    参数:
      result: 仿真结果对象
      remaining_frac: 后处理计算得到的剩余电量比例
      u_load: 后处理计算得到的负载电压
      output_prefix: 输出文件的前缀
    """
    setup_publication_style()

    # 定义颜色和线型策略（黑白友好）
    c_energy = '#1f77b4'  # 深蓝
    c_temp = '#d62728'  # 深红
    c_volt = '#000000'  # 黑色
    c_vel = '#2ca02c'  # 绿色

    ls_solid = '-'
    ls_dash = '--'

    # --------------------------------------------------------
    # 图 1: 能量与温度随时间变化 (双轴展示，整合信息)
    # --------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    # 左轴：能量
    l1, = ax1.plot(result.t, result.E, color=c_energy, linestyle=ls_solid, label='Energy $E(t)$ [J]')
    ax1.set_xlabel('Time $t$ [s]')
    ax1.set_ylabel('Energy $E$ [J]', color=c_energy)
    ax1.tick_params(axis='y', labelcolor=c_energy)

    # 截断线
    if result.cutoff_index is not None:
        vline = ax1.axvline(result.t[result.cutoff_index], color='gray', linestyle=':', alpha=0.8, label='Cutoff')

    # 右轴：温度
    ax2 = ax1.twinx()
    l2, = ax2.plot(result.t, result.T, color=c_temp, linestyle=ls_dash, label='Temp $T(t)$ [°C]')
    ax2.set_ylabel('Temperature $T$ [°C]', color=c_temp)
    ax2.tick_params(axis='y', labelcolor=c_temp)

    # 合并图例
    lines = [l1, l2]
    if result.cutoff_index is not None:
        lines.append(vline)
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', frameon=True, framealpha=0.9)

    ax1.set_title('Energy and Temperature Evolution')

    # 保存
    plt.savefig(f"{output_prefix}_energy_temp.pdf")
    plt.savefig(f"{output_prefix}_energy_temp.png")
    print(f"图表已保存: {output_prefix}_energy_temp.pdf")

    # --------------------------------------------------------
    # 图 2: 剩余可用电量 (Remaining Usable Energy)
    # --------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.t, remaining_frac, color='black', linewidth=2, label='Remaining Fraction')

    ax.set_xlabel('Time $t$ [s]')
    ax.set_ylabel('Remaining Usable Capacity (Normalized)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Usable Energy Depletion Profile')

    # 添加辅助线
    ax.axhline(0, color='red', linewidth=1, linestyle='-')
    ax.axhline(1, color='gray', linewidth=1, linestyle=':')

    if result.cutoff_index is not None:
        ax.axvline(result.t[result.cutoff_index], color='gray', linestyle=':', label='Cutoff')
        ax.legend()

    plt.savefig(f"{output_prefix}_capacity.pdf")
    print(f"图表已保存: {output_prefix}_capacity.pdf")

    # --------------------------------------------------------
    # 图 3: 负载电压与速度 vs 剩余电量 (双轴关键特性图)
    # --------------------------------------------------------
    fig3, ax_v1 = plt.subplots(figsize=(8, 5))

    # 设置 X 轴倒序 (从 1.0 到 0.0)
    ax_v1.set_xlim(1.05, -0.05)

    # 左轴：电压 (使用实线)
    ln1, = ax_v1.plot(remaining_frac, u_load, color=c_volt, linestyle=ls_solid, linewidth=2,
                      label=r'Load Voltage $U_{\mathrm{load}}$')
    ax_v1.set_xlabel('Remaining Usable Energy Fraction')
    ax_v1.set_ylabel(r'Voltage [V]', color=c_volt)
    ax_v1.tick_params(axis='y', labelcolor=c_volt)

    # 右轴：速度 (使用虚线)
    ax_v2 = ax_v1.twinx()
    ln2, = ax_v2.plot(remaining_frac, result.v, color=c_vel, linestyle=ls_dash, linewidth=2, label=r'Velocity $v$')
    ax_v2.set_ylabel(r'Velocity [m/s]', color=c_vel)
    ax_v2.tick_params(axis='y', labelcolor=c_vel)

    # 合并图例
    lns = [ln1, ln2]
    labs = [l.get_label() for l in lns]
    ax_v1.legend(lns, labs, loc='upper center', frameon=True, framealpha=0.9)

    ax_v1.set_title('Performance Degradation vs. Energy State')

    plt.savefig(f"{output_prefix}_performance.pdf")
    print(f"图表已保存: {output_prefix}_performance.pdf")

    # 展示所有图片
    plt.show()


if __name__ == "__main__":
    # ============================================================
    # 1. 仿真参数与执行 (保持不变)
    # ============================================================
    # 设置模型参数（以下数值仅为占位符，需通过数据进行标定）
    params = ModelParams(
        V=1000.0,
        S=1.0,
        E_max=50000.0,  # 你满电能量（通常和 E0 一致）
        P_min=1.0,
        gamma=0.2,
        mu=0.5,
        C_th=200.0,
        h=1.0,
        T_a=20.0,
        alpha_min=0.5,
        beta=100.0
    )

    # 初始条件
    E0 = params.E_max  # 初始总能量（单位与 W·s = J 一致）
    T0 = params.T_a  # 初始温度（摄氏度）

    # 仿真设置
    dt = 0.2
    t_end = 70000.0

    # 执行仿真
    result = simulate(E0, T0, t_end, dt, example_P_func, params)

    # ============================================================
    # 2. 文本输出 (保持不变)
    # ============================================================
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

    # ============================================================
    # 3. 后处理计算 (保持逻辑不变，仅为了准备绘图数据)
    # ============================================================

    # 回路等效能力
    rho_loop = result.rho_loop

    # 回路侧能力阈值计算
    rho_loop_min_list = [
        4.0 * params.mu * params.P_min / (params.S * max(alpha, 1e-12) ** 2)
        for alpha in result.alpha
    ]

    # 剩余可用电量计算
    remaining_frac = [
        max(min((rho_l - rho_min) / (rho_loop[0] - rho_min), 1.0), 0.0)
        for rho_l, rho_min in zip(rho_loop, rho_loop_min_list)
    ]

    # 调试打印 (保留)
    if result.cutoff_index is not None:
        k = result.cutoff_index
        print("rho_loop(t):", rho_loop[k])
        print("rho_loop_min(t):", rho_loop_min_list[k])
        print("rho_loop(0):", rho_loop[0])
        print("rho_loop_min(0):", rho_loop_min_list[0])
        print("numerator:", rho_loop[k] - rho_loop_min_list[k])
        print("denominator:", rho_loop[0] - rho_loop_min_list[k])

    # 负载端电压计算
    u_load = [
        rho_l * params.S - params.mu * rho_l * params.S * v
        for rho_l, v in zip(rho_loop, result.v)
    ]

    # ============================================================
    # 4. 调用新的科研绘图函数 (替代原本散落的 plt 代码)
    # ============================================================
    print("\n正在生成科研图表...")
    plot_for_publication(result, remaining_frac, u_load, output_prefix="paper_fig")
    print("绘图完成。")



