import math
from dataclasses import dataclass
from typing import Callable, List, Optional
import matplotlib.pyplot as plt

# =========================
# Plot style: single color
# =========================
MAIN_BLUE = "#2878B5"
plt.style.use("default")


# ============================================================
# Model parameters
# ============================================================

@dataclass
class ModelParams:
    # Energy pool geometry/scale
    V: float            # [unit volume] effective energy-pool volume

    # Channel geometry
    S: float            # [area] channel cross-sectional area

    # For rho nonlinearity
    E_max: float        # [J] full-charge energy (normalization)
    P_min: float        # [W] minimum acceptable output power (defines rho_min)
    gamma: float        # [-] nonlinearity exponent (>=0)

    # Channel dissipation strength
    mu: float           # [1/speed] friction coefficient in P_loss = mu * rho * S * v^2

    # Thermal parameters
    C_th: float         # [J/K] equivalent thermal capacity
    h: float            # [W/K] heat transfer coefficient to ambient
    T_a: float          # [degC] ambient temperature

    # alpha(T) parameters (kept for completeness, not used in alpha_of_T below)
    alpha_min: float
    beta: float

    # Reference temperature (kept for completeness)
    T_ref: float = 25.0


# ============================================================
# alpha(T): low-temperature capability
# (Your piecewise/log form)
# ============================================================

def alpha_of_T(T: float) -> float:
    """
    Low-temperature capability alpha(T)
    """
    if T >= 10.0:
        return 1.0
    elif T >= -30.0:
        return max(0.1, math.log10((9.0 / 40.0) * (T + 30.0) + 1.0))
    else:
        return 0.1


# ============================================================
# One-step DAE + Euler update
# ============================================================

@dataclass
class StepResult:
    # Updated state
    E_next: float
    T_next: float

    # Capabilities
    rho_raw: float
    rho_loop: float

    # Diagnostics
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
    # --------- checks ----------
    if p.V <= 0:
        raise ValueError("V must be > 0")
    if p.S <= 0:
        raise ValueError("S must be > 0")
    if p.mu <= 0:
        raise ValueError("mu must be > 0")
    if p.C_th <= 0:
        raise ValueError("C_th must be > 0")
    if p.h < 0:
        raise ValueError("h must be >= 0")
    if p.E_max <= 0:
        raise ValueError("E_max must be > 0")
    if p.gamma < 0:
        raise ValueError("gamma must be >= 0")

    # --------- rho, alpha ----------
    E_eff = max(E_n, 0.0)
    soc = min(max(E_eff / p.E_max, 0.0), 1.0)
    rho_raw = E_eff / p.V
    alpha_n = alpha_of_T(T_n)
    rho_n = rho_raw * (soc ** p.gamma) * alpha_n  # loop capability

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
            cutoff_reason="Energy depleted (rho <= 0)",
        )

    # --------- demand power ----------
    P_dem = float(P_func(t_n, E_n, T_n))
    if P_dem < 0:
        P_dem = 0.0

    # no demand: only cooling
    if P_dem == 0.0:
        dTdt = (0.0 - p.h * (T_n - p.T_a)) / p.C_th
        return StepResult(
            E_next=E_n,
            T_next=T_n + dTdt * dt,
            rho_raw=rho_raw,
            rho_loop=rho_n,
            alpha=alpha_n,
            v=0.0,
            P_out=0.0,
            P_loss=0.0,
            P_use=0.0,
            discriminant=1.0,
            feasible=True,
            cutoff_reason=None,
        )

    # --------- algebraic constraint: solve v ----------
    rhs = P_dem / (rho_n * p.S)  # >0
    disc = 1.0 - 4.0 * p.mu * rhs

    if disc < 0:
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
            cutoff_reason="Energy-side cutoff: discriminant < 0 (no feasible v)",
        )

    sqrt_disc = math.sqrt(disc)
    v_plus = (1.0 + sqrt_disc) / (2.0 * p.mu)
    v_minus = (1.0 - sqrt_disc) / (2.0 * p.mu)
    v_n = v_minus if choose_small_root else v_plus

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
            cutoff_reason="Numerical anomaly: chosen v is negative",
        )

    # --------- power decomposition ----------
    P_out = rho_n * v_n * p.S
    P_loss = p.mu * rho_n * p.S * (v_n ** 2)
    P_use = P_out - P_loss  # should ~ P_dem

    # --------- Euler update ----------
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
# Simulation driver
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
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t_end <= 0:
        raise ValueError("t_end must be > 0")

    n_steps = int(math.ceil(t_end / dt))

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

        P_dem = float(P_func(t_n, E_n, T_n))
        if P_dem < 0:
            P_dem = 0.0

        sr = step_euler_dae(E_n, T_n, t_n, dt, P_func, p, choose_small_root=choose_small_root)

        # record current time (pre-update)
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

        if not sr.feasible:
            cutoff_index = n
            cutoff_reason = sr.cutoff_reason
            break

        E_n, T_n = sr.E_next, sr.T_next

        if E_n <= 0:
            cutoff_index = n
            cutoff_reason = "Energy depleted after update (E <= 0)"
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
# Example power demand model
# ============================================================

def example_P_func(t: float, E: float, T: float) -> float:
    # constant demand
    return 4.2


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    params = ModelParams(
        V=500.0,
        S=1.0,
        E_max=50000.0,
        P_min=4.2,
        gamma=0.3,
        mu=0.3,
        C_th=200.0,
        h=1.0,
        T_a=25.0,
        alpha_min=0.5,
        beta=100.0,
    )

    E0 = params.E_max
    T0 = params.T_a

    dt = 0.2
    t_end = 50000.0

    result = simulate(E0, T0, t_end, dt, example_P_func, params, choose_small_root=True)

    print(f"Steps: {len(result.t)}")
    if result.cutoff_index is not None:
        k = result.cutoff_index
        print(f"Cutoff at step {k}, time {result.t[k]:.3f} s")
        print(f"Reason: {result.cutoff_reason}")
        print(f"E at cutoff: {result.E[k]:.6g}")
        print(f"T at cutoff: {result.T[k]:.6g}")
        print(f"alpha at cutoff: {result.alpha[k]:.6g}")
        print(f"disc at cutoff: {result.disc[k]:.6g}")
    else:
        print("No cutoff within simulation horizon.")

    # ================== Post-processing ==================
    rho_loop = result.rho_loop
    rho_loop_min = 4.0 * params.mu * params.P_min / params.S
    rho_loop_min_list = [rho_loop_min for _ in result.t]

    remaining_frac = [
        max(min((rho_l - rho_min) / (rho_loop[0] - rho_min), 1.0), 0.0)
        for rho_l, rho_min in zip(rho_loop, rho_loop_min_list)
    ]

    # Debug prints (safe if cutoff exists)
    if result.cutoff_index is not None:
        k = result.cutoff_index
        print("rho_loop(t):", rho_loop[k])
        print("rho_loop_min(t):", rho_loop_min_list[k])
        print("rho_loop(0):", rho_loop[0])
        print("rho_loop_min(0):", rho_loop_min_list[0])
        print("numerator:", rho_loop[k] - rho_loop_min_list[k])
        print("denominator:", rho_loop[0] - rho_loop_min_list[k])

    # ================== Energy vs Time ==================
    plt.figure()
    plt.plot(result.t, result.E, color=MAIN_BLUE, linewidth=2.2)
    plt.xlabel("Time t (s)")
    plt.ylabel("Energy E(t) (J)")
    plt.title("Energy vs Time")
    plt.grid(True)

    if result.cutoff_index is not None:
        plt.axvline(result.t[result.cutoff_index], linestyle="--", color=MAIN_BLUE, alpha=0.7)

    plt.tight_layout()
    plt.show()

    # ================== Remaining vs Time ==================
    plt.figure()
    plt.plot(result.t, remaining_frac, color=MAIN_BLUE, linewidth=2.2)
    plt.xlabel("Time t (s)")
    plt.ylabel("Remaining Usable Energy")
    plt.title("State of Charge (Remaining Usable Energy)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ================== Temperature vs Time ==================
    plt.figure()
    plt.plot(result.t, result.T, color=MAIN_BLUE, linewidth=2.2)
    plt.xlabel("Time t (s)")
    plt.ylabel("Temperature T(t) (°C)")
    plt.title("Temperature Evolution")
    plt.grid(True)

    if result.cutoff_index is not None:
        plt.axvline(result.t[result.cutoff_index], linestyle="--", color=MAIN_BLUE, alpha=0.7)

    plt.tight_layout()
    plt.show()

    # ================== Load voltage & v vs Remaining ==================
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
        color=MAIN_BLUE,
        linewidth=2.2,
        label="Load Voltage $U_{load}$"
    )

    line2, = ax2.plot(
        remaining_frac,
        result.v,
        color=MAIN_BLUE,
        linestyle="--",
        linewidth=2.2,
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
    ax1.legend(lines, labels, frameon=False, loc="best")

    plt.title("Load Voltage and Operating Velocity vs Remaining Usable Energy")
    plt.tight_layout()
    plt.show()
