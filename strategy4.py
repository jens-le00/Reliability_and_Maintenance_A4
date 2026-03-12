# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:16:22 2026

@author: MANASA HT
"""

# ============================================================
# A4 – WIND FARM MAINTENANCE SIMULATION
# STRATEGY 4 : CONDITION BASED MAINTENANCE
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# INITIALIZATION
# ============================================================

# Weibull parameters
weibull_params = {
    "rotor":     {"shape": 3, "scale": 3000},
    "bearing":   {"shape": 2, "scale": 3750},
    "gearbox":   {"shape": 3, "scale": 2400},
    "generator": {"shape": 2, "scale": 3300}
}

# Costs
C_fixed = 50
C_access = 10

# Corrective replacement costs
C1 = {"rotor":112,"bearing":60,"gearbox":152,"generator":100}

# Preventive replacement costs
C2 = {"rotor":28,"bearing":15,"gearbox":38,"generator":25}

# Condition thresholds
max_age = 0.95
min_age = 0.50

# Wind farm
N_turbines = 50
COMPONENTS = ["rotor","bearing","gearbox","generator"]

# Monte Carlo settings
ITER = 1000
N_cycles = 10

np.random.seed(42)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sample_failure_age(component):
    """Sample failure age from Weibull distribution"""
    shape = weibull_params[component]["shape"]
    scale = weibull_params[component]["scale"]
    return np.random.weibull(shape) * scale


def init_windfarm():
    """Create turbines and components"""
    farm = []

    for _ in range(N_turbines):
        turbine = {}

        for comp in COMPONENTS:
            turbine[comp] = {
                "virtual_age":0.0,
                "failure_age":sample_failure_age(comp)
            }

        farm.append(turbine)

    return farm

# ============================================================
# PLOT FUNCTIONS
# ============================================================

def plot_amc_stabilization(all_AMC):

    running_avg = np.cumsum(all_AMC) / np.arange(1,len(all_AMC)+1)

    plt.figure(figsize=(7,4))
    plt.plot(running_avg)
    plt.title("AMC Stabilization Curve")
    plt.xlabel("Monte Carlo Iteration")
    plt.ylabel("Average AMC")
    plt.grid()
    plt.show()


def plot_coefficient_of_variation(all_AMC):

    running_mean = np.cumsum(all_AMC) / np.arange(1,len(all_AMC)+1)

    running_var = (np.cumsum((all_AMC-running_mean)**2)) / np.arange(1,len(all_AMC)+1)

    running_std = np.sqrt(running_var)

    running_cv = running_std / running_mean

    plt.figure(figsize=(7,4))
    plt.plot(running_cv)
    plt.title("Coefficient of Variation Convergence")
    plt.xlabel("Monte Carlo Iteration")
    plt.ylabel("CV")
    plt.grid()
    plt.show()


# ============================================================
# MONTE CARLO FUNCTION
# ============================================================

def run_simple_montecarlo(strategy_function):

    all_AMC = []
    all_total_cost = []
    all_total_days = []
    all_fail_counts = []

    for _ in range(ITER):

        total_cost,total_days,AMC,failure_counts = strategy_function()

        all_AMC.append(AMC)
        all_total_cost.append(total_cost)
        all_total_days.append(total_days)
        all_fail_counts.append(failure_counts)

    avg_AMC = float(np.mean(all_AMC))
    std_AMC = float(np.std(all_AMC,ddof=1))
    avg_total_cost = float(np.mean(all_total_cost))
    avg_total_days = float(np.mean(all_total_days))

    avg_failures = {
        comp: float(np.mean([fc[comp] for fc in all_fail_counts]))
        for comp in COMPONENTS
    }

    return all_AMC,avg_AMC,std_AMC,avg_total_cost,avg_total_days,avg_failures


# ============================================================
# STRATEGY 4 – CONDITION BASED MAINTENANCE
# ============================================================

def simulate_S4():

    farm = init_windfarm()

    total_cost = 0.0
    total_days = 0.0

    failure_counts = {c:0 for c in COMPONENTS}

    for c in range(N_cycles):

        min_residual = np.inf
        failed_turbine_idx = None
        failed_component = None


        # ------------------------------------------------
        # FIND NEXT FAILURE EVENT
        # ------------------------------------------------
        for ti in range(N_turbines):
            for comp in COMPONENTS:

                comp_state = farm[ti][comp]

                residual = comp_state["failure_age"] - comp_state["virtual_age"]

                if residual < min_residual:

                    min_residual = residual
                    failed_turbine_idx = ti
                    failed_component = comp


        # ------------------------------------------------
        # ADVANCE TIME
        # ------------------------------------------------
        dt = max(min_residual,0)

        total_days += dt

        for ti in range(N_turbines):
            for comp in COMPONENTS:
                farm[ti][comp]["virtual_age"] += dt


        # ------------------------------------------------
        # CONDITION BASED PREVENTIVE MAINTENANCE
        # ------------------------------------------------
        for ti in range(N_turbines):
            for comp in COMPONENTS:

                comp_state = farm[ti][comp]

                health_ratio = comp_state["virtual_age"] / comp_state["failure_age"]

                if health_ratio >= max_age:

                    cycle_cost = C_fixed + C_access + C2[comp]

                    total_cost += cycle_cost

                    comp_state["virtual_age"] = 0.0
                    comp_state["failure_age"] = sample_failure_age(comp)


        # ------------------------------------------------
        # CORRECTIVE MAINTENANCE
        # ------------------------------------------------
        cycle_cost = C_fixed + C_access + C1[failed_component]

        total_cost += cycle_cost

        failure_counts[failed_component] += 1

        farm[failed_turbine_idx][failed_component]["virtual_age"] = 0.0
        farm[failed_turbine_idx][failed_component]["failure_age"] = sample_failure_age(failed_component)


    AMC = total_cost / total_days * 365.0

    return total_cost,total_days,AMC,failure_counts


# ============================================================
# EXECUTION
# ============================================================

all_AMC,avg_AMC,std_AMC,avg_total_cost,avg_total_days,avg_failures = run_simple_montecarlo(simulate_S4)


print("============================================================")
print("CONDITION BASED MAINTENANCE (STRATEGY 4)")
print("============================================================")

print(f"Iterations: {ITER}")
print(f"Turbines: {N_turbines}")
print(f"Cycles per iteration: {N_cycles}")

print(f"\nAverage Total Cost: {avg_total_cost:.2f} EUR")
print(f"Average Total Days: {avg_total_days:.2f} days")

print(f"\nAverage AMC: {avg_AMC:.2f} EUR/year")
print(f"Standard Deviation AMC: {std_AMC:.2f}")

print("\nAverage Failure Counts:")

for comp,val in avg_failures.items():
    print(f"{comp}: {val:.2f}")

# ============================================================
# PLOTS
# ============================================================

plot_amc_stabilization(all_AMC)
plot_coefficient_of_variation(all_AMC)