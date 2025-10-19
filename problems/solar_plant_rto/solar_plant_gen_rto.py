"""Solar Plant Real-Time Optimization (RTO) Module

This module contains functions for modeling and optimizing a solar thermal
plant with collector loops, power generation and thermal oil circulation.
"""

import casadi as cas

# Constants
COLLECTOR_VALVE_RANGEABILITY = 50.0
COLLECTOR_VALVE_G_SQUIGGLE = 0.671
COLLECTOR_VALVE_ALPHA = 0.05 * (
    850 / 30**2 - 0.671 / (10 * (50 ** (0.95 - 1)) ** 2) ** 2
)
COLLECTOR_VALVE_CV = 10.0
MIRROR_CONCENTRATION_FACTOR = 52
PUMP_SPEED_MIN = 1000
PUMP_SPEED_MAX = 2970  # ✓
BOILER_FLOW_LOSS_FACTOR = 0.038
PUMP_DP_MAX = 1004.2368
PUMP_QMAX = 224.6293
PUMP_EXPONENT = 4.346734
OIL_RHO = 800  # Kg/m^3
OIL_RHO_CP = 1600
HO = 0.00361
D_OUT = 0.07  # m
GENERATOR_EFFICIENCY = 0.85  # ✓

LOOP_THERMAL_EFFICIENCIES = [  # ✓
    0.9,
    0.88,
    0.86,
    0.84,
    0.82,
    0.8,
    0.78,
    0.76,
    0.88,
    0.86,
    0.84,
    0.82,
    0.8,
    0.78,
    0.76,
]

OIL_EXIT_TEMPS_SP = (
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
    395.0,
)


# Pump and Flow Calculations


def actual_pump_speed_from_scaled(speed_scaled):
    """Convert scaled pump speed (0.2-1.0) to actual speed (rpm)."""
    return (
        PUMP_SPEED_MIN
        + (speed_scaled - 0.3) * (PUMP_SPEED_MAX - PUMP_SPEED_MIN) / 0.7
    )


def calculate_pump_and_drive_efficiency(flow_rate, actual_pump_speed):
    """Calculate pump and drive efficiency from flow rate and pump speed.
    Excel BI36: =(($AD$23/$F$1)/$C$7)*(48.91052-123.18953*(($AD$23/$F$1)/$C$7)^0.392747)
    """
    x = flow_rate / actual_pump_speed
    return x * (48.91052 - 123.18953 * x**0.392747)


def calculate_pump_fluid_power(total_flow_rate, loop_dp):
    """Calculate pump fluid power from flow rate and differential pressure."""
    return (total_flow_rate / 3600) * loop_dp


def calculate_collector_flow_rate(
    valve_position,
    loop_dp,
    rangeability=COLLECTOR_VALVE_RANGEABILITY,
    g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
    alpha=COLLECTOR_VALVE_ALPHA,
    cv=COLLECTOR_VALVE_CV,
    sqrt=cas.sqrt,
):
    """Calculate flow rate through a collector loop from valve position
    and loop dp.
    Cell AD8: =$J$28*D8*SQRT(($AC$23)/($F$4+$F$5*($J$28*D8)^2))
    """
    f = rangeability ** (valve_position - 1.0)
    flow_rate = cv * f * sqrt(loop_dp / (g_squiggle + alpha * (cv * f) ** 2))
    return flow_rate


def calculate_total_flowrate(
    valve_positions, loop_dp, sum=cas.sum1, sqrt=cas.sqrt
):
    """Calculate total flow rate from all collector loops."""
    flow_rates = calculate_collector_flow_rate(
        valve_positions, loop_dp, sqrt=sqrt
    )
    return sum(flow_rates)


def calculate_boiler_dp(total_flow_rate):
    """Calculate boiler differential pressure from total flow rate.
    Excel Q_max = M27 = (M25-M26)/K26^2
    """
    N_pumps = 5
    a = 0.05
    M25 = 850 / 30**2 - 0.671 / (10 * (50 ** (0.95 - 1)) ** 2) ** 2
    Q_max = (1.0 - a) * M25 / N_pumps**2
    boiler_dp = Q_max * total_flow_rate**2

    return boiler_dp


def calculate_pump_dp(
    actual_speed,
    total_flow_rate,
    m_pumps,
    dp_max=PUMP_DP_MAX,
    q_max=PUMP_QMAX,
    max_speed=PUMP_SPEED_MAX,
    exponent=PUMP_EXPONENT,
):
    """Calculate pump differential pressure from speed, flow rate, and
    number of pumps.
    Excel AB21: =IF((AA23/$F$1)*$C$5/($C$4*$C$7)<1,
      $C$3*(($C$7/$C$5)^2)*(1-(AA23/$F$1)*$C$5/($C$4*$C$7))^$C$6, 0
    )
    """
    dp = (
        dp_max
        * ((actual_speed / max_speed) ** 2)
        * (
            1
            - (total_flow_rate / m_pumps) * max_speed / (q_max * actual_speed)
        )
        ** exponent
    )
    return dp


def calculate_pressure_balance(loop_dp, pump_dp, boiler_dp):
    """Calculate pressure balance residual."""
    return loop_dp - (pump_dp - boiler_dp)


def make_pressure_balance_function(
    n_lines, m_pumps, sum=cas.sum1, sqrt=cas.sqrt
):
    """Create a CasADi function for pressure balance calculation."""
    valve_positions = cas.SX.sym("v", n_lines)
    pump_speed_scaled = cas.SX.sym("s")
    loop_dp = cas.SX.sym("dp")

    actual_pump_speed = actual_pump_speed_from_scaled(pump_speed_scaled)

    total_flow_rate = calculate_total_flowrate(
        valve_positions, loop_dp, sum=sum, sqrt=sqrt
    )

    boiler_dp = calculate_boiler_dp(total_flow_rate)

    pump_dp = calculate_pump_dp(actual_pump_speed, total_flow_rate, m_pumps)

    pressure_balance = calculate_pressure_balance(loop_dp, pump_dp, boiler_dp)

    return cas.Function(
        "pressure_balance",
        [valve_positions, pump_speed_scaled, loop_dp],
        [pressure_balance],
        ["valve_positions", "pump_speed_scaled", "loop_dp"],
        ["pressure_balance"],
    )


def make_calculate_pump_and_drive_power_function(
    n_lines, m_pumps, sum=cas.sum1, sqrt=cas.sqrt
):
    """Create a CasADi function for pump and drive power calculation
    with pressure balance.
    """
    valve_positions = cas.SX.sym("v", n_lines)
    pump_speed_scaled = cas.SX.sym("s")

    pressure_balance_function = make_pressure_balance_function(
        n_lines, m_pumps, sum=sum, sqrt=sqrt
    )

    # Make rootfinder to solve pressure balance
    x = cas.SX.sym("x")
    p = cas.vertcat(valve_positions, pump_speed_scaled)
    residual = pressure_balance_function(valve_positions, pump_speed_scaled, x)
    rf = cas.rootfinder("RF", "newton", {"x": x, "p": p, "g": residual})

    # Root finder solution
    sol_rf = rf(x0=[30.0], p=p)
    loop_dp = sol_rf["x"]

    flow_rates = calculate_collector_flow_rate(
        valve_positions, loop_dp, sqrt=sqrt
    )

    total_flow_rate = sum(flow_rates)

    actual_pump_speed = actual_pump_speed_from_scaled(pump_speed_scaled)

    pump_and_drive_efficiency = calculate_pump_and_drive_efficiency(
        total_flow_rate / m_pumps, actual_pump_speed
    )

    pump_dp = calculate_pump_dp(actual_pump_speed, total_flow_rate, m_pumps)

    pump_fluid_power = calculate_pump_fluid_power(total_flow_rate, pump_dp)

    pump_and_drive_power = pump_fluid_power / pump_and_drive_efficiency

    return cas.Function(
        "calculate_pump_and_drive_power",
        [valve_positions, pump_speed_scaled],
        [pump_and_drive_power],
        ["valve_positions", "pump_speed_scaled"],
        ["pump_and_drive_power"],
    )


# Oil Exit Temperature Calculations


def calculate_collector_oil_exit_temp(
    flow_rate,
    oil_return_temp,
    ambient_temp,
    solar_rate,
    loop_thermal_efficiency,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    fluid_ho=HO,
    fluid_rho_cp=OIL_RHO_CP,
    d_out=D_OUT,
    exp=cas.exp,
    pi=cas.pi,
):
    """Calculate oil exit temperature for a collector loop."""
    a = (
        solar_rate
        * mirror_concentration_factor
        * loop_thermal_efficiency
        / (2 * 1000 * fluid_ho)
    ) + ambient_temp
    b = a - oil_return_temp
    tau = (flow_rate / 3600) * fluid_rho_cp / pi / fluid_ho / d_out
    return a - b * exp(-96 / tau)


def calculate_rms_oil_exit_temps(
    oil_exit_temps, oil_exit_temps_sp, sqrt=cas.sqrt, sumsqr=cas.sumsqr
):
    """Calculate RMS deviation of oil exit temperatures from setpoints."""
    N = oil_exit_temps.shape[0]
    return sqrt(sumsqr(oil_exit_temps_sp - oil_exit_temps) / N)


# Combined System Model


def make_collector_exit_temps_and_pump_power_function(
    n_lines, m_pumps, sum=cas.sum1, sqrt=cas.sqrt, exp=cas.exp, pi=cas.pi
):
    """Create a CasADi function for complete system calculation."""
    valve_positions = cas.SX.sym("v", n_lines)
    pump_speed_scaled = cas.SX.sym("pump_speed_scaled")
    oil_return_temp = cas.SX.sym("oil_return_temp")
    ambient_temp = cas.SX.sym("ambient_temp")
    solar_rate = cas.SX.sym("solar_rate")

    pressure_balance_function = make_pressure_balance_function(
        n_lines, m_pumps, sum=sum, sqrt=sqrt
    )

    # Make rootfinder to solve pressure balance
    x = cas.SX.sym("x")
    p = cas.vertcat(valve_positions, pump_speed_scaled)
    residual = pressure_balance_function(valve_positions, pump_speed_scaled, x)
    rf = cas.rootfinder("RF", "newton", {"x": x, "p": p, "g": residual})

    # Root finder solution
    sol_rf = rf(x0=[30.0], p=p)
    loop_dp = sol_rf["x"]

    collector_flow_rates = calculate_collector_flow_rate(
        valve_positions, loop_dp, sqrt=sqrt
    )

    total_flow_rate = sum(collector_flow_rates)

    actual_pump_speed = actual_pump_speed_from_scaled(pump_speed_scaled)

    pump_and_drive_efficiency = calculate_pump_and_drive_efficiency(
        total_flow_rate / m_pumps, actual_pump_speed
    )

    pump_dp = calculate_pump_dp(actual_pump_speed, total_flow_rate, m_pumps)

    pump_fluid_power = calculate_pump_fluid_power(total_flow_rate, pump_dp)

    pump_and_drive_power = pump_fluid_power / pump_and_drive_efficiency

    oil_exit_temps = calculate_collector_oil_exit_temp(
        collector_flow_rates,
        oil_return_temp,
        ambient_temp,
        solar_rate,
        LOOP_THERMAL_EFFICIENCIES,
        exp=exp,
        pi=pi,
    )

    return cas.Function(
        "calculate_collector_exit_temps_and_pump_power",
        [
            valve_positions,
            pump_speed_scaled,
            oil_return_temp,
            ambient_temp,
            solar_rate,
        ],
        [collector_flow_rates, pump_and_drive_power, oil_exit_temps],
        [
            "valve_positions",
            "pump_speed_scaled",
            "oil_return_temp",
            "ambient_temp",
            "solar_rate",
        ],
        ["collector_flow_rates", "pump_and_drive_power", "oil_exit_temps"],
    )


# Steam Generator Model


def calculate_net_power(
    steam_power, pump_fluid_power, pump_and_drive_efficiency
):
    """Calculate net power output from steam power and pump power."""
    return (
        steam_power * GENERATOR_EFFICIENCY
        - pump_fluid_power / pump_and_drive_efficiency
    )


# RTO Solver


def solar_plant_rto_solve(
    solar_rate,
    ambient_temp,
    oil_return_temp,
    m_pumps,
    n_lines,
    valve_positions_init=0.55,
    pump_speed_scaled_init=0.6,
    max_oil_exit_temps=OIL_EXIT_TEMPS_SP,
    solver_name="ipopt",
    solver_opts=None,
):
    """Solve the solar plant RTO optimization problem.

    Maximizes net potential energy generation by optimizing valve positions
    and pump speed subject to temperature and equipment constraints.

    Parameters
    ----------
    solar_rate : float
        Solar irradiation rate (W/m^2)
    ambient_temp : float
        Ambient temperature (°C)
    oil_return_temp : float
        Oil return temperature (°C)
    m_pumps : int
        Number of pumps operating
    n_lines : int
        Number of collector lines
    valve_positions_init : float or array, optional
        Initial valve positions (default: 0.55)
    pump_speed_scaled_init : float, optional
        Initial scaled pump speed (default: 0.6)
    max_oil_exit_temps : list or array, optional
        Maximum oil exit temperatures for each line
    solver_name : str, optional
        Name of the optimizer (default: 'ipopt')
    solver_opts : dict, optional
        Solver options dictionary

    Returns
    -------
    sol : OptiSol
        CasADi optimization solution object
    variables : dict
        Dictionary containing optimized variables and outputs
    """
    if solver_opts is None:
        solver_opts = {}

    # Initialize optimization session
    opti = cas.Opti()

    # Construct system model calculation function
    calculate_exit_temps_and_pump_power = (
        make_collector_exit_temps_and_pump_power_function(n_lines, m_pumps)
    )

    max_oil_exit_temps = cas.DM(max_oil_exit_temps)

    # Decision variables
    valve_positions = opti.variable(n_lines)
    pump_speed_scaled = opti.variable()
    x = cas.vertcat(valve_positions, pump_speed_scaled)

    # System model outputs
    collector_flow_rates, pump_and_drive_power, oil_exit_temps = (
        calculate_exit_temps_and_pump_power(
            valve_positions,
            pump_speed_scaled,
            oil_return_temp,
            ambient_temp,
            solar_rate,
        )
    )
    assert collector_flow_rates.shape == (n_lines, 1)
    assert pump_and_drive_power.shape == (1, 1)
    assert oil_exit_temps.shape == (n_lines, 1)

    # Add constraints
    opti.subject_to(opti.bounded(0.1, valve_positions, 1.0))
    opti.subject_to(opti.bounded(0.2, pump_speed_scaled, 1.0))
    opti.subject_to(oil_exit_temps < max_oil_exit_temps)

    total_flow_rate = cas.sum1(collector_flow_rates)
    oil_exit_temp = (
        cas.sum1(collector_flow_rates * oil_exit_temps) / total_flow_rate
    )

    # Carnot cycle work output
    potential_work = (
        total_flow_rate
        * OIL_RHO_CP
        / 3600
        * (
            (oil_exit_temp - oil_return_temp)
            * (1.0 - (ambient_temp + 273.15) / (oil_exit_temp + 273.15))
        )
    )

    # Alternative "exergy" approach method recommended by Claude.ai
    # References:
    #   - "The availability (exergy) of a flow stream is defined as "the
    #     maximum amount of work that can be obtained by reversibly bringing
    #     the stream into equilibrium with a reference environment.", Moran,
    #     et al., Fundamentals of Engineering Thermodynamics.
    #   - Exergy/Availability,
    #     https://www.sciencedirect.com/topics/engineering/exergy-availability
    # potential_work = total_flow_rate * OIL_RHO_CP / 3600 * (
    #     (oil_exit_temp - oil_return_temp)
    #     - oil_return_temp * cas.log(oil_exit_temp / oil_return_temp)
    # )

    # Assumed efficiency of the power generator (e.g. boiler, steam turbine,
    # generator)
    expected_power_output = 0.25 * potential_work

    # Cost function - maximise net potential energy generation
    f = 1 / (expected_power_output - pump_and_drive_power)
    assert f.shape == (1, 1)
    opti.minimize(f)

    # Set initial values
    opti.set_initial(valve_positions, valve_positions_init)
    opti.set_initial(pump_speed_scaled, pump_speed_scaled_init)

    # For debugging
    grad_f = cas.gradient(f, x)
    hess_f = cas.hessian(f, x)[0]

    # Solver options
    opti.solver(solver_name, solver_opts)
    sol = opti.solve()

    variables = {
        "valve_positions": opti.value(valve_positions),
        "oil_exit_temps": opti.value(oil_exit_temps),
        "collector_flow_rates": opti.value(collector_flow_rates),
        "pump_speed_scaled": opti.value(pump_speed_scaled),
        "pump_and_drive_power": opti.value(pump_and_drive_power),
        "potential_work": opti.value(potential_work),
        "f": f,
        "grad_f": grad_f,
        "hess_f": hess_f,
    }

    return sol, variables
