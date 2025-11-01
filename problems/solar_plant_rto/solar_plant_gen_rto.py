"""Solar Plant Real-Time Optimization (RTO) Module

This module contains functions for modeling and optimizing a solar thermal
plant with collector loops, power generation and thermal oil circulation.

Functions
---------

Fluid Property Functions
    calculate_fluid_density
    calculate_fluid_viscosity
    calculate_fluid_heat_capacity
    calculate_fluid_thermal_conductivity
    calculate_reynolds_number
    calculate_prandtl_number
    calculate_heat_transfer_coefficient_nusselt
    calculate_heat_transfer_coefficient_turbulent

Pump and Flow Calculations
    actual_pump_speed_from_scaled
    calculate_pump_and_drive_efficiency
    calculate_pump_fluid_power
    calculate_collector_flow_rate
    calculate_total_oil_flowrate
    calculate_boiler_dp
    calculate_pump_dp
    calculate_pressure_balance
    make_pressure_balance_function
    make_calculate_pump_and_drive_power_function

Collector Line Temperature Calculations
    calculate_collector_oil_exit_temp
    calculate_collector_oil_exit_and_mean_temps
    calculate_collector_oil_exit_and_mean_temps_no_loss
    calculate_mixed_oil_exit_temp
    calculate_rms_oil_exit_temps

Combined System Model
    make_calculate_collector_exit_temps_and_pump_power

Steam Generator Model
    calculate_dtlm_hx1
    calculate_dtlm_hx2
    calculate_dtlm_hx3
    calculate_actual_heat_transfer_coefficient
    calculate_Q_dot_hx1
    calculate_Q_dot_hx2
    calculate_Q_dot_hx3
    calculate_hx_area
    calculate_hx_temperatures
    heat_exchanger_solution_error
    calculate_steam_power
    calculate_net_power

RTO Solvers
    solar_plant_gen_rto_solve
    steam_generator_solve
    solar_plant_rto_solve
    solar_plant_gen_db_rto_solve
"""

import casadi as cas

# =============================================================================
# DEFAULT PARAMETERS & CONSTANTS
# =============================================================================

# Collector lines
COLLECTOR_D_INT = 0.066  # m
COLLECTOR_D_OUT = 0.07  # m
COLLECTOR_LENGTH = 96.0  # m

# Collector FCV parameters
COLLECTOR_VALVE_RANGEABILITY = 50.0
COLLECTOR_VALVE_G_SQUIGGLE = 0.671
COLLECTOR_VALVE_ALPHA = 0.05 * (
    850 / 30**2 - 0.671 / (10 * (50 ** (0.95 - 1)) ** 2) ** 2
)
COLLECTOR_VALVE_CV = 10.0
MIRROR_CONCENTRATION_FACTOR = 52
H_OUTER = 0.00361  # kW/m2-K

# Oil pumps and flows
PUMP_SPEED_MIN = 1000
PUMP_SPEED_MAX = 2970
BOILER_FLOW_LOSS_FACTOR = 0.038
PUMP_DP_MAX = 1004.2368
PUMP_QMAX = 224.6293
PUMP_EXPONENT = 4.346734

# Oil properties - static
OIL_RHO = 636.52  # Kg/m^3 (Syltherm800 at 330 °C)
OIL_RHO_CP = OIL_RHO * 2.138  # kJ/m^3-K (Syltherm800 at 330 °C)

# Time-varying property correlations for Syltherm800
OIL_T_min_C = 200  # degC
OIL_T_max_C = 400  # degC
OIL_RHO_A0 = 1312.3439956092
OIL_RHO_A1 = -1.1259259736573723
OIL_CP_A0 = 1108.027261915654
OIL_CP_A1 = 1.707142857126835
OIL_CONDUCTIVITY_A0 = 0.19090761573601903
OIL_CONDUCTIVITY_A1 = -0.00018939883743798126
OIL_VISCOSITY_A = 3.940589927031612e-05
OIL_VISCOSITY_B = 1636.9992862433103
OIL_VISCOSITY_C = -0.00021147616722260807

# Power generator
GENERATOR_EFFICIENCY = 0.85

LOOP_THERMAL_EFFICIENCIES = [
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

# Power generator default constants
T_CONDENSATE = 75.0  # deg. C
T_STEAM = 385.0  # deg. C
H_VAP = 2260.0  # kJ/kg
CP_STEAM = 1.996  # kJ/kg-K
CP_WATER = 4.182  # kJ/kg-K
BOILER_T_STEAM_SP = 385.0  # deg. C
BOILER_T_BOIL = 310.0  # deg. C
BOILER_T_CONDENSATE = 60  # deg. C
HX_AREA = 0.25  # m^2
HX1_U_LIQUID = 900.0  # W/m^2-K
HX2_U_BOIL = 3000.0  # W/m^2-K
HX3_U_STEAM = 1000.0  # W/m^2-K
F_OIL_NOMINAL = 200.0 / 3600.0  # m^3/s
TURBINE_DELTA_H = 3049.0 - 2207.0  # kJ/kg

# =============================================================================
# THERMAL FLUID PROPERTIES
# =============================================================================


def calculate_fluid_density(T_K, a0=OIL_RHO_A0, a1=OIL_RHO_A1):
    """Linear approximation for Syltherm800 density (kg/m³) fitted to
    data, valid for temperatures in the range 200 to 400°C.
    """
    return a0 + a1 * T_K


def calculate_fluid_viscosity(
    T_K, A=OIL_VISCOSITY_A, B=OIL_VISCOSITY_B, C=OIL_VISCOSITY_C, exp=cas.exp
):
    """Exponential approximation for Syltherm800 dynamic viscosity (Pa·s)
    fitted to data, valid for temperatures in the range 200 to 400°C.
    """
    return A * exp(B / T_K) + C


def calculate_fluid_heat_capacity(T_K, a0=OIL_CP_A0, a1=OIL_CP_A1):
    """Linear approximation for Syltherm800 specific heat capacity (J/kg-K)
    fitted to data, valid for temperatures in the range 200 to 400°C.
    """
    return a0 + a1 * T_K


def calculate_fluid_thermal_conductivity(
    T_K, a0=OIL_CONDUCTIVITY_A0, a1=OIL_CONDUCTIVITY_A1
):
    """Linear approximation for Syltherm800 thermal conductivity (W/m-K)
    fitted to data, valid for temperatures in the range 200 to 400°C.
    """
    return a0 + a1 * T_K


def calculate_reynolds_number(
    velocity, pipe_diameter, fluid_density, fluid_viscosity
):
    """Calculate Reynolds number for pipe flow."""
    return fluid_density * velocity * pipe_diameter / fluid_viscosity


def calculate_prandtl_number(
    fluid_viscosity, fluid_specific_heat, fluid_thermal_conductivity
):
    """Calculate Prandtl number for fluid."""
    return fluid_viscosity * fluid_specific_heat / fluid_thermal_conductivity


def calculate_heat_transfer_coefficient_nusselt(
    pipe_diameter, fluid_thermal_conductivity, Nu
):
    """Calculate internal heat transfer coefficient for laminar flow
    in a pipe.

    Valid range:
    - 0.7 ≤ Pr ≤ 160
    - Re > 10,000 (but works reasonably well down to Re ≈ 4000)
    - L/D > 10 (fully developed flow)
    """

    # Heat transfer coefficient
    h = Nu * fluid_thermal_conductivity / pipe_diameter

    return h


def calculate_heat_transfer_coefficient_turbulent(
    velocity,
    pipe_diameter,
    fluid_density,
    fluid_viscosity,
    fluid_thermal_conductivity,
    fluid_specific_heat,
):
    """
    Calculate internal heat transfer coefficient in a pipe using
    Dittus-Boelter correlation.

    The Dittus-Boelter correlation is widely used for turbulent flow in
    smooth pipes with moderate temperature differences. It provides
    sufficient accuracy (±25%) for most engineering applications.

    Parameters:
    -----------
    velocity : float
        Fluid velocity [m/s]
    pipe_diameter : float
        Pipe inner diameter [m]
    fluid_density : float
        Fluid density [kg/m³]
    fluid_viscosity : float
        Dynamic viscosity [Pa·s]
    fluid_thermal_conductivity : float
        Thermal conductivity [W/m·K]
    fluid_specific_heat : float
        Specific heat capacity [J/kg·K]

    Returns:
    --------
    h : float
        Heat transfer coefficient [W/m²·K]
    Re : float
        Reynolds number [-]
    Pr : float
        Prandtl number [-]
    Nu : float
        Nusselt number [-]

    Notes:
    ------
    - Reynolds number: Re = ρ*v*D/μ (inertial forces / viscous forces)
    - Prandtl number: Pr = μ*cp/k (momentum diffusivity / thermal diffusivity)
    - Nusselt number: Nu = h*D/k (convective / conductive heat transfer)

    Correlations used:
    - Turbulent flow (Re > 4000): Nu = 0.023 * Re^0.8 * Pr^0.4

    For Laminar flow (Re ≤ 4000), use Nu = 4.36 and call
    calculate_heat_transfer_coefficient_nusselt directly.

    Valid range:
    - 0.7 ≤ Pr ≤ 160
    - Re > 10,000 (but works reasonably well down to Re ≈ 4000)
    - L/D > 10 (fully developed flow)
    """
    # Reynolds number
    Re = calculate_reynolds_number(
        velocity, pipe_diameter, fluid_density, fluid_viscosity
    )

    # Prandtl number
    Pr = calculate_prandtl_number(
        fluid_viscosity, fluid_specific_heat, fluid_thermal_conductivity
    )

    # Nusselt number (Dittus-Boelter correlation)
    Nu = 0.023 * Re**0.8 * Pr**0.4

    # Heat transfer coefficient
    h = calculate_heat_transfer_coefficient_nusselt(
        pipe_diameter, fluid_thermal_conductivity, Nu
    )

    return h, Re, Pr, Nu


# =============================================================================
# PUMP AND FLOW CALCULATIONS
# =============================================================================


def actual_pump_speed_from_scaled(
    speed_scaled, speed_min=PUMP_SPEED_MIN, speed_max=PUMP_SPEED_MAX
):
    """Convert scaled pump speed (0.3-1.0) to actual speed (rpm)."""
    return speed_min + (speed_scaled - 0.3) * (speed_max - speed_min) / 0.7


def calculate_pump_and_drive_efficiency(flow_rate, actual_pump_speed):
    """Calculate pump and drive efficiency from flow rate and pump speed.
    Excel BI36:
      =(($AD$23/$F$1)/$C$7)*(48.91052-123.18953*(($AD$23/$F$1)/$C$7)^0.392747)
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


def calculate_total_oil_flowrate(
    valve_positions,
    loop_dp,
    rangeability=COLLECTOR_VALVE_RANGEABILITY,
    g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
    alpha=COLLECTOR_VALVE_ALPHA,
    cv=COLLECTOR_VALVE_CV,
    sum=cas.sum1,
    sqrt=cas.sqrt,
):
    """Calculate total flow rate from all collector loops."""
    flow_rates = calculate_collector_flow_rate(
        valve_positions,
        loop_dp,
        rangeability=rangeability,
        g_squiggle=g_squiggle,
        alpha=alpha,
        cv=cv,
        sqrt=sqrt,
    )
    return sum(flow_rates)


def calculate_boiler_dp(total_flow_rate):
    """Calculate boiler differential pressure from total flow rate.
    Excel Q_max = M27 = (M25-M26)/K26^2
    """
    N_pumps = 5  # TODO: Is this correct?
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
    speed_max=PUMP_SPEED_MAX,
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
        * ((actual_speed / speed_max) ** 2)
        * (
            1
            - (total_flow_rate / m_pumps) * speed_max / (q_max * actual_speed)
        )
        ** exponent
    )
    return dp


def calculate_pressure_balance(loop_dp, pump_dp, boiler_dp):
    """Calculate pressure balance residual."""
    return loop_dp - (pump_dp - boiler_dp)


def make_pressure_balance_function(
    n_lines,
    m_pumps,
    speed_min=PUMP_SPEED_MIN,
    speed_max=PUMP_SPEED_MAX,
    rangeability=COLLECTOR_VALVE_RANGEABILITY,
    g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
    alpha=COLLECTOR_VALVE_ALPHA,
    cv=COLLECTOR_VALVE_CV,
    dp_max=PUMP_DP_MAX,
    q_max=PUMP_QMAX,
    exponent=PUMP_EXPONENT,
    sum=cas.sum1,
    sqrt=cas.sqrt,
):
    """Create a CasADi function for pressure balance calculation."""
    valve_positions = cas.SX.sym("v", n_lines)
    pump_speed_scaled = cas.SX.sym("s")
    loop_dp = cas.SX.sym("dp")

    actual_pump_speed = actual_pump_speed_from_scaled(
        pump_speed_scaled, speed_min=speed_min, speed_max=speed_max
    )

    total_flow_rate = calculate_total_oil_flowrate(
        valve_positions,
        loop_dp,
        rangeability=rangeability,
        g_squiggle=g_squiggle,
        alpha=alpha,
        cv=cv,
        sum=sum,
        sqrt=sqrt,
    )

    boiler_dp = calculate_boiler_dp(total_flow_rate)  # TODO: Add params?

    pump_dp = calculate_pump_dp(
        actual_pump_speed,
        total_flow_rate,
        m_pumps,
        dp_max=dp_max,
        q_max=q_max,
        speed_max=speed_max,
        exponent=exponent,
    )

    pressure_balance = calculate_pressure_balance(loop_dp, pump_dp, boiler_dp)

    return cas.Function(
        "pressure_balance",
        [valve_positions, pump_speed_scaled, loop_dp],
        [pressure_balance],
        ["valve_positions", "pump_speed_scaled", "loop_dp"],
        ["pressure_balance"],
    )


def make_calculate_pump_and_drive_power_function(
    n_lines,
    m_pumps,
    speed_min=PUMP_SPEED_MIN,
    speed_max=PUMP_SPEED_MAX,
    rangeability=COLLECTOR_VALVE_RANGEABILITY,
    g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
    alpha=COLLECTOR_VALVE_ALPHA,
    cv=COLLECTOR_VALVE_CV,
    dp_max=PUMP_DP_MAX,
    q_max=PUMP_QMAX,
    exponent=PUMP_EXPONENT,
    sum=cas.sum1,
    sqrt=cas.sqrt,
):
    """Create a CasADi function for pump and drive power calculation
    with pressure balance.
    """
    valve_positions = cas.SX.sym("v", n_lines)
    pump_speed_scaled = cas.SX.sym("s")

    pressure_balance_function = make_pressure_balance_function(
        n_lines,
        m_pumps,
        speed_min=speed_min,
        speed_max=speed_max,
        rangeability=rangeability,
        g_squiggle=g_squiggle,
        alpha=alpha,
        cv=cv,
        dp_max=dp_max,
        q_max=q_max,
        exponent=exponent,
        sum=sum,
        sqrt=sqrt,
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
        valve_positions,
        loop_dp,
        rangeability=rangeability,
        g_squiggle=g_squiggle,
        alpha=alpha,
        cv=cv,
        sqrt=sqrt,
    )

    total_flow_rate = sum(flow_rates)

    actual_pump_speed = actual_pump_speed_from_scaled(
        pump_speed_scaled, speed_min=speed_min, speed_max=speed_max
    )

    pump_and_drive_efficiency = calculate_pump_and_drive_efficiency(
        total_flow_rate / m_pumps, actual_pump_speed
    )

    pump_dp = calculate_pump_dp(
        actual_pump_speed,
        total_flow_rate,
        m_pumps,
        dp_max=dp_max,
        q_max=q_max,
        speed_max=speed_max,
        exponent=exponent,
    )

    pump_fluid_power = calculate_pump_fluid_power(total_flow_rate, pump_dp)

    pump_and_drive_power = pump_fluid_power / pump_and_drive_efficiency

    return cas.Function(
        "calculate_pump_and_drive_power",
        [valve_positions, pump_speed_scaled],
        [pump_and_drive_power],
        ["valve_positions", "pump_speed_scaled"],
        ["pump_and_drive_power"],
    )


# =============================================================================
# OIL EXIT TEMPERATURE CALCULATIONS
# =============================================================================


def solar_heat_input(
    solar_rate,
    loop_thermal_efficiency,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    d_out=COLLECTOR_D_OUT,
    collector_length=COLLECTOR_LENGTH,
    pi=cas.pi,
):
    """Calculate solar heat input (kW)

    The solar energy is concentrated by the parabolic mirrors and absorbed
    on half of the outer pipe surface area (π * d_out * collector_length / 2).
    """
    Q_solar = (
        solar_rate
        * mirror_concentration_factor
        * loop_thermal_efficiency
        * collector_length
        * pi
        * d_out
        / (2 * 1000)
    )
    return Q_solar


def calculate_collector_oil_exit_and_mean_temps_no_loss(
    flow_rate,
    inlet_temp,
    solar_rate,
    loop_thermal_efficiency,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    oil_rho_cp=OIL_RHO_CP,
    d_out=COLLECTOR_D_OUT,
    collector_length=COLLECTOR_LENGTH,
    pi=cas.pi,
):
    """Calculate oil exit temperature for a collector loop assuming no heat
    losses to ambient (i.e. h_outer = 0).

    With no heat losses, the energy balance simplifies to:
    - All absorbed solar energy goes into heating the oil
    - Exit temperature = inlet temperature + temperature rise from solar
      heating

    Parameters
    ----------
    flow_rate : float
        Oil flow rate (m³/h)
    inlet_temp : float
        Oil inlet temperature (°C)
    solar_rate : float
        Solar irradiation rate (W/m²)
    loop_thermal_efficiency : float
        Thermal efficiency of collector loop (dimensionless)
    mirror_concentration_factor : float, optional
        Mirror concentration factor (dimensionless), default:
        MIRROR_CONCENTRATION_FACTOR
    oil_rho_cp : float, optional
        Oil volumetric heat capacity (kJ/m³-K), default: OIL_RHO_CP
    d_out : float, optional
        Outer diameter of collector pipe (m), default: COLLECTOR_D_OUT
    collector_length : float, optional
        Length of collector (m), default: COLLECTOR_LENGTH
    pi : function, optional
        Provide an alternate value/function for pi, default: cas.pi

    Returns
    -------
    exit_temp : float
        Oil exit temperature (°C)
    mean_temp : float
        Mean oil temperature (°C)
    """
    # Solar heat input (kW)
    Q_solar = solar_heat_input(
        solar_rate,
        loop_thermal_efficiency,
        mirror_concentration_factor=mirror_concentration_factor,
        d_out=d_out,
        collector_length=collector_length,
        pi=pi,
    )

    # Temperature rise from solar heating (°C)
    delta_T = Q_solar / ((flow_rate / 3600) * oil_rho_cp)

    # Exit temperature
    exit_temp = inlet_temp + delta_T

    # Mean temperature (simple average for uniform heating)
    mean_temp = (inlet_temp + exit_temp) / 2

    return exit_temp, mean_temp


def calculate_collector_oil_exit_temp(
    flow_rate,
    inlet_temp,
    ambient_temp,
    solar_rate,
    loop_thermal_efficiency,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    h_outer=H_OUTER,
    oil_rho_cp=OIL_RHO_CP,
    d_out=COLLECTOR_D_OUT,
    collector_length=COLLECTOR_LENGTH,
    exp=cas.exp,
    pi=cas.pi,
):
    """Calculate oil exit temperature for a collector loop."""
    equil_temp = (
        solar_rate
        * mirror_concentration_factor
        * loop_thermal_efficiency
        / (2 * 1000 * h_outer)
    ) + ambient_temp
    tau = (flow_rate / 3600) * oil_rho_cp / pi / h_outer / d_out
    alpha = -collector_length / tau
    exit_temp = equil_temp - (equil_temp - inlet_temp) * exp(alpha)
    return exit_temp


def calculate_collector_oil_exit_and_mean_temps(
    flow_rate,
    inlet_temp,
    ambient_temp,
    solar_rate,
    loop_thermal_efficiency,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    h_outer=H_OUTER,
    oil_rho_cp=OIL_RHO_CP,
    d_out=COLLECTOR_D_OUT,
    collector_length=COLLECTOR_LENGTH,
    exp=cas.exp,
    pi=cas.pi,
):
    """Calculate oil exit temperature for a collector loop."""
    equil_temp = (
        solar_rate
        * mirror_concentration_factor
        * loop_thermal_efficiency
        / (2 * 1000 * h_outer)
    ) + ambient_temp
    tau = flow_rate * oil_rho_cp / (3600 * h_outer * pi * d_out)
    alpha = collector_length / tau
    exp_m_alpha = exp(-alpha)
    exit_temp = equil_temp - (equil_temp - inlet_temp) * exp_m_alpha
    f = (alpha - 1 + exp_m_alpha) / (alpha * (1 - exp_m_alpha))
    mean_temp = inlet_temp + f * (exit_temp - inlet_temp)
    return exit_temp, mean_temp


def calculate_mixed_oil_exit_temp(oil_exit_temps, oil_flow_rates, sum=cas.sum):
    """
    Excel AJ24: =SUM(AJ8:AJ22)/AD23
    """
    mixed_oil_exit_temp = sum(oil_exit_temps * oil_flow_rates) / sum(
        oil_flow_rates
    )
    return mixed_oil_exit_temp


def calculate_rms_oil_exit_temps(
    oil_exit_temps, oil_exit_temps_sp, sqrt=cas.sqrt, sumsqr=cas.sumsqr
):
    """Calculate RMS deviation of oil exit temperatures from setpoints."""
    N = oil_exit_temps.shape[0]
    return sqrt(sumsqr(oil_exit_temps_sp - oil_exit_temps) / N)


# =============================================================================
# COMBINED SYSTEM MODEL
# =============================================================================


def make_calculate_collector_exit_temps_and_pump_power(
    n_lines,
    m_pumps,
    rangeability=COLLECTOR_VALVE_RANGEABILITY,
    g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
    alpha=COLLECTOR_VALVE_ALPHA,
    pump_speed_min=PUMP_SPEED_MIN,
    pump_speed_max=PUMP_SPEED_MAX,
    cv=COLLECTOR_VALVE_CV,
    loop_thermal_efficiencies=LOOP_THERMAL_EFFICIENCIES,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    h_outer=H_OUTER,
    oil_rho_cp=OIL_RHO_CP,
    d_out=COLLECTOR_D_OUT,
    sum=cas.sum1,
    sqrt=cas.sqrt,
):
    """Create a CasADi function for complete system calculation."""
    valve_positions = cas.SX.sym("v", n_lines)
    pump_speed_scaled = cas.SX.sym("pump_speed_scaled")
    oil_return_temp = cas.SX.sym("oil_return_temp")
    ambient_temp = cas.SX.sym("ambient_temp")
    solar_rate = cas.SX.sym("solar_rate")
    loop_dp = cas.SX.sym("loop_dp")

    pressure_balance_function = make_pressure_balance_function(
        n_lines,
        m_pumps,
        rangeability=rangeability,
        g_squiggle=g_squiggle,
        alpha=alpha,
        cv=cv,
        sum=sum,
        sqrt=sqrt,
    )

    residual = pressure_balance_function(
        valve_positions, pump_speed_scaled, loop_dp
    )

    # Make rootfinder to solve pressure balance
    x = loop_dp
    p = cas.vertcat(valve_positions, pump_speed_scaled)
    rf = cas.rootfinder("RF", "newton", {"x": x, "p": p, "g": residual})

    # Root finder solution
    sol_rf = rf(x0=[2.0], p=p)
    loop_dp_sol = sol_rf["x"]

    collector_flow_rates = calculate_collector_flow_rate(
        valve_positions,
        loop_dp_sol,
        rangeability=COLLECTOR_VALVE_RANGEABILITY,
        g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
        alpha=COLLECTOR_VALVE_ALPHA,
        cv=COLLECTOR_VALVE_CV,
        sqrt=sqrt,
    )

    total_flow_rate = cas.sum(collector_flow_rates)

    actual_pump_speed = actual_pump_speed_from_scaled(
        pump_speed_scaled, speed_min=pump_speed_min, speed_max=pump_speed_max
    )

    pump_and_drive_efficiency = calculate_pump_and_drive_efficiency(
        total_flow_rate / m_pumps, actual_pump_speed
    )

    pump_dp = calculate_pump_dp(
        actual_pump_speed,
        total_flow_rate,
        m_pumps,
        dp_max=PUMP_DP_MAX,
        q_max=PUMP_QMAX,
        speed_max=PUMP_SPEED_MAX,
        exponent=PUMP_EXPONENT,
    )

    pump_fluid_power = calculate_pump_fluid_power(total_flow_rate, pump_dp)

    pump_and_drive_power = pump_fluid_power / pump_and_drive_efficiency

    oil_exit_temps = calculate_collector_oil_exit_temp(
        collector_flow_rates,
        oil_return_temp,
        ambient_temp,
        solar_rate,
        loop_thermal_efficiencies,
        mirror_concentration_factor=mirror_concentration_factor,
        h_outer=h_outer,
        oil_rho_cp=oil_rho_cp,
        d_out=d_out,
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


# =============================================================================
# STEAM GENERATOR MODEL
# =============================================================================


def calculate_dtlm_hx1(
    T1,
    mixed_oil_exit_temp,
    T_steam_sp=BOILER_T_STEAM_SP,
    T_boil=BOILER_T_BOIL,
    log=cas.log,
):
    """
    Calculate log mean temperature difference, HX1
    Excel BM31: =(($AJ$25-$BP$1)-(BH31-$BP$2))/LN(($AJ$25-$BP$1)/(BH31-$BP$2))
    """
    dtlm = ((mixed_oil_exit_temp - T_steam_sp) - (T1 - T_boil)) / log(
        (mixed_oil_exit_temp - T_steam_sp) / (T1 - T_boil)
    )
    return dtlm


def calculate_dtlm_hx2(
    T1,
    T2,
    T_boil=BOILER_T_BOIL,
    log=cas.log,
):
    """
    Calculate log mean temperature difference, HX2
    Excel BL31: =(BH31-BI31)/LN((BH31-$BP$2)/(BI31-$BP$2))
    """
    dtlm = (T1 - T2) / log((T1 - T_boil) / (T2 - T_boil))
    return dtlm


def calculate_dtlm_hx3(
    T2,
    Tr,
    T_condensate=BOILER_T_CONDENSATE,
    T_boil=BOILER_T_BOIL,
    log=cas.log,
):
    """
    Calculate log mean temperature difference, HX3
    Excel BK31: =((BI31-$BP$2)-(BJ31-$BP$3)/LN((BI31-$BP$2)/(BJ31-$BP$3)))
    """
    dtlm = (T2 - T_boil) - (Tr - T_condensate) / log(
        (T2 - T_boil) / (Tr - T_condensate)
    )
    return dtlm


def calculate_actual_heat_transfer_coefficient(
    total_flow_rate, U, F_oil_nominal=F_OIL_NOMINAL
):
    """
    Excel BL2: =BT2/((BT4/($AD$23/3600))^0.8)
    Excel BL3: =BT3/((BT4/($AD$23/3600))^0.8)
    Excel BL4: =BX2/((BT4/($AD$23/3600))^0.8)
    """
    U_actual = U / ((F_oil_nominal * 3600 / total_flow_rate) ** 0.8)
    return U_actual


def calculate_Q_dot_hx1(
    m_dot,
    cp_steam=CP_STEAM,
    T_steam_sp=BOILER_T_STEAM_SP,
    T_boil=BOILER_T_BOIL,
):
    """Calculate heat transfer rate in HX1 (superheating steam).

    Args:
        m_dot: Mass flow rate of water/steam (kg/s)
        cp_steam: Specific heat capacity of steam (kJ/kg-K)
        T_steam_sp: Steam setpoint temperature (°C)
        T_boil: Boiling temperature (°C)

    Returns:
        Heat transfer rate (kW)
    """
    Q_dot = m_dot * cp_steam * (T_steam_sp - T_boil)
    return Q_dot


def calculate_Q_dot_hx2(m_dot, h_vap=H_VAP):
    """Calculate heat transfer rate in HX2 (boiling water to steam).

    Args:
        m_dot: Mass flow rate of water/steam (kg/s)
        h_vap: Heat of vaporization (kJ/kg)

    Returns:
        Heat transfer rate (kW)
    """
    Q_dot = m_dot * h_vap
    return Q_dot


def calculate_Q_dot_hx3(
    m_dot, cp_water=CP_WATER, T_boil=BOILER_T_BOIL, T_condensate=T_CONDENSATE
):
    """Calculate heat transfer rate in HX3 (preheating water).

    Args:
        m_dot: Mass flow rate of water/steam (kg/s)
        cp_water: Specific heat capacity of water (kJ/kg-K)
        T_boil: Boiling temperature (°C)
        T_condensate: Condensate temperature (°C)

    Returns:
        Heat transfer rate (kW)
    """
    Q_dot = m_dot * cp_water * (T_boil - T_condensate)
    return Q_dot


def calculate_hx_area(Q_dot, dtlm, U):
    """Calculate required heat exchanger area.

    Args:
        Q_dot: Heat transfer rate (kW)
        dtlm: Log mean temperature difference (°C)
        U: Heat transfer coefficient (W/m^2-K)

    Returns:
        Required heat exchanger area (m^2)
    """
    return Q_dot / (dtlm * U)


def calculate_hx_temperatures(
    m_dot,
    mixed_oil_exit_temp,
    oil_flow_rate,
    cp_steam=CP_STEAM,
    T_steam_sp=BOILER_T_STEAM_SP,
    T_boil=BOILER_T_BOIL,
    cp_water=CP_WATER,
    T_condensate=BOILER_T_CONDENSATE,
    h_vap=H_VAP,
    oil_rho_cp=OIL_RHO_CP,
):
    """Calculate all three intermediate oil temperatures in heat exchangers.

    Steam boiler consists of 3 heat exchangers:
    1. HX1: Superheating steam from boiling to setpoint
    2. HX2: Boiling water to steam
    3. HX3: Preheating water from condensate to boiling

    Args:
        m_dot: Mass flow rate of water/steam (kg/s)
        mixed_oil_exit_temp: Mixed oil exit temperature from collectors (°C)
        oil_flow_rate: Thermal oil flow rate (kg/s)
        cp_steam: Specific heat capacity of steam (kJ/kg-K)
        T_steam_sp: Steam setpoint temperature (°C)
        T_boil: Boiling temperature (°C)
        cp_water: Specific heat capacity of water (kJ/kg-K)
        T_condensate: Condensate temperature (°C)
        h_vap: Heat of vaporization (kJ/kg)
        oil_rho_cp: Oil volumetric heat capacity (kJ/m^3-K)

    Returns:
        tuple: (T1, T2, Tr) - Oil temperatures after HX1, HX2, and HX3 (°C)
            T1: Oil temperature after HX1 (Excel BH31)
            T2: Oil temperature after HX2 (Excel BI31)
            Tr: Oil return temperature after HX3 (Excel BJ31)
    """
    # T1: Oil temperature after HX1 (superheating)
    # Excel BH31: =$AJ$25-BG31*$BG$3*($BP$1-$BP$2)/(($AD$23/3600)*$AO$4)
    T1 = mixed_oil_exit_temp - m_dot * cp_steam * (T_steam_sp - T_boil) / (
        (oil_flow_rate / 3600.0) * oil_rho_cp
    )

    # T2: Oil temperature after HX2 (boiling)
    # Excel BI31: =BH31-BG31*$AO$5/(($AD$23/3600)*$AO$4)
    T2 = T1 - m_dot * h_vap / ((oil_flow_rate / 3600.0) * oil_rho_cp)

    # Tr: Oil return temperature after HX3 (preheating)
    # Excel BJ31: =BI31-BG31*($BG$4*($BP$2-$BP$3))/(($AD$23/3600)*$AO$4)
    Tr = T2 - m_dot * (cp_water * (T_boil - T_condensate)) / (
        (oil_flow_rate / 3600.0) * oil_rho_cp
    )

    return T1, T2, Tr


def heat_exchanger_solution_error(
    T1,
    T2,
    Tr,
    m_dot,
    oil_flow_rate,
    mixed_oil_exit_temp,
    T_steam_sp=BOILER_T_STEAM_SP,
    U_steam=HX3_U_STEAM,
    U_boil=HX2_U_BOIL,
    U_liquid=HX1_U_LIQUID,
    hx_area=HX_AREA,
    F_oil_nominal=F_OIL_NOMINAL,
    cp_water=CP_WATER,
    T_boil=BOILER_T_BOIL,
    T_condensate=BOILER_T_CONDENSATE,
    cp_steam=CP_STEAM,
    h_vap=H_VAP,
    log=cas.log,
):
    """Calculate area constraint error for steam boiler heat exchangers.

    Steam boiler consists of 3 heat exchangers:
    1. HX1: Superheating steam from boiling to setpoint
    2. HX2: Boiling water to steam
    3. HX3: Preheating water from condensate to boiling

    The intermediate oil temperatures (T1, T2, Tr) are now passed as arguments
    rather than being calculated internally.

    WARNING: This function is difficult to solve with a rootfinder due to NaN
    regions. The log mean temperature difference (DTLM) calculations contain
    logarithms that become NaN when temperature differences are non-positive.

    CONSTRAINTS TO AVOID NaN VALUES:
    The following constraints must be satisfied to avoid NaN in the DTLM
    calculations:

    For HX1 (calculate_dtlm_hx1):
        - mixed_oil_exit_temp > T_steam_sp (default: > 385°C)
        - T1 > T_boil (default: > 310°C)

    For HX2 (calculate_dtlm_hx2):
        - T1 > T2
        - T1 > T_boil (default: > 310°C)
        - T2 > T_boil (default: > 310°C)  <-- may be violated at high m_dot

    For HX3 (calculate_dtlm_hx3):
        - T2 > T_boil (default: > 310°C)
        - Tr > T_condensate (default: > 60°C)

    Args:
        T1: Oil temperature after HX1 (°C)
        T2: Oil temperature after HX2 (°C)
        Tr: Oil return temperature after HX3 (°C)
        m_dot: Mass flow rate of water/steam (kg/s)
        oil_flow_rate: Thermal oil flow rate (kg/s)
        mixed_oil_exit_temp: Mixed oil exit temperature from collectors (°C)
        T_steam_sp: Steam setpoint temperature (°C)
        U_steam: Nominal heat transfer coefficient for steam (W/m^2-K)
        U_boil: Nominal heat transfer coefficient for boiling (W/m^2-K)
        U_liquid: Nominal heat transfer coefficient for liquid (W/m^2-K)
        hx_area: Total available heat exchanger area (m^2)
        F_oil_nominal: Nominal oil flow rate for U coefficient (m^3/s)
        cp_water: Specific heat capacity of water (kJ/kg-K)
        T_boil: Boiling temperature (°C)
        T_condensate: Condensate temperature (°C)
        cp_steam: Specific heat capacity of steam (kJ/kg-K)
        h_vap: Heat of vaporization (kJ/kg)
        log: Provide an alternate function for log operations

    Returns:
        Error value (zero when sum of areas equals total available area)
    """

    # Calculate log mean temperature differences
    dtlm_hx1 = calculate_dtlm_hx1(
        T1,
        mixed_oil_exit_temp,
        T_steam_sp=T_steam_sp,
        T_boil=T_boil,
        log=log,
    )

    dtlm_hx2 = calculate_dtlm_hx2(
        T1,
        T2,
        T_boil=T_boil,
        log=log,
    )

    dtlm_hx3 = calculate_dtlm_hx3(
        T2,
        Tr,
        T_condensate=T_condensate,
        T_boil=T_boil,
        log=log,
    )

    # Calculate heat transfer rates
    Q_dot_hx1 = calculate_Q_dot_hx1(
        m_dot, cp_steam=cp_steam, T_steam_sp=T_steam_sp, T_boil=T_boil
    )
    Q_dot_hx2 = calculate_Q_dot_hx2(m_dot, h_vap=h_vap)
    Q_dot_hx3 = calculate_Q_dot_hx3(
        m_dot,
        cp_water=cp_water,
        T_boil=T_boil,
        T_condensate=T_condensate,
    )

    # Calculate actual heat transfer coefficients
    U_steam_actual = calculate_actual_heat_transfer_coefficient(
        oil_flow_rate, U_steam, F_oil_nominal=F_oil_nominal
    )
    U_boil_actual = calculate_actual_heat_transfer_coefficient(
        oil_flow_rate, U_boil, F_oil_nominal=F_oil_nominal
    )
    U_liquid_actual = calculate_actual_heat_transfer_coefficient(
        oil_flow_rate, U_liquid, F_oil_nominal=F_oil_nominal
    )

    # Required area for each heat exchanger section
    hx1_area = calculate_hx_area(Q_dot_hx1, dtlm_hx1, U_steam_actual)
    hx2_area = calculate_hx_area(Q_dot_hx2, dtlm_hx2, U_boil_actual)
    hx3_area = calculate_hx_area(Q_dot_hx3, dtlm_hx3, U_liquid_actual)

    # Error is zero when sum of areas equals total available area
    return hx_area - hx1_area - hx2_area - hx3_area


def calculate_steam_power(m_dot, turbine_delta_h=TURBINE_DELTA_H):
    """Calculate steam power (kW).
    Excel BI33.
    """
    return m_dot * turbine_delta_h


def calculate_net_power(
    steam_power,
    pump_and_drive_power,
    generator_efficiency=GENERATOR_EFFICIENCY,
):
    """Calculate net power output from steam power and pump power.
    Excel BI37: =BI33*BI34-BI35/BI36
    """
    return steam_power * generator_efficiency - pump_and_drive_power


# =============================================================================
# RTO SOLVERS
# =============================================================================


def solar_plant_gen_rto_solve(
    ambient_temp,
    solar_rate,
    n_lines,
    m_pumps,
    valve_positions_init=0.9,
    pump_speed_scaled_init=0.3,
    m_dot_init=0.75,
    oil_return_temp_init=260.0,
    rangeability=COLLECTOR_VALVE_RANGEABILITY,
    g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
    alpha=COLLECTOR_VALVE_ALPHA,
    pump_speed_min=PUMP_SPEED_MIN,
    pump_speed_max=PUMP_SPEED_MAX,
    max_oil_exit_temps=OIL_EXIT_TEMPS_SP,
    cv=COLLECTOR_VALVE_CV,
    loop_thermal_efficiencies=LOOP_THERMAL_EFFICIENCIES,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    h_outer=H_OUTER,
    oil_rho_cp=OIL_RHO_CP,
    d_out=COLLECTOR_D_OUT,
    T_steam_sp=BOILER_T_STEAM_SP,
    U_steam=HX3_U_STEAM,
    U_boil=HX2_U_BOIL,
    U_liquid=HX1_U_LIQUID,
    hx_area=HX_AREA,
    F_oil_nominal=F_OIL_NOMINAL,
    cp_water=CP_WATER,
    T_boil=BOILER_T_BOIL,
    T_condensate=BOILER_T_CONDENSATE,
    turbine_delta_h=TURBINE_DELTA_H,
    cp_steam=CP_STEAM,
    h_vap=H_VAP,
    solver_name="ipopt",
    solver_opts=None,
    sum=cas.sum1,
    sqrt=cas.sqrt,
):
    """Solve the combined solar plant and steam generator RTO optimization
    problem.

    Optimizes valve positions, pump speed, oil return temperature, and steam
    mass flow rate to maximize net power while satisfying heat exchanger area
    and temperature constraints.

    This function integrates the collector field optimization with the steam
    generator, solving for all decision variables simultaneously.

    Parameters
    ----------
    ambient_temp : float
        Ambient temperature (°C)
    solar_rate : float
        Solar irradiation rate (W/m^2)
    n_lines : int
        Number of collector lines
    m_pumps : int
        Number of pumps operating
    valve_positions_init : float or array, optional
        Initial valve positions (default: 0.75)
    pump_speed_scaled_init : float, optional
        Initial scaled pump speed (default: 0.5)
    m_dot_init : float, optional
        Initial guess for steam mass flow rate (kg/s) (default: 1.2)
    oil_return_temp_init : float, optional
        Initial guess for oil return temperature (°C) (default: 270.0)
    rangeability : float, optional
        Valve rangeability parameter
    g_squiggle : float, optional
        Valve flow characteristic parameter
    alpha : float, optional
        Valve alpha parameter
    pump_speed_min : float, optional
        Minimum pump speed (RPM)
    pump_speed_max : float, optional
        Maximum pump speed (RPM)
    max_oil_exit_temps : float or array, optional
        Maximum oil exit temperatures for each line (°C)
    cv : float, optional
        Valve flow coefficient
    loop_thermal_efficiencies : list or array, optional
        Thermal efficiency for each collector loop
    mirror_concentration_factor : float, optional
        Mirror concentration factor
    h_outer : float, optional
        Outer heat transfer coefficient (kW/m^2-K)
    oil_rho_cp : float, optional
        Oil volumetric heat capacity (kJ/m^3-K)
    d_out : float, optional
        Outer diameter of collector tube (m)
    T_steam_sp : float, optional
        Steam setpoint temperature (°C)
    U_steam : float, optional
        Nominal heat transfer coefficient for steam (W/m^2-K)
    U_boil : float, optional
        Nominal heat transfer coefficient for boiling (W/m^2-K)
    U_liquid : float, optional
        Nominal heat transfer coefficient for liquid (W/m^2-K)
    hx_area : float, optional
        Total available heat exchanger area (m^2)
    F_oil_nominal : float, optional
        Nominal oil flow rate for U coefficient (m^3/s)
    cp_water : float, optional
        Specific heat capacity of water (kJ/kg-K)
    T_boil : float, optional
        Boiling temperature (°C)
    T_condensate : float, optional
        Condensate temperature (°C)
    turbine_delta_h : float, optional
        Turbine enthalpy drop (kJ/kg)
    cp_steam : float, optional
        Specific heat capacity of steam (kJ/kg-K)
    h_vap : float, optional
        Heat of vaporization (kJ/kg)
    solver_name : str, optional
        Name of the optimizer (default: 'ipopt')
    solver_opts : dict, optional
        Solver options dictionary
    sum : function, optional
        Provide an alternate function for sum operations
    sqrt : function, optional
        Provide an alternate function for sqrt operations

    Returns
    -------
    sol : OptiSol
        CasADi optimization solution object
    variables : dict
        Dictionary containing optimized variables and outputs including:
        - valve_positions: Optimal valve positions
        - pump_speed_scaled: Optimal scaled pump speed
        - collector_flow_rates: Flow rates for each collector line (kg/s)
        - oil_exit_temps: Oil exit temperatures for each collector line (°C)
        - oil_return_temp: Optimal oil return temperature (°C)
        - m_dot: Optimal steam mass flow rate (kg/s)
        - T1, T2, Tr: Oil temperatures through heat exchangers (°C)
        - pump_and_drive_power: Pump and drive power (kW)
        - steam_power: Steam power output (kW)
        - net_power: Net power output (kW)
        - hx_area_error: Heat exchanger area constraint residual
    """
    if solver_opts is None:
        solver_opts = {}

    # Construct function to calculate collector exit temps and pump power
    calculate_collector_exit_temps_and_pump_power = (
        make_calculate_collector_exit_temps_and_pump_power(
            n_lines,
            m_pumps,
            rangeability=rangeability,
            g_squiggle=g_squiggle,
            alpha=alpha,
            pump_speed_min=pump_speed_min,
            pump_speed_max=pump_speed_max,
            cv=cv,
            loop_thermal_efficiencies=loop_thermal_efficiencies,
            mirror_concentration_factor=mirror_concentration_factor,
            h_outer=h_outer,
            oil_rho_cp=oil_rho_cp,
            d_out=d_out,
            sum=sum,
            sqrt=sqrt,
        )
    )

    # Initialize optimization session
    opti = cas.Opti()

    # Decision variables
    valve_positions = opti.variable(n_lines)
    pump_speed_scaled = opti.variable()
    oil_return_temp = opti.variable()
    m_dot = opti.variable()

    collector_flow_rates, pump_and_drive_power, oil_exit_temps = (
        calculate_collector_exit_temps_and_pump_power(
            valve_positions,
            pump_speed_scaled,
            oil_return_temp,
            ambient_temp,
            solar_rate,
        )
    )

    mixed_oil_exit_temp = calculate_mixed_oil_exit_temp(
        oil_exit_temps, collector_flow_rates
    )
    oil_flow_rate = cas.sum(collector_flow_rates)

    # Calculate intermediate temperatures
    T1, T2, Tr = calculate_hx_temperatures(
        m_dot,
        mixed_oil_exit_temp,
        oil_flow_rate,
        cp_steam=cp_steam,
        T_steam_sp=T_steam_sp,
        T_boil=T_boil,
        cp_water=cp_water,
        T_condensate=T_condensate,
        h_vap=h_vap,
        oil_rho_cp=oil_rho_cp,
    )

    # Calculate heat exchanger area constraint error
    hx_area_error = heat_exchanger_solution_error(
        T1,
        T2,
        Tr,
        m_dot,
        oil_flow_rate,
        mixed_oil_exit_temp,
        T_steam_sp=T_steam_sp,
        U_steam=U_steam,
        U_boil=U_boil,
        U_liquid=U_liquid,
        hx_area=hx_area,
        F_oil_nominal=F_oil_nominal,
        cp_water=cp_water,
        T_boil=T_boil,
        T_condensate=T_condensate,
        cp_steam=cp_steam,
        h_vap=h_vap,
        log=cas.log,
    )

    # Calculate other output variables
    steam_power = calculate_steam_power(m_dot, turbine_delta_h=turbine_delta_h)
    net_power = calculate_net_power(steam_power, pump_and_drive_power)

    # Collector and pump constraints
    opti.subject_to(opti.bounded(0.1, valve_positions, 1.0))
    opti.subject_to(opti.bounded(0.2, pump_speed_scaled, 1.0))
    opti.subject_to(oil_exit_temps < max_oil_exit_temps)

    # Temperature constraints to avoid NaN in DTLM calculations
    opti.subject_to(mixed_oil_exit_temp > T_steam_sp)  # For HX1
    opti.subject_to(T1 > T_boil)  # For HX1 and HX2
    opti.subject_to(T2 > T_boil)  # For HX2 and HX3 (critical constraint)
    opti.subject_to(Tr > T_condensate)  # For HX3
    opti.subject_to(T1 > T2)  # For HX2

    # Heat exchanger area constraint
    opti.subject_to(hx_area_error == 0)
    opti.subject_to(oil_return_temp == Tr)

    # Cost function - maximize net power generation
    opti.minimize(-net_power)

    # Set initial values
    opti.set_initial(m_dot, m_dot_init)
    opti.set_initial(valve_positions, valve_positions_init)
    opti.set_initial(pump_speed_scaled, pump_speed_scaled_init)
    opti.set_initial(oil_return_temp, oil_return_temp_init)

    # Solver options
    opti.solver(solver_name, solver_opts)
    sol = opti.solve()

    variables = {
        "valve_positions": opti.value(valve_positions),
        "pump_speed_scaled": opti.value(pump_speed_scaled),
        "collector_flow_rates": opti.value(collector_flow_rates),
        "oil_exit_temps": opti.value(oil_exit_temps),
        "oil_return_temp": opti.value(oil_return_temp),
        "m_dot": opti.value(m_dot),
        "T1": opti.value(T1),
        "T2": opti.value(T2),
        "Tr": opti.value(Tr),
        "pump_and_drive_power": opti.value(pump_and_drive_power),
        "steam_power": opti.value(steam_power),
        "net_power": opti.value(net_power),
        "hx_area_error": opti.value(hx_area_error),
    }

    return sol, variables


def steam_generator_solve(
    mixed_oil_exit_temp,
    oil_flow_rate,
    m_dot_init=0.75,
    T_steam_sp=BOILER_T_STEAM_SP,
    U_steam=HX3_U_STEAM,
    U_boil=HX2_U_BOIL,
    U_liquid=HX1_U_LIQUID,
    hx_area=HX_AREA,
    F_oil_nominal=F_OIL_NOMINAL,
    cp_water=CP_WATER,
    T_boil=BOILER_T_BOIL,
    T_condensate=BOILER_T_CONDENSATE,
    turbine_delta_h=TURBINE_DELTA_H,
    cp_steam=CP_STEAM,
    oil_rho_cp=OIL_RHO_CP,
    h_vap=H_VAP,
    solver_name="ipopt",
    solver_opts=None,
):
    """Solve the steam generator optimization problem.

    Optimizes steam mass flow rate to maximize steam power while satisfying
    heat exchanger area constraints, given the mixed oil exit temperature and
    oil flow rate from the collector field.

    Parameters
    ----------
    mixed_oil_exit_temp : float
        Mixed oil exit temperature from collectors (°C)
    oil_flow_rate : float
        Total oil flow rate (kg/s)
    m_dot_init : float, optional
        Initial guess for steam mass flow rate (kg/s) (default: 1.2)
    T_steam_sp : float, optional
        Steam setpoint temperature (°C)
    U_steam : float, optional
        Nominal heat transfer coefficient for steam (W/m^2-K)
    U_boil : float, optional
        Nominal heat transfer coefficient for boiling (W/m^2-K)
    U_liquid : float, optional
        Nominal heat transfer coefficient for liquid (W/m^2-K)
    hx_area : float, optional
        Total available heat exchanger area (m^2)
    F_oil_nominal : float, optional
        Nominal oil flow rate for U coefficient (m^3/s)
    cp_water : float, optional
        Specific heat capacity of water (kJ/kg-K)
    T_boil : float, optional
        Boiling temperature (°C)
    T_condensate : float, optional
        Condensate temperature (°C)
    turbine_delta_h : float, optional
        Turbine enthalpy drop (kJ/kg)
    cp_steam : float, optional
        Specific heat capacity of steam (kJ/kg-K)
    oil_rho_cp : float, optional
        Oil volumetric heat capacity (kJ/m^3-K)
    h_vap : float, optional
        Heat of vaporization (kJ/kg)
    solver_name : str, optional
        Name of the optimizer (default: 'ipopt')
    solver_opts : dict, optional
        Solver options dictionary

    Returns
    -------
    sol : OptiSol
        CasADi optimization solution object
    variables : dict
        Dictionary containing optimized variables and outputs including:
        - m_dot: Optimal steam mass flow rate (kg/s)
        - T1, T2, Tr: Oil temperatures through heat exchangers (°C)
        - steam_power: Steam power output (kW)
        - hx_area_error: Heat exchanger area constraint residual
    """
    if solver_opts is None:
        solver_opts = {}

    # Initialize optimization session
    opti = cas.Opti()

    # Decision variable
    m_dot = opti.variable()

    # Calculate intermediate temperatures
    T1, T2, Tr = calculate_hx_temperatures(
        m_dot,
        mixed_oil_exit_temp,
        oil_flow_rate,
        cp_steam=cp_steam,
        T_steam_sp=T_steam_sp,
        T_boil=T_boil,
        cp_water=cp_water,
        T_condensate=T_condensate,
        h_vap=h_vap,
        oil_rho_cp=oil_rho_cp,
    )

    # Calculate heat exchanger area constraint error
    hx_area_error = heat_exchanger_solution_error(
        T1,
        T2,
        Tr,
        m_dot,
        oil_flow_rate,
        mixed_oil_exit_temp,
        T_steam_sp=T_steam_sp,
        U_steam=U_steam,
        U_boil=U_boil,
        U_liquid=U_liquid,
        hx_area=hx_area,
        F_oil_nominal=F_oil_nominal,
        cp_water=cp_water,
        T_boil=T_boil,
        T_condensate=T_condensate,
        cp_steam=cp_steam,
        h_vap=h_vap,
        log=cas.log,
    )

    # Calculate steam power
    steam_power = calculate_steam_power(m_dot, turbine_delta_h=turbine_delta_h)

    # Temperature constraints to avoid NaN in DTLM calculations
    if not mixed_oil_exit_temp > T_steam_sp:
        raise ValueError(
            "mixed_oil_exit_temp must be greater than T_steam_sp."
        )
    opti.subject_to(T1 > T_boil)  # For HX1 and HX2
    opti.subject_to(T2 > T_boil)  # For HX2 and HX3 (critical constraint)
    opti.subject_to(Tr > T_condensate)  # For HX3
    opti.subject_to(T1 > T2)  # For HX2

    # Heat exchanger area constraint
    opti.subject_to(hx_area_error == 0)

    # Cost function - maximize steam power generation
    opti.minimize(-steam_power)

    # Set initial value
    opti.set_initial(m_dot, m_dot_init)

    # Solver options
    opti.solver(solver_name, solver_opts)
    sol = opti.solve()

    variables = {
        "m_dot": opti.value(m_dot),
        "T1": opti.value(T1),
        "T2": opti.value(T2),
        "Tr": opti.value(Tr),
        "steam_power": opti.value(steam_power),
        "hx_area_error": opti.value(hx_area_error),
    }

    return sol, variables


def solar_plant_rto_solve(
    solar_rate,
    ambient_temp,
    oil_return_temp,
    m_pumps,
    n_lines,
    valve_positions_init=0.9,
    pump_speed_scaled_init=0.3,
    rangeability=COLLECTOR_VALVE_RANGEABILITY,
    g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
    alpha=COLLECTOR_VALVE_ALPHA,
    pump_speed_min=PUMP_SPEED_MIN,
    pump_speed_max=PUMP_SPEED_MAX,
    cv=COLLECTOR_VALVE_CV,
    loop_thermal_efficiencies=LOOP_THERMAL_EFFICIENCIES,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    h_outer=H_OUTER,
    oil_rho_cp=OIL_RHO_CP,
    d_out=COLLECTOR_D_OUT,
    max_oil_exit_temps=OIL_EXIT_TEMPS_SP,
    sum=cas.sum1,
    sqrt=cas.sqrt,
    solver_name="ipopt",
    solver_opts=None,
):
    """Solve the solar plant RTO optimization problem (collector field only).

    Maximizes net potential energy generation by optimizing valve positions
    and pump speed subject to temperature and equipment constraints. This
    function optimizes only the collector field without the steam generator.

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
    rangeability : float, optional
        Valve rangeability parameter
    g_squiggle : float, optional
        Valve flow characteristic parameter
    alpha : float, optional
        Valve alpha parameter
    pump_speed_min : float, optional
        Minimum pump speed (RPM)
    pump_speed_max : float, optional
        Maximum pump speed (RPM)
    cv : float, optional
        Valve flow coefficient
    loop_thermal_efficiencies : list or array, optional
        Thermal efficiency for each collector loop
    mirror_concentration_factor : float, optional
        Mirror concentration factor
    h_outer : float, optional
        Outer heat transfer coefficient (kW/m^2-K)
    oil_rho_cp : float, optional
        Oil volumetric heat capacity (kJ/m^3-K)
    d_out : float, optional
        Outer diameter of collector tube (m)
    max_oil_exit_temps : list or array, optional
        Maximum oil exit temperatures for each line (°C)
    sum : function, optional
        Provide an alternate function for sum operations
    sqrt : function, optional
        Provide an alternate function for sqrt operations
    solver_name : str, optional
        Name of the optimizer (default: 'ipopt')
    solver_opts : dict, optional
        Solver options dictionary

    Returns
    -------
    sol : OptiSol
        CasADi optimization solution object
    variables : dict
        Dictionary containing optimized variables and outputs including:
        - valve_positions: Optimal valve positions
        - oil_exit_temps: Oil exit temperatures for each collector line (°C)
        - collector_flow_rates: Flow rates for each collector line (kg/s)
        - pump_speed_scaled: Optimal scaled pump speed
        - pump_and_drive_power: Pump and drive power (kW)
        - potential_work: Potential work output from Carnot cycle (kW)
        - f: Objective function value
        - grad_f: Gradient of objective function
        - hess_f: Hessian of objective function
    """
    if solver_opts is None:
        solver_opts = {}

    # Initialize optimization session
    opti = cas.Opti()

    # Construct system model calculation function
    calculate_collector_exit_temps_and_pump_power = (
        make_calculate_collector_exit_temps_and_pump_power(
            n_lines,
            m_pumps,
            rangeability=rangeability,
            g_squiggle=g_squiggle,
            alpha=alpha,
            pump_speed_min=pump_speed_min,
            pump_speed_max=pump_speed_max,
            cv=cv,
            loop_thermal_efficiencies=loop_thermal_efficiencies,
            mirror_concentration_factor=mirror_concentration_factor,
            h_outer=h_outer,
            oil_rho_cp=oil_rho_cp,
            d_out=d_out,
            sum=sum,
            sqrt=sqrt,
        )
    )

    max_oil_exit_temps = cas.DM(max_oil_exit_temps)

    # Decision variables
    valve_positions = opti.variable(n_lines)
    pump_speed_scaled = opti.variable()
    x = cas.vertcat(valve_positions, pump_speed_scaled)

    # System model outputs
    collector_flow_rates, pump_and_drive_power, oil_exit_temps = (
        calculate_collector_exit_temps_and_pump_power(
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
    # TODO: Make this into a function
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
        "pump_speed_scaled": opti.value(pump_speed_scaled),
        "collector_flow_rates": opti.value(collector_flow_rates),
        "oil_exit_temps": opti.value(oil_exit_temps),
        "pump_and_drive_power": opti.value(pump_and_drive_power),
        "potential_work": opti.value(potential_work),
        "f": f,
        "grad_f": grad_f,
        "hess_f": hess_f,
    }

    return sol, variables


def solar_plant_gen_db_rto_solve(
    ambient_temp,
    solar_rate,
    n_lines,
    m_pumps,
    valve_positions_init=0.9,
    pump_speed_scaled_init=0.3,
    m_dot_init=0.75,
    oil_return_temp_init=260.0,
    rangeability=COLLECTOR_VALVE_RANGEABILITY,
    g_squiggle=COLLECTOR_VALVE_G_SQUIGGLE,
    alpha=COLLECTOR_VALVE_ALPHA,
    pump_speed_min=PUMP_SPEED_MIN,
    pump_speed_max=PUMP_SPEED_MAX,
    max_oil_exit_temps=OIL_EXIT_TEMPS_SP,
    cv=COLLECTOR_VALVE_CV,
    loop_thermal_efficiencies=LOOP_THERMAL_EFFICIENCIES,
    mirror_concentration_factor=MIRROR_CONCENTRATION_FACTOR,
    h_outer=H_OUTER,
    oil_rho_cp=OIL_RHO_CP,
    d_out=COLLECTOR_D_OUT,
    T_steam_sp=BOILER_T_STEAM_SP,
    U_steam=HX3_U_STEAM,
    U_boil=HX2_U_BOIL,
    U_liquid=HX1_U_LIQUID,
    hx_area=HX_AREA,
    F_oil_nominal=F_OIL_NOMINAL,
    cp_water=CP_WATER,
    T_boil=BOILER_T_BOIL,
    T_condensate=BOILER_T_CONDENSATE,
    turbine_delta_h=TURBINE_DELTA_H,
    cp_steam=CP_STEAM,
    h_vap=H_VAP,
    solver_name="ipopt",
    solver_opts=None,
    sum=cas.sum1,
    sqrt=cas.sqrt,
):
    """Solve the combined solar plant and steam generator RTO optimization
    problem.

    Optimizes valve positions, pump speed, oil return temperature, and steam
    mass flow rate to maximize net power while satisfying heat exchanger area
    and temperature constraints.

    This function integrates the collector field optimization with the steam
    generator, solving for all decision variables simultaneously.

    Parameters
    ----------
    ambient_temp : float
        Ambient temperature (°C)
    solar_rate : float
        Solar irradiation rate (W/m^2)
    n_lines : int
        Number of collector lines
    m_pumps : int
        Number of pumps operating
    valve_positions_init : float or array, optional
        Initial valve positions (default: 0.75)
    pump_speed_scaled_init : float, optional
        Initial scaled pump speed (default: 0.5)
    m_dot_init : float, optional
        Initial guess for steam mass flow rate (kg/s) (default: 1.2)
    oil_return_temp_init : float, optional
        Initial guess for oil return temperature (°C) (default: 270.0)
    rangeability : float, optional
        Valve rangeability parameter
    g_squiggle : float, optional
        Valve flow characteristic parameter
    alpha : float, optional
        Valve alpha parameter
    pump_speed_min : float, optional
        Minimum pump speed (RPM)
    pump_speed_max : float, optional
        Maximum pump speed (RPM)
    max_oil_exit_temps : float or array, optional
        Maximum oil exit temperatures for each line (°C)
    cv : float, optional
        Valve flow coefficient
    loop_thermal_efficiencies : list or array, optional
        Thermal efficiency for each collector loop
    mirror_concentration_factor : float, optional
        Mirror concentration factor
    h_outer : float, optional
        Outer heat transfer coefficient (kW/m^2-K)
    oil_rho_cp : float, optional
        Oil volumetric heat capacity (kJ/m^3-K)
    d_out : float, optional
        Outer diameter of collector tube (m)
    T_steam_sp : float, optional
        Steam setpoint temperature (°C)
    U_steam : float, optional
        Nominal heat transfer coefficient for steam (W/m^2-K)
    U_boil : float, optional
        Nominal heat transfer coefficient for boiling (W/m^2-K)
    U_liquid : float, optional
        Nominal heat transfer coefficient for liquid (W/m^2-K)
    hx_area : float, optional
        Total available heat exchanger area (m^2)
    F_oil_nominal : float, optional
        Nominal oil flow rate for U coefficient (m^3/s)
    cp_water : float, optional
        Specific heat capacity of water (kJ/kg-K)
    T_boil : float, optional
        Boiling temperature (°C)
    T_condensate : float, optional
        Condensate temperature (°C)
    turbine_delta_h : float, optional
        Turbine enthalpy drop (kJ/kg)
    cp_steam : float, optional
        Specific heat capacity of steam (kJ/kg-K)
    h_vap : float, optional
        Heat of vaporization (kJ/kg)
    solver_name : str, optional
        Name of the optimizer (default: 'ipopt')
    solver_opts : dict, optional
        Solver options dictionary
    sum : function, optional
        Provide an alternate function for sum operations
    sqrt : function, optional
        Provide an alternate function for sqrt operations

    Returns
    -------
    sol : OptiSol
        CasADi optimization solution object
    variables : dict
        Dictionary containing optimized variables and outputs including:
        - valve_positions: Optimal valve positions
        - pump_speed_scaled: Optimal scaled pump speed
        - collector_flow_rates: Flow rates for each collector line (kg/s)
        - oil_exit_temps: Oil exit temperatures for each collector line (°C)
        - oil_return_temp: Optimal oil return temperature (°C)
        - m_dot: Optimal steam mass flow rate (kg/s)
        - T1, T2, Tr: Oil temperatures through heat exchangers (°C)
        - pump_and_drive_power: Pump and drive power (kW)
        - steam_power: Steam power output (kW)
        - net_power: Net power output (kW)
        - hx_area_error: Heat exchanger area constraint residual
    """
    if solver_opts is None:
        solver_opts = {}

    # Construct function to calculate collector exit temps and pump power
    calculate_collector_exit_temps_and_pump_power = (
        make_calculate_collector_exit_temps_and_pump_power(
            n_lines,
            m_pumps,
            rangeability=rangeability,
            g_squiggle=g_squiggle,
            alpha=alpha,
            pump_speed_min=pump_speed_min,
            pump_speed_max=pump_speed_max,
            cv=cv,
            loop_thermal_efficiencies=loop_thermal_efficiencies,
            mirror_concentration_factor=mirror_concentration_factor,
            h_outer=h_outer,
            oil_rho_cp=oil_rho_cp,
            d_out=d_out,
            sum=sum,
            sqrt=sqrt,
        )
    )

    # Initialize optimization session
    opti = cas.Opti()

    # Decision variables
    valve_positions = opti.variable(n_lines)
    pump_speed_scaled = opti.variable()
    oil_return_temp = opti.variable()
    m_dot = opti.variable()

    collector_flow_rates, pump_and_drive_power, oil_exit_temps = (
        calculate_collector_exit_temps_and_pump_power(
            valve_positions,
            pump_speed_scaled,
            oil_return_temp,
            ambient_temp,
            solar_rate,
        )
    )

    mixed_oil_exit_temp = calculate_mixed_oil_exit_temp(
        oil_exit_temps, collector_flow_rates
    )
    oil_flow_rate = cas.sum(collector_flow_rates)

    # Calculate intermediate temperatures
    T1, T2, Tr = calculate_hx_temperatures(
        m_dot,
        mixed_oil_exit_temp,
        oil_flow_rate,
        cp_steam=cp_steam,
        T_steam_sp=T_steam_sp,
        T_boil=T_boil,
        cp_water=cp_water,
        T_condensate=T_condensate,
        h_vap=h_vap,
        oil_rho_cp=oil_rho_cp,
    )

    # Calculate heat exchanger area constraint error
    hx_area_error = heat_exchanger_solution_error(
        T1,
        T2,
        Tr,
        m_dot,
        oil_flow_rate,
        mixed_oil_exit_temp,
        T_steam_sp=T_steam_sp,
        U_steam=U_steam,
        U_boil=U_boil,
        U_liquid=U_liquid,
        hx_area=hx_area,
        F_oil_nominal=F_oil_nominal,
        cp_water=cp_water,
        T_boil=T_boil,
        T_condensate=T_condensate,
        cp_steam=cp_steam,
        h_vap=h_vap,
        log=cas.log,
    )

    # Calculate other output variables
    steam_power = calculate_steam_power(m_dot, turbine_delta_h=turbine_delta_h)
    net_power = calculate_net_power(steam_power, pump_and_drive_power)

    # Collector and pump constraints
    opti.subject_to(opti.bounded(0.1, valve_positions, 1.0))
    opti.subject_to(opti.bounded(0.2, pump_speed_scaled, 1.0))
    opti.subject_to(oil_exit_temps < max_oil_exit_temps)

    # Temperature constraints to avoid NaN in DTLM calculations
    opti.subject_to(mixed_oil_exit_temp > T_steam_sp)  # For HX1
    opti.subject_to(T1 > T_boil)  # For HX1 and HX2
    opti.subject_to(T2 > T_boil)  # For HX2 and HX3 (critical constraint)
    opti.subject_to(Tr > T_condensate)  # For HX3
    opti.subject_to(T1 > T2)  # For HX2

    # Heat exchanger area constraint
    opti.subject_to(hx_area_error == 0)
    opti.subject_to(oil_return_temp == Tr)

    # Cost function - maximize net power generation
    opti.minimize(-net_power)

    # Set initial values
    opti.set_initial(m_dot, m_dot_init)
    opti.set_initial(valve_positions, valve_positions_init)
    opti.set_initial(pump_speed_scaled, pump_speed_scaled_init)
    opti.set_initial(oil_return_temp, oil_return_temp_init)

    # Solver options
    opti.solver(solver_name, solver_opts)
    sol = opti.solve()

    variables = {
        "valve_positions": opti.value(valve_positions),
        "pump_speed_scaled": opti.value(pump_speed_scaled),
        "collector_flow_rates": opti.value(collector_flow_rates),
        "oil_exit_temps": opti.value(oil_exit_temps),
        "oil_return_temp": opti.value(oil_return_temp),
        "m_dot": opti.value(m_dot),
        "T1": opti.value(T1),
        "T2": opti.value(T2),
        "Tr": opti.value(Tr),
        "pump_and_drive_power": opti.value(pump_and_drive_power),
        "steam_power": opti.value(steam_power),
        "net_power": opti.value(net_power),
        "hx_area_error": opti.value(hx_area_error),
    }

    return sol, variables
