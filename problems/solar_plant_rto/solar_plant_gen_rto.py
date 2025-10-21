"""Solar Plant Real-Time Optimization (RTO) Module

This module contains functions for modeling and optimizing a solar thermal
plant with collector loops, power generation and thermal oil circulation.
"""

import casadi as cas

# =============================================================================
# DEFAULT CONSTANTS
# =============================================================================

# Collector lines
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
OIL_RHO_CP = 1600  # kJ/m^3-K
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


# =============================================================================
# PUMP AND FLOW CALCULATIONS
# =============================================================================


def actual_pump_speed_from_scaled(speed_scaled):
    """Convert scaled pump speed (0.2-1.0) to actual speed (rpm)."""
    return (
        PUMP_SPEED_MIN
        + (speed_scaled - 0.3) * (PUMP_SPEED_MAX - PUMP_SPEED_MIN) / 0.7
    )


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

    total_flow_rate = calculate_total_oil_flowrate(
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


# =============================================================================
# OIL EXIT TEMPERATURE CALCULATIONS
# =============================================================================


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
# TODO: Need to add steam generator to this


def make_collector_exit_temps_and_pump_power_function(n_lines, m_pumps):
    """Create a CasADi function for complete system calculation."""
    valve_positions = cas.SX.sym("v", n_lines)
    pump_speed_scaled = cas.SX.sym("pump_speed_scaled")
    oil_return_temp = cas.SX.sym("oil_return_temp")
    ambient_temp = cas.SX.sym("ambient_temp")
    solar_rate = cas.SX.sym("solar_rate")

    pressure_balance_function = make_pressure_balance_function(
        n_lines, m_pumps
    )

    # Make rootfinder to solve pressure balance
    x = cas.SX.sym("x")
    p = cas.vertcat(valve_positions, pump_speed_scaled)
    residual = pressure_balance_function(valve_positions, pump_speed_scaled, x)
    rf = cas.rootfinder("RF", "newton", {"x": x, "p": p, "g": residual})

    # Root finder solution
    sol_rf = rf(x0=[2.0], p=p)
    loop_dp = sol_rf["x"]

    collector_flow_rates = calculate_collector_flow_rate(
        valve_positions,
        loop_dp,
    )

    total_flow_rate = cas.sum(collector_flow_rates)

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
    return U / ((F_oil_nominal / (total_flow_rate / 3600)) ** 0.8)


def calculate_Q_dot_hx1(
    m_dot,
    cp_steam=CP_STEAM,
    T_steam_sp=BOILER_T_STEAM_SP,
    T_boil=BOILER_T_BOIL,
):
    Q_dot = m_dot * cp_steam * (T_steam_sp - T_boil)
    return Q_dot


def calculate_Q_dot_hx2(m_dot, h_vap=H_VAP):
    Q_dot = m_dot * h_vap
    return Q_dot


def calculate_Q_dot_hx3(
    m_dot, cp_water=CP_WATER, T_boil=BOILER_T_BOIL, T_condensate=T_CONDENSATE
):
    Q_dot = m_dot * cp_water * (T_boil - T_condensate)
    return Q_dot


def calculate_hx_area(Q_dot, dtlm, U):
    return Q_dot / (dtlm * U)


def calculate_T1(
    m_dot,
    mixed_oil_exit_temp,
    oil_flow_rate,
    cp_steam=CP_STEAM,
    T_steam_sp=BOILER_T_STEAM_SP,
    T_boil=BOILER_T_BOIL,
    oil_rho_cp=OIL_RHO_CP,
):
    """
    Excel BH31: =$AJ$25-BG31*$BG$3*($BP$1-$BP$2)/(($AD$23/3600)*$AO$4)
    """
    T1 = mixed_oil_exit_temp - m_dot * cp_steam * (T_steam_sp - T_boil) / (
        (oil_flow_rate / 3600.0) * oil_rho_cp
    )
    return T1


def calculate_T2(
    m_dot,
    T1,
    oil_flow_rate,
    h_vap=H_VAP,
    oil_rho_cp=OIL_RHO_CP,
):
    """
    Excel BI31: =BH31-BG31*$AO$5/(($AD$23/3600)*$AO$4)
    """
    T2 = T1 - m_dot * h_vap / ((oil_flow_rate / 3600.0) * oil_rho_cp)
    return T2


def calculate_oil_return_temp(
    m_dot,
    T2,
    oil_flow_rate,
    cp_water=CP_WATER,
    T_boil=BOILER_T_BOIL,
    T_condensate=BOILER_T_CONDENSATE,
    oil_rho_cp=OIL_RHO_CP,
):
    """
    Excel BJ31: =BI31-BG31*($BG$4*($BP$2-$BP$3))/(($AD$23/3600)*$AO$4)
    """
    oil_return_temp = T2 - m_dot * (cp_water * (T_boil - T_condensate)) / (
        (oil_flow_rate / 3600.0) * oil_rho_cp
    )
    return oil_return_temp


def heat_exchanger_solution_error(
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
    oil_rho_cp=OIL_RHO_CP,
    h_vap=H_VAP,
    log=cas.log,
):
    """Calculate area constraint error for steam boiler heat exchangers.

    Steam boiler consists of 3 heat exchangers:
    1. HX1: Superheating steam from boiling to setpoint
    2. HX2: Boiling water to steam
    3. HX3: Preheating water from condensate to boiling

    The function calculates the intermediate oil temperatures (T1, T2, Tr)
    internally based on the input parameters.

    WARNING: This function is difficult to solve with a rootfinder due to NaN
    regions. The log mean temperature difference (DTLM) calculations contain
    logarithms that become NaN when temperature differences are non-positive.

    CONSTRAINTS TO AVOID NaN VALUES:
    The following constraints must be satisfied to avoid NaN in the DTLM calculations:

    For HX1 (calculate_dtlm_hx1):
        - mixed_oil_exit_temp > T_steam_sp (default: > 385°C)
        - T1 > T_boil (default: > 310°C)

    For HX2 (calculate_dtlm_hx2):
        - T1 > T2
        - T1 > T_boil (default: > 310°C)
        - T2 > T_boil (default: > 310°C)  <-- CRITICAL: often violated at high m_dot

    For HX3 (calculate_dtlm_hx3):
        - T2 > T_boil (default: > 310°C)
        - Tr > T_condensate (default: > 60°C)

    The most common violation is T2 < T_boil, which occurs when m_dot is too large
    relative to the available heat transfer area. This typically happens during
    rootfinding when the solver tries values of m_dot > ~1.35 kg/s (depends on
    operating conditions).

    RECOMMENDED APPROACH:
    Instead of using a rootfinder, solve for m_dot as part of the global optimization
    problem with explicit bounds on m_dot or by adding the above constraints directly
    to the optimization formulation.

    Args:
        m_dot: Mass flow rate of water/steam (kg/s)
        oil_flow_rate: Thermal oil flow rate (m^3/h)
        mixed_oil_exit_temp: Mixed oil exit temperature from collectors (deg C)
        T_steam_sp: Steam setpoint temperature (deg C)
        U_steam: Nominal heat transfer coefficient for steam (W/m^2-K)
        U_boil: Nominal heat transfer coefficient for boiling (W/m^2-K)
        U_liquid: Nominal heat transfer coefficient for liquid (W/m^2-K)
        hx_area: Total available heat exchanger area (m^2)
        F_oil_nominal: Nominal oil flow rate for U coefficient (m^3/s)
        cp_water: Specific heat capacity of water (kJ/kg-K)
        T_boil: Boiling temperature (deg C)
        T_condensate: Condensate temperature (deg C)
        cp_steam: Specific heat capacity of steam (kJ/kg-K)
        oil_rho_cp: Oil volumetric heat capacity (kJ/m^3-K)
        h_vap: Heat of vaporization (kJ/kg)
        log: Logarithm function (for CasADi compatibility)

    Returns:
        Error value (zero when sum of areas equals total available area)
    """

    T1 = calculate_T1(
        m_dot,
        mixed_oil_exit_temp,
        oil_flow_rate,
        cp_steam=cp_steam,
        T_steam_sp=T_steam_sp,
        T_boil=T_boil,
        oil_rho_cp=oil_rho_cp,
    )

    T2 = calculate_T2(
        m_dot,
        T1,
        oil_flow_rate,
        h_vap=h_vap,
        oil_rho_cp=oil_rho_cp,
    )

    Tr = calculate_oil_return_temp(
        m_dot,
        T2,
        oil_flow_rate,
        cp_water=cp_water,
        T_boil=T_boil,
        T_condensate=T_condensate,
        oil_rho_cp=oil_rho_cp,
    )

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

    # Calculate actual
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


def calculate_steam_power(m_dot):
    """Calculate steam power (kW).
    Excel BI33.
    """
    return m_dot * (3049.0 - 2207.0)


def calculate_net_power(
    steam_power, pump_fluid_power, pump_and_drive_efficiency
):
    """Calculate net power output from steam power and pump power.
    Excel BI37: =BI33*BI34-BI35/BI36
    """
    return (
        steam_power * GENERATOR_EFFICIENCY
        - pump_fluid_power / pump_and_drive_efficiency
    )


def make_calculate_m_dot():
    """Create a CasADi function for calculating steam mass flow rate.

    WARNING: This function uses a rootfinder to solve heat_exchanger_solution_error
    for m_dot. This approach is NOT RECOMMENDED due to numerical issues - the residual
    function contains NaN regions that cause the rootfinder to fail (see detailed
    explanation in heat_exchanger_solution_error docstring).

    RECOMMENDED ALTERNATIVE:
    Instead of using this function, solve for m_dot as a decision variable in the
    global optimization problem, with constraints:
        - m_dot > 0.1 (reasonable lower bound)
        - T2 > T_boil (e.g., T2 > 310°C) to avoid NaN in DTLM calculations
        - heat_exchanger_solution_error(...) == 0 (area constraint)

    This function is kept for reference but should not be used in production code.
    """

    # Unknown
    m_dot = cas.SX.sym("m_dot")

    # Knowns
    oil_flow_rate = cas.SX.sym("oil_flow_rate")
    mixed_oil_exit_temp = cas.SX.sym("mixed_oil_exit_temp")

    # Set everything else to defaults
    residual = heat_exchanger_solution_error(
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
        oil_rho_cp=OIL_RHO_CP,
        h_vap=H_VAP,
        log=cas.log,
    )

    # Make rootfinder to solve heat exchanger area constraint
    x = m_dot
    p = cas.vertcat(oil_flow_rate, mixed_oil_exit_temp)

    # Use kinsol solver which is more robust for this type of problem
    opts = {
        'abstol': 1e-8,
        'max_iter': 100
    }
    rf = cas.rootfinder("RF", "kinsol", {"x": x, "p": p, "g": residual}, opts)

    # Root finder solution
    # Initial guess must be low enough to keep T2 > T_boil (310°C)
    sol_rf = rf(x0=[0.5], p=p)
    m_dot_sol = sol_rf["x"]

    return cas.Function(
        "calculate_m_dot",
        [oil_flow_rate, mixed_oil_exit_temp],
        [m_dot_sol],
        ["oil_flow_rate", "mixed_oil_exit_temp"],
        ["m_dot"],
    )


# =============================================================================
# RTO SOLVER
# =============================================================================


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
