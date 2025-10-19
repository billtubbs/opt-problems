"""Unit tests for solar_plant_rto module."""

import numpy as np
import casadi as cas
import pytest
from problems.solar_plant_rto.solar_plant_rto import (
    actual_pump_speed_from_scaled,
    calculate_pump_and_drive_efficiency,
    calculate_pump_fluid_power,
    calculate_collector_flow_rate,
    calculate_total_flowrate,
    calculate_boiler_dp,
    calculate_pump_dp,
    calculate_pressure_balance,
    calculate_collector_oil_exit_temp,
    calculate_rms_oil_exit_temps,
    calculate_net_power,
    make_pressure_balance_function,
    make_calculate_pump_and_drive_power_function,
    make_collector_exit_temps_and_pump_power_function,
    PUMP_SPEED_MIN,
    PUMP_SPEED_MAX,
)


# Test data from Excel spreadsheet
test_data = {
    "pump_speed_scaled": 0.75346732613277,
    "actual_pump_speed": 2362.913291,
    "pump_dp": 629.0871255,
    "boiler_dp": 592.0326755,
    "total_flow_rate": 124.8190981,
    "pump_fluid_power": 21.81169101,
    "valve_positions": np.array([
        1.000000000, 0.936959219, 0.896931101, 0.865584094, 0.836992607,
        0.812643981, 0.789969825, 0.769361992, 0.93925857, 0.898066194,
        0.864955164, 0.837553205, 0.812789834, 0.789567955, 0.769638106
    ]),
    "loop_dp": 37.05445002,
    "collector_flow_rates": np.array([
        9.176854533, 8.921921744, 8.699651745, 8.486496198, 8.259228335,
        8.039822348, 7.813881615, 7.590786245, 8.933134895, 8.706698355,
        8.481841868, 8.263997082, 8.041208355, 7.809690683, 7.593884098,
    ]),
    "pump_and_drive_efficiency": 0.533426021,
    "pump_and_drive_power": 40.88981445,
    "oil_return_temp": 273,
    "ambient_temp": 20,
    "solar_rate": 900,
    "loop_thermal_efficiencies": np.array([
        0.9, 0.88, 0.86, 0.84, 0.82,
        0.8, 0.78, 0.76, 0.88, 0.86,
        0.84, 0.82, 0.8, 0.78, 0.76
    ]),
    "oil_exit_temps": np.array([
        394.7139109242, 395.1327523393, 395.1245265650, 394.9873377920, 395.0471019449,
        394.9932710362, 395.0362115268, 395.0368832622, 394.9823714639, 395.0276211405,
        395.0529329871, 394.9781258321, 394.9726891283, 395.1002726780, 394.9882151924,
    ]),
    "rms_dev": 0.0949969508554
}


class TestPumpAndFlowCalculations:
    """Tests for pump and flow calculation functions."""

    def test_actual_pump_speed_from_scaled(self):
        """Test pump speed conversion."""
        assert actual_pump_speed_from_scaled(0.2) == PUMP_SPEED_MIN
        assert actual_pump_speed_from_scaled(1.0) == PUMP_SPEED_MAX

    def test_calculate_pump_fluid_power(self):
        """Test pump fluid power calculation."""
        total_flow_rate = test_data["total_flow_rate"]
        pump_dp = test_data["pump_dp"]
        pump_fluid_power = calculate_pump_fluid_power(total_flow_rate, pump_dp)
        assert np.isclose(pump_fluid_power, test_data["pump_fluid_power"])

    def test_calculate_collector_flow_rate(self):
        """Test collector flow rate calculation."""
        valve_position = test_data["valve_positions"][1]
        loop_dp = test_data["loop_dp"]
        collector_flow_rate = calculate_collector_flow_rate(
            valve_position, loop_dp, sqrt=np.sqrt
        )
        assert np.isclose(collector_flow_rate, test_data["collector_flow_rates"][1])

    def test_calculate_total_flowrate(self):
        """Test total flow rate calculation."""
        valve_positions = test_data["valve_positions"]
        loop_dp = test_data["loop_dp"]
        total_flowrate = calculate_total_flowrate(
            valve_positions, loop_dp, sum=np.sum, sqrt=np.sqrt
        )
        assert np.isclose(total_flowrate, test_data["total_flow_rate"])

    def test_calculate_boiler_dp(self):
        """Test boiler differential pressure calculation."""
        total_flow_rate = test_data["total_flow_rate"]
        boiler_dp = calculate_boiler_dp(total_flow_rate)
        assert np.isclose(boiler_dp, test_data["boiler_dp"])

    def test_calculate_pump_dp(self):
        """Test pump differential pressure calculation."""
        actual_pump_speed = test_data["actual_pump_speed"]
        total_flow_rate = test_data["total_flow_rate"]
        m_pumps = 2
        pump_dp = calculate_pump_dp(actual_pump_speed, total_flow_rate, m_pumps)
        assert np.isclose(pump_dp, test_data["pump_dp"])

    def test_calculate_pressure_balance(self):
        """Test pressure balance calculation."""
        loop_dp = test_data["loop_dp"]
        pump_dp = test_data["pump_dp"]
        boiler_dp = test_data["boiler_dp"]
        pressure_balance = calculate_pressure_balance(loop_dp, pump_dp, boiler_dp)
        assert np.isclose(pressure_balance, 0.0, atol=1e-7)

    def test_pump_and_drive_efficiency_and_power(self):
        """Test pump and drive efficiency and power calculations."""
        valve_positions = test_data["valve_positions"]
        loop_dp = test_data["loop_dp"]
        total_flow_rate = calculate_total_flowrate(
            valve_positions, loop_dp, sum=np.sum, sqrt=np.sqrt
        )
        assert np.isclose(total_flow_rate, test_data["total_flow_rate"])

        pump_speed_scaled = test_data["pump_speed_scaled"]
        actual_pump_speed = actual_pump_speed_from_scaled(pump_speed_scaled)
        assert np.isclose(actual_pump_speed, test_data["actual_pump_speed"])

        pump_dp = test_data["pump_dp"]
        pump_fluid_power = calculate_pump_fluid_power(total_flow_rate, pump_dp)
        assert np.isclose(pump_fluid_power, test_data["pump_fluid_power"])

        pump_and_drive_efficiency = calculate_pump_and_drive_efficiency(
            total_flow_rate, actual_pump_speed
        )
        assert np.isclose(
            pump_and_drive_efficiency, test_data["pump_and_drive_efficiency"]
        )

        pump_and_drive_power = pump_fluid_power / pump_and_drive_efficiency
        assert np.isclose(pump_and_drive_power, test_data["pump_and_drive_power"])


class TestOilTemperatureCalculations:
    """Tests for oil temperature calculation functions."""

    def test_calculate_collector_oil_exit_temp_single(self):
        """Test oil exit temperature calculation for single collector."""
        flow_rate = 8.968468201
        oil_return_temp = 273
        ambient_temp = 20
        solar_rate = 900
        loop_thermal_efficiency = 0.9
        oil_exit_temp = calculate_collector_oil_exit_temp(
            flow_rate,
            oil_return_temp,
            ambient_temp,
            solar_rate,
            loop_thermal_efficiency,
            exp=np.exp,
            pi=np.pi
        )
        assert np.isclose(oil_exit_temp, 397.4882567363479)

    def test_calculate_collector_oil_exit_temp_vectorized(self):
        """Test oil exit temperature calculation for multiple collectors."""
        collector_flow_rates = cas.DM(test_data["collector_flow_rates"])
        oil_return_temp = test_data["oil_return_temp"]
        ambient_temp = test_data["ambient_temp"]
        solar_rate = test_data["solar_rate"]
        loop_thermal_efficiencies = cas.DM(test_data["loop_thermal_efficiencies"])
        oil_exit_temps = calculate_collector_oil_exit_temp(
            collector_flow_rates,
            oil_return_temp,
            ambient_temp,
            solar_rate,
            loop_thermal_efficiencies
        )
        assert np.allclose(
            oil_exit_temps,
            test_data["oil_exit_temps"].reshape(-1, 1)
        )

    def test_calculate_rms_oil_exit_temps(self):
        """Test RMS deviation calculation."""
        oil_exit_temps = cas.DM(test_data["oil_exit_temps"])
        oil_exit_temps_sp = cas.DM([
            395.0, 395.0, 395.0, 395.0, 395.0,
            395.0, 395.0, 395.0, 395.0, 395.0,
            395.0, 395.0, 395.0, 395.0, 395.0,
        ])
        rms_dev = calculate_rms_oil_exit_temps(oil_exit_temps, oil_exit_temps_sp)
        assert np.isclose(rms_dev, test_data["rms_dev"], atol=0.00001)


class TestCasADiFunctions:
    """Tests for CasADi function constructors."""

    def test_make_pressure_balance_function(self):
        """Test pressure balance function creation."""
        n_lines = 15
        m_pumps = 2
        pressure_balance_function = make_pressure_balance_function(n_lines, m_pumps)

        valve_positions = test_data["valve_positions"]
        pump_speed_scaled = test_data["pump_speed_scaled"]
        loop_dp = test_data["loop_dp"]

        pressure_balance = pressure_balance_function(
            valve_positions, pump_speed_scaled, loop_dp
        )
        assert np.isclose(pressure_balance, [[0.0]], atol=1e-5)

    def test_make_calculate_pump_and_drive_power_function(self):
        """Test pump and drive power function creation."""
        n_lines = 15
        m_pumps = 2
        pump_and_drive_power_function = \
            make_calculate_pump_and_drive_power_function(n_lines, m_pumps)

        valve_positions = test_data["valve_positions"]
        pump_speed_scaled = test_data["pump_speed_scaled"]
        pump_and_drive_power = pump_and_drive_power_function(
            valve_positions, pump_speed_scaled
        )
        assert np.isclose(pump_and_drive_power, test_data["pump_and_drive_power"])

    def test_make_collector_exit_temps_and_pump_power_function(self):
        """Test complete system function creation."""
        n_lines = 15
        m_pumps = 2
        calculate_exit_temps_and_pump_power = \
            make_collector_exit_temps_and_pump_power_function(n_lines, m_pumps)

        valve_positions = test_data["valve_positions"]
        pump_speed_scaled = test_data["pump_speed_scaled"]
        oil_return_temp = test_data["oil_return_temp"]
        ambient_temp = test_data["ambient_temp"]
        solar_rate = test_data["solar_rate"]

        collector_flow_rates, pump_and_drive_power, oil_exit_temps = \
            calculate_exit_temps_and_pump_power(
                valve_positions, pump_speed_scaled, oil_return_temp,
                ambient_temp, solar_rate
            )

        assert np.allclose(
            collector_flow_rates,
            test_data["collector_flow_rates"].reshape(-1, 1)
        )
        assert np.isclose(pump_and_drive_power, test_data["pump_and_drive_power"])
        assert np.allclose(
            oil_exit_temps,
            test_data["oil_exit_temps"].reshape(-1, 1)
        )


class TestSteamGeneratorModel:
    """Tests for steam generator model functions."""

    def test_calculate_net_power(self):
        """Test net power calculation."""
        steam_power = 1639.813211
        pump_fluid_power = 21.81087724
        pump_and_drive_efficiency = 0.533387324
        net_power = calculate_net_power(
            steam_power, pump_fluid_power, pump_and_drive_efficiency
        )
        assert np.isclose(net_power, 1352.949974049002)


class TestDataConsistency:
    """Tests for test data internal consistency."""

    def test_flow_rates_sum(self):
        """Test that collector flow rates sum to total flow rate."""
        assert np.isclose(
            test_data["total_flow_rate"],
            np.sum(test_data["collector_flow_rates"])
        )
