"""Unit tests for solar_plant_rto module.

This replicates calculations in the Excel spreadsheet implementation
from R R Rhinehart so they can be solved using CasADi.

Excel file used for test data:
- `SS Solar Plant Optimization New Thermal & Flow Model 15 DVs I-O 2025-10-12a.xlsx`

"""

import casadi as cas
import numpy as np

from problems.solar_plant_rto.solar_plant_gen_rto import (
    PUMP_SPEED_MAX,
    PUMP_SPEED_MIN,
    actual_pump_speed_from_scaled,
    calculate_boiler_dp,
    calculate_collector_flow_rate,
    calculate_collector_oil_exit_temp,
    calculate_net_power,
    calculate_pressure_balance,
    calculate_pump_and_drive_efficiency,
    calculate_pump_dp,
    calculate_pump_fluid_power,
    calculate_rms_oil_exit_temps,
    calculate_total_flowrate,
    make_calculate_pump_and_drive_power_function,
    make_collector_exit_temps_and_pump_power_function,
    make_pressure_balance_function,
)

# Test data from Excel spreadsheet
test_data = {
    "n_lines": 15,  # ✓
    "m_pumps": 3,  # ✓
    "pump_speed_scaled": 0.739913018149835,  # ✓
    "actual_pump_speed": 2238.040923,  # ✓
    "pump_dp": 257.2307857,  # ✓
    "boiler_dp": 255.1689754,  # ✓
    "total_flow_rate": 84.98338307,  # ✓
    "pump_fluid_power": 6.072317334,  # ✓
    "valve_positions": np.array(  # ✓
        [
            1.0,
            0.955991652,
            0.922910885,
            0.894335128,
            0.869386075,
            0.846574862,
            0.826129052,
            0.80724668,
            0.956319885,
            0.922542333,
            0.894089015,
            0.869386891,
            0.846759913,
            0.826256799,
            0.807127717,
        ]
    ),
    "loop_dp": 2.061810328,  # ✓
    "collector_flow_rates": np.array(  # ✓
        [
            6.225501621,
            6.070138765,
            5.923894153,
            5.77480293,
            5.626327169,
            5.475290723,
            5.32759246,
            5.181170423,
            6.07145634,
            5.922109432,
            5.773422776,
            5.62633231,
            5.4765747,
            5.3285508,
            5.180218472,
        ]
    ),
    "pump_and_drive_efficiency": 0.338782267,  # ✓
    "pump_and_drive_power": 17.92395274,  # ✓
    "oil_return_temp": 273.5052133,  # ✓
    "ambient_temp": 20,  # ✓
    "solar_rate": 700,  # ✓
    "loop_thermal_efficiencies": np.array(  # ✓
        [
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
    ),
    "oil_exit_temps": np.array(
        [
            389.8923752,
            390.0208235,
            389.9789458,
            389.9915139,
            389.9921759,
            390.0464908,
            390.031934,
            389.9883183,
            389.9958936,
            390.0135412,
            390.0189488,
            389.9920711,
            390.0195934,
            390.0113121,
            390.0093716,
        ]
    ),
    "rms_dev": 4.999893601,
}


# Items changed compared t0 test_solar_plant_rto.py:
#   - m_pumps: 2 -> 3
#   - pump_speed_scaled (min): 0.2 -> 0.3
#   - collector valve flowrate formula changed
#   - change to calculate_pump_dp
#   - collector outlet temp formula changed (a, b)


class TestPumpAndFlowCalculations:
    """Tests for pump and flow calculation functions."""

    def test_total_flow_rate_data(self):  # ✓
        assert np.isclose(
            test_data["total_flow_rate"],
            np.sum(test_data["collector_flow_rates"]),
        )

    def test_actual_pump_speed_from_scaled(self):  # ✓
        """Test pump speed conversion."""
        assert actual_pump_speed_from_scaled(0.3) == PUMP_SPEED_MIN
        assert actual_pump_speed_from_scaled(1.0) == PUMP_SPEED_MAX
        pump_speed_scaled = test_data["pump_speed_scaled"]
        actual_pump_speed = test_data["actual_pump_speed"]
        assert np.isclose(
            actual_pump_speed_from_scaled(pump_speed_scaled), actual_pump_speed
        )

    def test_calculate_pump_fluid_power(self):  # ✓
        """Test pump fluid power calculation."""
        total_flow_rate = test_data["total_flow_rate"]
        pump_dp = test_data["pump_dp"]
        pump_fluid_power = calculate_pump_fluid_power(total_flow_rate, pump_dp)
        assert np.isclose(pump_fluid_power, test_data["pump_fluid_power"])

    def test_calculate_collector_flow_rate(self):  # ✓
        """Test collector flow rate calculation."""
        valve_position = test_data["valve_positions"][1]
        loop_dp = test_data["loop_dp"]
        collector_flow_rate = calculate_collector_flow_rate(
            valve_position, loop_dp, sqrt=np.sqrt
        )
        assert np.isclose(
            collector_flow_rate, test_data["collector_flow_rates"][1]
        )

    def test_calculate_total_flowrate(self):  # ✓
        """Test total flow rate calculation."""
        valve_positions = test_data["valve_positions"]
        loop_dp = test_data["loop_dp"]
        total_flowrate = calculate_total_flowrate(
            valve_positions, loop_dp, sum=np.sum, sqrt=np.sqrt
        )
        assert np.isclose(total_flowrate, test_data["total_flow_rate"])

    def test_calculate_boiler_dp(self):  # ✓
        """Test boiler differential pressure calculation."""
        total_flow_rate = test_data["total_flow_rate"]
        boiler_dp = calculate_boiler_dp(total_flow_rate)
        assert np.isclose(boiler_dp, test_data["boiler_dp"])

    def test_calculate_pump_dp(self):  # ✓
        """Test pump differential pressure calculation."""
        actual_pump_speed = test_data["actual_pump_speed"]
        total_flow_rate = test_data["total_flow_rate"]
        m_pumps = test_data["m_pumps"]
        pump_dp = calculate_pump_dp(
            actual_pump_speed, total_flow_rate, m_pumps
        )
        assert np.isclose(pump_dp, test_data["pump_dp"])

    def test_calculate_pressure_balance(self):  # ✓
        """Test pressure balance calculation."""
        loop_dp = test_data["loop_dp"]
        pump_dp = test_data["pump_dp"]
        boiler_dp = test_data["boiler_dp"]
        pressure_balance = calculate_pressure_balance(
            loop_dp, pump_dp, boiler_dp
        )
        assert np.isclose(pressure_balance, 0.0, atol=1e-7)

    def test_pump_and_drive_efficiency_and_power(self):  # ✓
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

        m_pumps = test_data["m_pumps"]
        pump_and_drive_efficiency = calculate_pump_and_drive_efficiency(
            total_flow_rate / m_pumps, actual_pump_speed
        )
        assert np.isclose(
            pump_and_drive_efficiency, test_data["pump_and_drive_efficiency"]
        )

        pump_and_drive_power = pump_fluid_power / pump_and_drive_efficiency
        assert np.isclose(
            pump_and_drive_power, test_data["pump_and_drive_power"]
        )


class TestOilTemperatureCalculations:
    """Tests for oil temperature calculation functions."""

    def test_calculate_collector_oil_exit_temp_single(self):  # ✓
        """Test oil exit temperature calculation for single collector."""
        flow_rate = test_data["collector_flow_rates"][0]
        oil_return_temp = test_data["oil_return_temp"]
        ambient_temp = test_data["ambient_temp"]
        solar_rate = test_data["solar_rate"]
        loop_thermal_efficiency = test_data["loop_thermal_efficiencies"][0]
        oil_exit_temp = calculate_collector_oil_exit_temp(
            flow_rate,
            oil_return_temp,
            ambient_temp,
            solar_rate,
            loop_thermal_efficiency,
            exp=np.exp,
            pi=np.pi,
        )
        assert np.isclose(oil_exit_temp, test_data["oil_exit_temps"][0])

    def test_calculate_collector_oil_exit_temp_vectorized(self):  # ✓
        """Test oil exit temperature calculation for multiple collectors."""
        collector_flow_rates = cas.DM(test_data["collector_flow_rates"])
        oil_return_temp = test_data["oil_return_temp"]
        ambient_temp = test_data["ambient_temp"]
        solar_rate = test_data["solar_rate"]
        loop_thermal_efficiencies = cas.DM(
            test_data["loop_thermal_efficiencies"]
        )
        oil_exit_temps = calculate_collector_oil_exit_temp(
            collector_flow_rates,
            oil_return_temp,
            ambient_temp,
            solar_rate,
            loop_thermal_efficiencies,
        )
        assert np.allclose(
            oil_exit_temps, test_data["oil_exit_temps"].reshape(-1, 1)
        )

    def test_calculate_rms_oil_exit_temps(self):  # ✓
        """Test RMS deviation calculation."""
        oil_exit_temps = cas.DM(test_data["oil_exit_temps"])
        oil_exit_temps_sp = cas.repmat(cas.DM(395.0), 15, 1)
        rms_dev = calculate_rms_oil_exit_temps(
            oil_exit_temps, oil_exit_temps_sp
        )
        assert np.isclose(rms_dev, test_data["rms_dev"], atol=0.00001)


class TestCasADiFunctions:
    """Tests for CasADi function constructors."""

    def test_make_pressure_balance_function(self):
        """Test pressure balance function creation."""
        n_lines = test_data["n_lines"]
        m_pumps = test_data["m_pumps"]
        pressure_balance_function = make_pressure_balance_function(
            n_lines, m_pumps
        )

        valve_positions = test_data["valve_positions"]
        pump_speed_scaled = test_data["pump_speed_scaled"]
        loop_dp = test_data["loop_dp"]

        pressure_balance = pressure_balance_function(
            valve_positions, pump_speed_scaled, loop_dp
        )
        assert np.isclose(pressure_balance, [[0.0]], atol=1e-5)

    def test_make_calculate_pump_and_drive_power_function(self):
        """Test pump and drive power function creation."""
        n_lines = test_data["n_lines"]
        m_pumps = test_data["m_pumps"]
        pump_and_drive_power_function = (
            make_calculate_pump_and_drive_power_function(n_lines, m_pumps)
        )

        valve_positions = test_data["valve_positions"]
        pump_speed_scaled = test_data["pump_speed_scaled"]
        pump_and_drive_power = pump_and_drive_power_function(
            valve_positions, pump_speed_scaled
        )
        assert np.isclose(
            pump_and_drive_power, test_data["pump_and_drive_power"]
        )

    def test_make_collector_exit_temps_and_pump_power_function(self):
        """Test complete system function creation."""
        n_lines = test_data["n_lines"]
        m_pumps = test_data["m_pumps"]
        calculate_exit_temps_and_pump_power = (
            make_collector_exit_temps_and_pump_power_function(n_lines, m_pumps)
        )

        valve_positions = test_data["valve_positions"]
        pump_speed_scaled = test_data["pump_speed_scaled"]
        oil_return_temp = test_data["oil_return_temp"]
        ambient_temp = test_data["ambient_temp"]
        solar_rate = test_data["solar_rate"]

        collector_flow_rates, pump_and_drive_power, oil_exit_temps = (
            calculate_exit_temps_and_pump_power(
                valve_positions,
                pump_speed_scaled,
                oil_return_temp,
                ambient_temp,
                solar_rate,
            )
        )

        assert np.allclose(
            collector_flow_rates,
            test_data["collector_flow_rates"].reshape(-1, 1),
        )
        assert np.isclose(
            pump_and_drive_power, test_data["pump_and_drive_power"]
        )
        assert np.allclose(
            oil_exit_temps, test_data["oil_exit_temps"].reshape(-1, 1)
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
            np.sum(test_data["collector_flow_rates"]),
        )
