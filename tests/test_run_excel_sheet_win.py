import os
import time
import pytest
import win32com.client as win32
import pythoncom
import gc
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from excel_tools.run_excel_sheet_win import (
    validate_name_value,
    get_var_value,
    set_var_value,
    evaluate_excel_sheet
)
from pandas.testing import assert_frame_equal
from tqdm import tqdm


TEST_DATA_DIR = 'test/data'
TEST_RESULTS_DIR = 'test/results'
PROBLEMS_DIR = "problems"

TEST_PROBLEMS = {
    "toy_1d": {
        "filename": "Toy-1D-Problem.xlsx",
        "sheet": 1,
        "expected_values": {
            "name": "Toy1DProblem",
            "x_lb": -5.0,
            "x_ub": 5.0,
            "f(x)": 0.18727730945265594,
            "x": 2.0,
        },
        "cell_refs": {
            "A1": {
                'name': ('B2', 'C2'),
                'f(x)': ('B3', 'C3'),
                'x': ('B4', 'C4'),
                'x_lb': ('B5', 'C5'),
                'x_ub': ('B6', 'C6')
            },
            "R1C1": {
                'name': ((2, 2), (2, 3)),
                'f(x)': ((3, 2), (3, 3)),
                'x': ((4, 2), (4, 3)),
                'x_lb': ((5, 2), (5, 3)),
                'x_ub': ((6, 2), (6, 3))
            }
        }
    },
    "toy_2d_const": {
        "filename": "Toy-2D-Problem-Constraint.xlsx",
        "sheet": 1,
        "expected_values": {
            "name": "Toy2DProblemConstraint",
            "x_lb": [-5.0, -5.0],
            "x_ub": [5.0, 5.0],
            "f(x)": 0.0,
            "g(x)": 10.0,
            "x": [0.0, 0.0]
        },
        "cell_refs": {
            "A1": {
                'name': ('B2', 'C2'),
                'f(x)': ('B3', 'C3'),
                'x': ('B4', ['C4', 'D4']),
                'g(x)': ('B5', 'C5'),
                'x_lb': ('B6', ['C6', 'D6']),
                'x_ub': ('B7', ['C7', 'D7'])
            },
            "R1C1": {
                'name': ((2, 2), (2, 3)),
                'f(x)': ((3, 2), (3, 3)),
                'x': ((4, 2), [(4, 3), (4, 4)]),
                'g(x)': ((5, 2), (5, 3)),
                'x_lb': ((6, 2), [(6, 3), (6, 4)]),
                'x_ub': ((7, 2), [(7, 3), (7, 4)]),
            },
        }
    }
}


@pytest.fixture
def excel_app():
    """Pytest fixture that provides a clean Excel instance for each test."""
    pythoncom.CoInitialize()
    excel = win32.Dispatch('Excel.Application')
    excel.Visible = False
    excel.DisplayAlerts = False

    yield excel

    # Cleanup after test
    try:
        # Close any remaining workbooks
        while excel.Workbooks.Count > 0:
            excel.Workbooks(1).Close(SaveChanges=False)
    except:
        pass

    try:
        excel.DisplayAlerts = True
        excel.Quit()
    except:
        pass

    gc.collect()
    pythoncom.CoUninitialize()


@pytest.mark.parametrize("ref_style", ["A1", "R1C1"])
def test_validate_name_value(excel_app, ref_style):
    problem = "toy_1d"
    problem_data = TEST_PROBLEMS[problem]

    # Path to Excel file
    filename = problem_data['filename']
    filepath = os.path.abspath(os.path.join(PROBLEMS_DIR, problem, filename))

    sheet = problem_data['sheet']
    cell_refs = problem_data['cell_refs'][ref_style]
    expected_values = problem_data['expected_values']

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")

    wb = None
    try:
        # Use the fixture-provided Excel instance
        wb = excel_app.Workbooks.Open(filepath)
        ws = wb.Sheets(sheet)

        expected_name = expected_values['name']
        name_value_cell_ref = cell_refs['name'][1]
        validate_name_value(ws, expected_name, name_value_cell_ref)

    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)


@pytest.mark.parametrize("problem", ["toy_1d", "toy_2d_const"])
@pytest.mark.parametrize("ref_style", ["A1", "R1C1"])
def test_set_and_get_var_value(excel_app, problem, ref_style):

    problem_data = TEST_PROBLEMS[problem]

    # Path to Excel file
    filename = problem_data['filename']
    filepath = os.path.abspath(os.path.join(PROBLEMS_DIR, problem, filename))

    sheet = problem_data['sheet']
    cell_refs = problem_data['cell_refs'][ref_style]
    expected_values = problem_data['expected_values']

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")

    x_name = 'x'
    x = expected_values[x_name]
    values = {x_name: x}
    other_var_names = list(expected_values.keys())
    other_var_names.remove(x_name)

    wb = None
    try:
        # Use the fixture-provided Excel instance
        wb = excel_app.Workbooks.Open(filepath)
        ws = wb.Sheets(sheet)

        set_var_value(ws, x_name, cell_refs[x_name], x)

        # Wait until all asynchronous queries (e.g. Power Query) are complete
        excel = wb.Application
        excel.CalculateUntilAsyncQueriesDone()

        # Force Excel to recalculate all formulas
        excel.CalculateFullRebuild()

        for name in other_var_names:
            values[name] = get_var_value(ws, name, cell_refs[name])

    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)

    assert values == expected_values


def test_on_Toy1DProblem(excel_app):

    # Path to Excel file
    problem_name = "toy_1d"
    filename = "Toy-1D-Problem.xlsx"
    cell_refs = TEST_PROBLEMS[problem_name]["cell_refs"]["A1"]
    filepath = os.path.abspath(
        os.path.join(PROBLEMS_DIR, problem_name, filename)
    )

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")

    wb = None
    try:
        # Use the fixture-provided Excel instance
        wb = excel_app.Workbooks.Open(filepath)

        # Check the problem parameters
        inputs = {'x': 0.0}
        outputs = evaluate_excel_sheet(
            wb, inputs, cell_refs, output_vars=['name', 'x_lb', 'x_ub']
        )
        print("Test name: ", outputs['name'])
        assert outputs['name'] == 'Toy1DProblem'
        assert outputs['x_lb'] == -5
        assert outputs['x_ub'] == 5

        x_values = np.linspace(-5, 5, 11)
        f_eval = []
        t0 = time.time()
        timings = []
        print("Evaluating excel sheet...")
        for x in tqdm(x_values):
            inputs = {'x': x}
            outputs = evaluate_excel_sheet(
                wb,
                inputs,
                cell_refs,
                output_vars=['f(x)'],
                sheet=1
            )
            f_eval.append(outputs['f(x)'])
            timings.append(time.time() - t0)

    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)

    # Save results and timings
    results_summary = pd.DataFrame(
        {'Time (seconds)': timings, 'x': x_values, 'f(x)': f_eval}
    )
    filename = f"{problem_name}_win32.csv"
    results_summary.to_csv(os.path.join(TEST_RESULTS_DIR, filename))

    # Check results match data on file
    filename = 'Toy1DProblem.csv'
    expected_results = pd.read_csv(
        os.path.join(TEST_DATA_DIR, filename), index_col=0
    )

    assert_frame_equal(results_summary[['x', 'f(x)']], expected_results)


def test_on_Toy2DProblemConstraint(excel_app):

    # Path to Excel file
    problem_name = "toy_2d_const"
    filename = "Toy-2D-Problem-Constraint.xlsx"
    cell_refs = TEST_PROBLEMS[problem_name]["cell_refs"]["A1"]
    filepath = os.path.abspath(
        os.path.join(PROBLEMS_DIR, problem_name, filename)
    )

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")

    wb = None
    try:
        # Use the fixture-provided Excel instance
        wb = excel_app.Workbooks.Open(filepath)

        # Check the problem parameters
        inputs = {'x': [0.0, 0.0]}
        outputs = evaluate_excel_sheet(
            wb, inputs, cell_refs, output_vars=['name', 'x_lb', 'x_ub']
        )
        print("Test: ", outputs['name'])
        assert outputs['name'] == 'Toy2DProblemConstraint'
        assert outputs['x_lb'] == [-5, -5]
        assert outputs['x_ub'] == [5, 5]

        x1_values = np.linspace(-5, 5, 11)
        x2_values = np.linspace(-5, 5, 11)
        n_iters = x1_values.shape[0] * x2_values.shape[0]
        results = defaultdict(list)
        t0 = time.time()
        print("Evaluating excel sheet...")
        for x1, x2 in tqdm(product(x1_values, x2_values), total=n_iters):
            inputs = {'x': [x1, x2]}
            outputs = evaluate_excel_sheet(
                wb, inputs, cell_refs, output_vars=['f(x)', 'g(x)']
            )
            results['x1'].append(x1)
            results['x2'].append(x2)
            results['f_eval'].append(outputs['f(x)'])
            results['g_eval'].append(outputs['g(x)'])
            results['timings'].append(time.time() - t0)

    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)

    # Save results and timings
    results_summary = pd.DataFrame(results)
    filename = f"{problem_name}_win32.csv"
    results_summary.to_csv(os.path.join(TEST_RESULTS_DIR, filename))

    # Check results match data on file
    filename = 'Toy2DProblemConstraint.csv'
    expected_results = pd.read_csv(
        os.path.join(TEST_DATA_DIR, filename), index_col=0
    )
    assert_frame_equal(
        results_summary[['x1', 'x2', 'f_eval', 'g_eval']],
        expected_results
    )
