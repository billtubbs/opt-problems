import os
import numpy as np

from excel_tools.opt_problems import MSExcelOptProblem


# References to locations of input-output variables in Excel sheet.
# These are in (row, col) format.  They can also be supplied in "A1" style.
CELL_REFS = {
    'name': ((2, 2), (2, 3)),
    'f(x)': ((6, 13), (6, 14)),
    'x': (
        (5, 8),
        [
            (6, 8), (9, 8), (10, 8), (11, 8), (12, 8),
            (13, 8), (14, 8), (15, 8), (16, 8), (17, 8),
            (18, 8), (19, 8), (20, 8), (21, 8), (22, 8)
        ]
    ),
    'g(x)': None,
    'x_lb': (
        (5, 9),
        [
            (6, 9), (9, 9), (10, 9), (11, 9), (12, 9),
            (13, 9), (14, 9), (15, 9), (16, 9), (17, 9),
            (18, 9), (19, 9), (20, 9), (21, 9), (22, 9)
        ]
    ),
    'x_ub': (
        (5, 10),
        [
            (6, 10), (9, 10), (10, 10), (11, 10), (12, 10),
            (13, 10), (14, 10), (15, 10), (16, 10), (17, 10),
            (18, 10), (19, 10), (20, 10), (21, 10), (22, 10)
        ]
    )
}


# Path to Excel file
PROBLEMS_DIR = 'problems'
PROBLEM_NAME = 'solar_plant_rto'
EXCEL_FILENAME = 'Solar Plant Optimization of N-Pumps I-O 2025-08-29.xlsm'
FILEPATH = os.path.abspath(
    os.path.join(PROBLEMS_DIR, PROBLEM_NAME, EXCEL_FILENAME)
)
SHEET_NO = 1


class SolarPlantRTO(MSExcelOptProblem):

    def __init__(self, filepath=FILEPATH, cell_refs=CELL_REFS, sheet=SHEET_NO):
        super().__init__(filepath, cell_refs, sheet=sheet)


# Test problem instance
with SolarPlantRTO() as problem:
    assert problem.name == "SolarPlantRTO"
    assert len(problem.bounds) == 15
    x = [
        0.5,
        0.917004705,
        0.886780974,
        0.853623916,
        0.826111695,
        0.801142972,
        0.783225649,
        0.760051185,
        0.900587117,
        0.884820289,
        0.85561633,
        0.827130134,
        0.797646554,
        0.78549898,
        0.755400025
    ]
    assert np.isclose(problem(x), 43.5494444649505)
