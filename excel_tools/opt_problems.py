import sys
import numpy as np

from problems.optprob.problems import ConstrainedScalarOptimizationProblem


if sys.platform == "darwin":  # macOS
    from excel_tools.run_excel_sheet_mac import (
        open_excel,
        open_workbook,
        get_worksheet,
        close_workbook,
        quit_excel,
        get_var_value,
        evaluate_excel_sheet
    )
elif sys.platform == "win32":  # Windows
    from excel_tools.run_excel_sheet_win import (
        open_excel,
        open_workbook,
        get_worksheet,
        close_workbook,
        quit_excel,
        get_var_value,
        evaluate_excel_sheet
    )
else:
    raise ImportError(f"Unsupported platform: {sys.platform}")


class MSExcelOptProblem(ConstrainedScalarOptimizationProblem):

    def __init__(self, filepath, cell_refs, sheet=1, global_minimum=None):
        self.filepath = filepath
        self.cell_refs = cell_refs
        self.sheet_no = sheet
        self._excel = None
        self._wb = None
        self._ws = None
        name = None
        bounds = None
        super().__init__(bounds, name=name, global_minimum=None)

    @property
    def name(self) -> str:
        name = get_var_value(self._ws, "name", self.cell_refs["name"])
        return name

    @property
    def bounds(self):
        lower_bounds = get_var_value(self._ws, "x_lb", self.cell_refs["x_lb"])
        upper_bounds = get_var_value(self._ws, "x_ub", self.cell_refs["x_ub"])
        return np.stack([lower_bounds, upper_bounds]).T

    @bounds.setter
    def bounds(self, value: np.ndarray) -> None:
        raise AttributeError(
            "Cannot set bounds using this method. Set them in the spreadsheet."
        )

    @property
    def ws(self):
        return self._ws

    @property
    def wb(self):
        return self._wb

    @property
    def excel(self):
        return self._excel

    def __enter__(self):
        self._excel = open_excel()
        self._wb = open_workbook(self._excel, self.filepath)
        self._ws = get_worksheet(self._wb, self.sheet_no)
        self._bounds = self.bounds
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        close_workbook(self._wb)
        quit_excel(self._excel)

    def cost_function_to_minimize(self, x, *args) -> float:
        inputs = {'x': x}
        outputs = evaluate_excel_sheet(
            self._wb,
            inputs,
            self.cell_refs,
            output_vars=['f(x)'],
            sheet=self.sheet_no
        )
        cost = outputs['f(x)']
        return cost

    # TODO: How to implement constraint function?
    # TODO: Way to get attributes after exiting context manager (name, bounds, etc)
