import xlwings as xw
import numpy as np

from problems.optprob.problems import ConstrainedScalarOptimizationProblem
from excel_tools.run_excel_sheet_mac import get_var_value, evaluate_excel_sheet


class MSExcelOptProblem(ConstrainedScalarOptimizationProblem):

    def __init__(self, filepath, cell_refs, sheet=1, global_minimum=None):
        self.filepath = filepath
        self.cell_refs = cell_refs
        self.sheet = sheet
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

    def __enter__(self):
        self._wb = xw.Book(self.filepath)
        self._ws = self._wb.sheets[self.sheet - 1]  # 0-based indexing
        self._bounds = self.bounds
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._wb.save()
        self._wb.close()

    def cost_function_to_minimize(self, x, *args) -> float:
        inputs = {'x': x}
        outputs = evaluate_excel_sheet(
            self._wb,
            inputs,
            self.cell_refs,
            output_vars=['f(x)'],
            sheet=self.sheet
        )
        cost = outputs['f(x)']
        return cost

    # TODO: Implement constraint function
