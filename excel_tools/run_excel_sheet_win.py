import os
import time
import win32com.client as win32
import pywintypes
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict


def get_cell(ws, cell_ref):
    try:
        cell = ws.Range(cell_ref)
    except pywintypes.com_error:
        try:
            cell = ws.Cells(*cell_ref)
        except Exception as e:
            raise ValueError(f"Failed to read cell {cell_ref}: {e}")
    return cell


def validate_name_value(ws, name, cell_ref):
    """Validate variable name in cell in Excel sheet"""
    name_cell_value = get_cell(ws, cell_ref).Value
    if name_cell_value != name:
        raise ValueError(
            f"Variable name mismatch for '{name}': expected "
            f"'{name}', got '{name_cell_value}' in cell {cell_ref}"
        )


def get_var_value(ws, name, cell_ref):
    """Retrieve a value from a cell or group of cells in the Excel
    sheet that matches the name and cell reference given.

    Args:
        ws (xlwings.main.Sheet): Excel worksheet.
        name (str): The name of the variable. This will be used to compare to
            the name in the Excel sheet.
        cell_ref: Tuple containing a reference to a cell containing the
            variable name and a cell reference or list of cell references
            containing the value(s). E.g. ("B2", "C3") for one value, or ("B2",
            ["C3", "D3", "E3]) for a vector of three values.

    Returns:
        value: Either the value from the specified cell or a list of values
            if more than one is specified.
    """

    name_cell_ref, value_cell_refs = cell_ref

    # Check variable name matches variable label in sheet
    validate_name_value(ws, name, name_cell_ref)

    if isinstance(value_cell_refs, (str, tuple)):
        # Single cell reference, "A1" or (row, col) style,
        # or a range like "C4:D4".
        return get_cell(ws, value_cell_refs).Value
    elif isinstance(value_cell_refs, list):
        # Get multiple values
        value = [get_cell(ws, cell_ref).Value for cell_ref in value_cell_refs]
    else:
        raise TypeError(
            f"Invalid cell reference type: {type(value_cell_refs)}. "
            "Cell reference must be str, tuple or a list of these types."
        )

    return value


def set_var_value(ws, name, cell_ref, values):
    """Retrieve a value from a cell or group of cells in the Excel
    sheet that matches the name and cell reference given.

    Args:
        ws (xlwings.main.Sheet): Excel worksheet.
        name (str): The name of the variable. This will be used to compare to
            the name in the Excel sheet.
        cell_ref: Tuple containing a reference to a cell containing the
            variable name and a cell reference or list of cell references
            containing the value(s). E.g. ("B2", "C3") for one value, or ("B2",
            ["C3", "D3", "E3]) for a vector of three values.
        values: (str, int, float or list of these types)

    Returns:
        value: Either the value from the specified cell or a list of values
            if more than one is specified.
    """
    name_cell_ref, value_cell_refs = cell_ref

    # Check variable name matches variable label in sheet
    validate_name_value(ws, name, name_cell_ref)

    # Set value(s)
    if isinstance(value_cell_refs, (str, tuple)):
        # Single cell reference, "A1" or (row, col) style,
        # or a range like "C4:D4".
        get_cell(ws, value_cell_refs).Value = values
    elif isinstance(value_cell_refs, list):
        # Set multiple values
        for cell_ref, value in zip(value_cell_refs, values):
            get_cell(ws, cell_ref).Value = value
    else:
        raise TypeError(
            f"Invalid cell reference type: {type(value_cell_refs)}. "
            "Cell reference must be str, tuple or a list of these types."
        )


def evaluate_excel_sheet(
    excel, wb, inputs, cell_refs, output_vars=None, sheet=1
):
    """Inserts input variable values into specified cells,
    recalculates the sheet, and then reads the values of the
    remaining specified cells and returns them as a dictionary.

    The names and cell locations of the input and output
    variables are specified in the cell_refs dictionary.

    Args:
        wb (win32com.gen_py.Workbook): An open Excel workbook.
        inputs (dict): values to be assigned to input variables.
            E.g. inputs = {'x': [0.0, 1.0]}
        sheet (int): Sheet number (1-based).
    
    Returns:
        outputs (dict): The cell values for all variables except the
            input variable(s) after the sheet was recalculated.

    Example:

    To set up a function evaluation, f(x), that takes two inputs,
    x[0] and x[1], has a constraint function, g(x) <= 0, and lower
    and upper bounds on x, x_lb and x_ub, the Excel sheet could
    look something like this:

         | A       B       C       D        
    -----|----------------------------------
       1 |                                  
       2 |         name    My Problem       
       3 |         f(x)    3.0              
       4 |         x       1.0     2.0      
       5 |         g(x)    10.0             
       6 |         x_lb    0.0     0.0      
       7 |         x_ub    5.0     5.0      

    The cell references for this example would be:

    cell_refs = {
        'name': ('B2', 'C2'),
        'f(x)': ('B3', 'C3'),
        'x': ('B4', ['C4', 'D4']),
        'g(x)': ('B5', 'C5'),
        'x_lb': ('B6', ['C6', 'D6']),
        'x_ub': ('B7', ['C7', 'D7'])
    }

    Notes:

    The reason the name fields are included is that the algorithm
    checks that the text in the specified cells matches the variables
    names to reduce the risk of referencing errors.

    To avoid unnecessary re-calculation, set the Excel calculation
    option to 'manual'. Recalculation will be automatically triggered
    when this function is called.
    """

    # Get the worksheet
    ws = wb.Sheets(sheet)

    # Copy input variable values to specified cells
    for (var_name, values) in inputs.items():
        name_cell, value_cells = cell_refs[var_name]

        # Check variable name matches
        name_cell_value = ws.Range(name_cell).Value
        msg = (
            f"variable name mismatch: expected {var_name}, "
            f"got {name_cell_value}"
        )
        assert name_cell_value == var_name, msg

        # Set values
        try:
            len(values)
        except TypeError:
            ws.Range(value_cells).Value = values
        else:
            for cell_ref, value in zip(value_cells, values):
                ws.Range(cell_ref).Value = value

    # Wait until all asynchronous queries (e.g. Power Query) are complete
    excel.CalculateUntilAsyncQueriesDone()

    # Force Excel to recalculate all formulas
    excel.CalculateFullRebuild()

    # If not specified, read all variable values except the inputs
    if output_vars is None:
        output_vars = set(cell_refs.keys()) - set(inputs.keys())

    outputs = {}
    for var_name in output_vars:
        name_cell, value_cells = cell_refs[var_name]
        assert ws.Range(name_cell).Value == var_name, "variable name mismatch"
        try:
            value = ws.Range(value_cells).Value
        except pywintypes.com_error:
            # Allow cell_ref to be a list of cell refs
            value = [
                ws.Range(cell_ref).Value for cell_ref in value_cells
            ]
        outputs[var_name] = value

    return outputs
