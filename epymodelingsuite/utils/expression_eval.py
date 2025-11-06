"""Safe expression evaluation utilities for model parameters.

This module provides tools for safely evaluating numeric expressions from strings,
allowing literal numbers, basic arithmetic operators, and functions from numpy and scipy.

Classes
-------
SafeEvalVisitor : ast.NodeVisitor
    Validates AST nodes to ensure only allowed expressions (numeric, numpy, scipy)
RetrieveName : ast.NodeTransformer
    Substitutes parameter values and contact matrix eigenvalues in expressions

Functions
---------
_safe_eval : Safely evaluate a numeric expression from a string
"""

import ast
import operator
from typing import Any

import numpy as np
import scipy
from epydemix.model import EpiModel

# Allowed binary operators mapping
_allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

# Allowed unary operators mapping
_allowed_unary_operators = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Names of top-level modules we allow
_allowed_modules = {"np", "scipy"}


class SafeEvalVisitor(ast.NodeVisitor):
    """
    A NodeVisitor that only allows numeric, numpy, and scipy expressions,
    and enables binary operations on numpy arrays.
    """

    def visit(self, node):
        t = type(node)
        # Permit only these node types
        if t in (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.List,
            ast.Load,
            ast.Name,
            ast.Attribute,
            ast.Call,
        ):
            return super().visit(node)
        raise ValueError(f"Disallowed expression: {t.__name__}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if type(node.op) not in _allowed_operators:
            raise ValueError(f"Operator {type(node.op).__name__} not allowed")

        # Access node data, handle arrays
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            left = left.value
            right = right.value
        elif isinstance(left, ast.Constant) and isinstance(right, ast.List):
            left = np.array([left.value])
            right = np.array([c.value for c in right.elts], dtype=float)
        elif isinstance(left, ast.List) and isinstance(right, ast.Constant):
            left = np.array([c.value for c in left.elts], dtype=float)
            right = np.array([right.value])
        elif isinstance(left, ast.List) and isinstance(right, ast.List):
            left = np.array([c.value for c in left.elts], dtype=float)
            right = np.array([c.value for c in right.elts], dtype=float)
        else:
            raise ValueError(f"Attempted BinOp on unsupported objects:\n\n{left}\n\n{right}")

        # Perform calculation
        calc_val = None
        if isinstance(node.op, ast.Add):
            calc_val = np.add(left, right, dtype=float)
        elif isinstance(node.op, ast.Sub):
            calc_val = np.subtract(left, right, dtype=float)
        elif isinstance(node.op, ast.Mult):
            calc_val = np.multiply(left, right, dtype=float)
        elif isinstance(node.op, ast.Div):
            calc_val = np.divide(left, right, dtype=float)

        if isinstance(calc_val, np.ndarray):
            ast_nodes = [ast.Constant(value=item) for item in calc_val.flatten()]
            ast_list = ast.List(elts=ast_nodes, ctx=ast.Load())
            return ast.fix_missing_locations(ast_list)
        return ast.fix_missing_locations(ast.Constant(value=calc_val))

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        if type(node.op) not in _allowed_unary_operators:
            raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")

    def visit_Constant(self, node):
        # Only allow numeric constants
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Constant of type {type(node.value).__name__} not allowed")
        return node

    def visit_List(self, node):
        for v in node.elts:
            self.visit(v)
        return node

    def visit_Name(self, node):
        # Only allow top‐level names 'np' and 'scipy'
        if node.id not in _allowed_modules:
            raise ValueError(f"Name '{node.id}' is not allowed")

    def visit_Attribute(self, node):
        # Recursively ensure base is allowed module (np or scipy)
        if self._is_allowed_attr_chain(node):
            # visit the base value to enforce nested checks
            self.visit(node.value)
        else:
            raise ValueError(f"Attribute access '{ast.dump(node)}' not allowed")

    def _is_allowed_attr_chain(self, node):
        # Base case: node.value is Name in allowed_modules
        if isinstance(node.value, ast.Name) and node.value.id in _allowed_modules:
            return True
        # Recursive: node.value is another Attribute
        if isinstance(node.value, ast.Attribute):
            return self._is_allowed_attr_chain(node.value)
        return False

    def visit_Call(self, node):
        # Only allow calls of form (np.xxx(...)) or (scipy.xxx(...))
        if isinstance(node.func, ast.Attribute):
            # validate the attribute chain (np or scipy)
            self.visit(node.func)
            # validate all arguments
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)
        else:
            raise ValueError("Function calls other than np.xxx or scipy.xxx are not allowed")


class RetrieveName(ast.NodeTransformer):
    """
    A NodeTransformer for substituting terms in an expression with parameter values or contact matrix eigenvalue from an EpiModel.
    Used for calculated parameters.
    Constructor requires an EpiModel with contact matrices.
    """

    def __init__(self, model: EpiModel):
        self.model = model

    def visit_Name(self, node):
        if node.id not in _allowed_modules:
            if node.id == "eigenvalue":
                try:
                    C = np.sum([c for _, c in self.model.population.contact_matrices.items()], axis=0)
                    eigenvalue = np.linalg.eigvals(C).real.max()
                    return ast.fix_missing_locations(ast.Constant(value=float(eigenvalue)))
                except Exception as e:
                    raise ValueError(f"Error calculating eigenvalue of contact matrix: {e}")
            else:
                try:
                    value = self.model.get_parameter(node.id)
                    if isinstance(value, np.ndarray):
                        assert value.shape[0] == 1, (
                            "Parameter calculation using parameters with array values is only implemented for age-varying parameters."
                        )
                        ast_nodes = [ast.Constant(value=float(item)) for item in value.flatten()]
                        return ast.fix_missing_locations(ast.List(elts=ast_nodes, ctx=ast.Load()))
                    return ast.fix_missing_locations(ast.Constant(value=float(value)))
                except Exception as e:
                    raise ValueError(f"Error obtaining parameter value during calculation: {e}")


def safe_eval(expr: str) -> Any:
    """
    Safely evaluate a numeric expression from a string, allowing literal numbers,
    basic arithmetic operators, and functions from numpy and scipy.

    Parameters
    ----------
        expr: The expression to evaluate (e.g. "1/10" or "np.exp(-2) + 3 * np.sqrt(4)").

    Returns
    -------
        The result of evaluating the expression. Depending on the expression,
        this may be one of:
            - A Python numeric type: int, float, or complex.
            - A NumPy scalar (e.g. numpy.int64, numpy.float64).
            - A NumPy ndarray.
            - A SciPy sparse matrix (subclass of scipy.sparse.spmatrix).

    Raises
    ------
        ValueError: If the expression contains disallowed operations or syntax.
        SyntaxError: If the expression has invalid Python syntax.

    """
    # Parse into an AST
    tree = ast.parse(expr, mode="eval")

    # Validate AST nodes
    SafeEvalVisitor().visit(tree)

    # Compile and evaluate with restricted globals
    code = compile(tree, filename="<safe_eval>", mode="eval")
    return eval(code, {"__builtins__": None, "np": np, "scipy": scipy}, {})
