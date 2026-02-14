import cexprtk
import numpy as np
import sympy

from typing import List

from src.view.plot import GridMetadata, SpaceMetadata, PlotMetadata


class Equation:
    def __init__(self, equation: str):
        self.equation = equation

    def solve_point(self, x: float, y: float) -> List[float]:
        pass

    def solve(self, space_metadata: SpaceMetadata) -> PlotMetadata:
        x, y = space_metadata.get_grid()
        z = np.empty((x.shape[0], x.shape[1]), dtype=object)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                try:
                    result = self.solve_point(float(x[i, j]), float(y[i, j]))
                    z[i, j] = np.array(list(filter(space_metadata.check_z_bounds, result)), dtype=float)
                except Exception:
                    z[i, j] = np.array([], dtype=float)
        return PlotMetadata(x, y, z)


class ExplicitSurfaceEquation(Equation):
    '''
    z = f(x, y)
    '''
    def __init__(self, equation: str):
        super().__init__(equation)
        self.symbol_table = cexprtk.Symbol_Table({'x': 0.0, 'y': 0.0}, add_constants=True)
        self.expression_obj = cexprtk.Expression(equation, self.symbol_table)

    def solve_point(self, x: float, y: float) -> List[float]:
        self.symbol_table.variables['x'] = x
        self.symbol_table.variables['y'] = y
        return [float(self.expression_obj())]


class ImplicitSurfaceEquation(Equation):
    '''
    F(x, y, z) = 0
    '''

    def __init__(
        self,
        equation: str
    ):
        super().__init__(equation)
        x_sym, y_sym, z_sym = sympy.symbols("x y z", real=True)
        equation_normalized = equation.replace("^", "**")
        expr = sympy.sympify(
            equation_normalized, locals={"x": x_sym, "y": y_sym, "z": z_sym}
        )
        solutions = sympy.solve(expr, z_sym)
        if not isinstance(solutions, (list, tuple)):
            solutions = [solutions]
        self._x_sym = x_sym
        self._y_sym = y_sym
        self._z_solutions = solutions

    def solve_point(self, x: float, y: float) -> List[float]:
        """Return all real z such that F(x, y, z) = 0, optionally in [z_min, z_max]."""
        result: List[float] = []
        for sol in self._z_solutions:
            try:
                val = complex(sol.subs([(self._x_sym, x), (self._y_sym, y)]).evalf())
                if abs(val.imag) < 1e-12 and np.isfinite(val.real):
                    result.append(float(val.real))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        result.sort()
        return result
