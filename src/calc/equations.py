import cexprtk
import numpy as np

from typing import List

from src.view.plot import SpaceMetadata, PlotMetadata


class Equation:
    def __init__(self, equation: str):
        self.equation = equation

    def solve_point(self, x: float, y: float) -> List[float]:
        pass

    def solve(self, space_metadata: SpaceMetadata) -> PlotMetadata:
        x, y = space_metadata.get_grid()
        z = np.empty((x.shape[0], x.shape[1]), dtype=float)
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
