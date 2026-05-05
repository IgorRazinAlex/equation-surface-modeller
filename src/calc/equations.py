import numpy as np

from typing import List

from src.view.plot import SpaceMetadata, PlotMetadata


class Equation:
    def __init__(self, equation: str):
        self.equation = equation

    def solve_point(self, x: float, y: float) -> List[float]:
        pass

    def solve(self, space_metadata: SpaceMetadata) -> PlotMetadata:
        x, y = space_metadata.get_grid_2d()
        z = np.full(x.shape, np.nan, dtype=float)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                try:
                    z[i, j] = self.solve_point(float(x[i, j]), float(y[i, j]))
                except Exception:
                    continue
        return PlotMetadata(x, y, z)


class ExplicitSurfaceEquation(Equation):
    def solve_point(self, x: float, y: float) -> float:
        env = {k: v for k, v in np.__dict__.items() if callable(v) or isinstance(v, (int, float, np.number))}
        env.update(
            {
                'x': x,
                'y': y,
            }
        )
        return eval(self.equation, {"__builtins__": {}}, env)


def sanitize_equation(equation: str) -> str:
    return equation.replace('^', '**')