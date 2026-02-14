import numpy as np
import pytest

from src.calc.equations import (
    Equation,
    ExplicitSurfaceEquation,
    ImplicitSurfaceEquation,
)
from src.view.plot import PlotMetadata, SpaceMetadata, UniformGridMetadata


class TestEquation:
    """Tests for base Equation class."""

    def test_equation_stores_equation_string(self):
        eq = Equation("x + y")
        assert eq.equation == "x + y"


class TestExplicitSurfaceEquation:
    """Tests for ExplicitSurfaceEquation (z = f(x, y))."""

    def test_creation_with_simple_expression(self):
        eq = ExplicitSurfaceEquation("x + y")
        assert eq.equation == "x + y"

    def test_solve_point_linear(self):
        eq = ExplicitSurfaceEquation("x + y")
        result = eq.solve_point(1.0, 2.0)
        assert isinstance(result, list) and len(result) == 1
        assert result[0] == pytest.approx(3.0)

    def test_solve_point_multiplication(self):
        eq = ExplicitSurfaceEquation("x * y")
        result = eq.solve_point(3.0, 4.0)
        assert float(np.asarray(result).flat[0]) == pytest.approx(12.0)

    def test_solve_point_power(self):
        eq = ExplicitSurfaceEquation("x^2 + y^2")
        result = eq.solve_point(3.0, 4.0)
        assert float(np.asarray(result).flat[0]) == pytest.approx(25.0)

    def test_solve_point_constant(self):
        eq = ExplicitSurfaceEquation("5")
        result = eq.solve_point(10.0, 20.0)
        assert float(np.asarray(result).flat[0]) == pytest.approx(5.0)

    def test_solve_point_division(self):
        eq = ExplicitSurfaceEquation("x / y")
        result = eq.solve_point(6.0, 2.0)
        assert float(np.asarray(result).flat[0]) == pytest.approx(3.0)

    def test_solve_point_uses_cexprtk_constants(self):
        eq = ExplicitSurfaceEquation("pi")
        result = eq.solve_point(0.0, 0.0)
        assert float(np.asarray(result).flat[0]) == pytest.approx(np.pi)

    def test_solve_returns_plot_metadata(self):
        eq = ExplicitSurfaceEquation("x + y")
        grid = UniformGridMetadata(
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            x_points=3,
            y_points=3,
        )
        space = SpaceMetadata(grid)
        result = eq.solve(space)
        assert isinstance(result, PlotMetadata)
        assert result.x.shape == (3, 3)
        assert result.y.shape == (3, 3)
        assert result.z.shape == (3, 3)

    def test_solve_values_match_solve_point(self):
        eq = ExplicitSurfaceEquation("x * y")
        grid = UniformGridMetadata(
            x_range=(0.0, 2.0),
            y_range=(0.0, 2.0),
            x_points=2,
            y_points=2,
        )
        space = SpaceMetadata(grid)
        plot = eq.solve(space)
        print(plot.z)
        x_axis, y_axis = grid.get_axis_markers()
        for i, xv in enumerate(x_axis):
            for j, yv in enumerate(y_axis):
                expected = eq.solve_point(xv, yv)
                cell = plot.z[j, i]
                assert cell.size == 1
                assert float(cell.flat[0]) == pytest.approx(float(np.asarray(expected).flat[0]))

    def test_invalid_expression_raises_on_creation(self):
        with pytest.raises(Exception):
            ExplicitSurfaceEquation("invalid +++ syntax")


class TestImplicitSurfaceEquation:
    """Tests for ImplicitSurfaceEquation F(x, y, z) = 0."""

    def test_creation_and_equation_stored(self):
        eq = ImplicitSurfaceEquation("x^2 + y^2 + z^2 - 1")
        assert eq.equation == "x^2 + y^2 + z^2 - 1"

    def test_solve_point_plane_z_minus_x_minus_y(self):
        # F(x,y,z) = z - x - y = 0  =>  z = x + y; single solution
        eq = ImplicitSurfaceEquation("z - x - y")
        assert eq.solve_point(1.0, 2.0) == [pytest.approx(3.0)]
        assert eq.solve_point(0.0, 0.0) == [pytest.approx(0.0)]
        assert eq.solve_point(-1.0, 1.0) == [pytest.approx(0.0)]

    def test_solve_point_paraboloid_finds_single_root(self):
        # F(x,y,z) = z - x^2 - y^2 = 0  =>  z = x^2 + y^2. Single solution.
        eq = ImplicitSurfaceEquation("z - x^2 - y^2")
        assert eq.solve_point(0.0, 0.0)[0] == pytest.approx(0.0)
        assert eq.solve_point(1.0, 0.0)[0] == pytest.approx(1.0)
        assert eq.solve_point(2.0, 1.0)[0] == pytest.approx(5.0)
        assert len(eq.solve_point(2.0, 1.0)) == 1

    def test_solve_point_sphere(self):
        # x^2 + y^2 + z^2 - 1 = 0. At (0,0) roots z = ±1 (all real solutions).
        eq = ImplicitSurfaceEquation("x^2 + y^2 + z^2 - 1")
        solutions = eq.solve_point(0.0, 0.0)
        assert len(solutions) == 2
        assert sorted(solutions) == pytest.approx([-1.0, 1.0])

    def test_solve_returns_plot_metadata(self):
        eq = ImplicitSurfaceEquation("z - x - y")
        grid = UniformGridMetadata(
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            x_points=3,
            y_points=3,
        )
        space = SpaceMetadata(grid)
        result = eq.solve(space)
        assert isinstance(result, PlotMetadata)
        assert result.x.shape == (3, 3)
        assert result.y.shape == (3, 3)
        assert result.z.shape == (3, 3)

    def test_solve_values_match_solve_point(self):
        eq = ImplicitSurfaceEquation("z - x - y")
        grid = UniformGridMetadata(
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            x_points=2,
            y_points=2,
        )
        space = SpaceMetadata(grid)
        plot = eq.solve(space)
        x_axis, y_axis = grid.get_axis_markers()
        for i, xv in enumerate(x_axis):
            for j, yv in enumerate(y_axis):
                solutions = eq.solve_point(xv, yv)
                actual = plot.z[j, i]
                assert isinstance(actual, np.ndarray)
                np.testing.assert_array_almost_equal(actual, np.array(solutions))

    def test_solve_stores_all_solutions_per_cell(self):
        # Sphere: at (0,0) two z solutions; each grid cell holds array of all z
        eq = ImplicitSurfaceEquation("x^2 + y^2 + z^2 - 1")
        grid = UniformGridMetadata(
            x_range=(-0.5, 0.5),
            y_range=(-0.5, 0.5),
            x_points=3,
            y_points=3,
        )
        space = SpaceMetadata(grid)
        plot = eq.solve(space)
        # Center cell (0, 0) has two solutions; grid index (1,1) is center
        center_z = plot.z[1, 1]
        assert isinstance(center_z, np.ndarray)
        assert len(center_z) == 2
        np.testing.assert_array_almost_equal(np.sort(center_z), [-1.0, 1.0])


class TestEquationEvaluateGrid:
    """Tests for evaluate_grid behavior (shared logic in Equation)."""

    def test_solve_mesh_shape_matches_axis_lengths(self):
        eq = ExplicitSurfaceEquation("1")
        grid = UniformGridMetadata(
            x_range=(0.0, 1.0),
            y_range=(0.0, 2.0),
            x_points=5,
            y_points=7,
        )
        space = SpaceMetadata(grid)
        plot = eq.solve(space)
        assert plot.x.shape == (7, 5)
        assert plot.y.shape == (7, 5)
        assert plot.z.shape == (7, 5)
