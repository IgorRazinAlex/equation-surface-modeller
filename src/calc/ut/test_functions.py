import numpy as np
import pytest
from src.calc.functions import (
    Variable,
    VariablePoint,
    ConstantFunction,
    VariableFunction,
    AdditionFunction,
    SubstractionFunction,
    MultiplicationFunction,
    DivisionFunction,
    PowerFunction,
)


class TestVariable:
    def test_variable_creation(self):
        """Test Variable creation and name method"""
        var = Variable("x")
        assert var.name() == "x"

        var2 = Variable("y")
        assert var2.name() == "y"
        assert var.name() == "x"  # Ensure no side effects


class TestVariablePoint:
    def test_variable_point_operations(self):
        """Test VariablePoint get and set operations"""
        var_x = Variable("x")
        var_y = Variable("y")

        point = VariablePoint()

        # Test get on non-existent variable
        assert point.get(var_x) is None

        # Test set and get
        point.set(var_x, 10.0)
        point.set(var_y, 20.0)

        assert point.get(var_x) == 10.0
        assert point.get(var_y) == 20.0

        # Test updating value
        point.set(var_x, 30.0)
        assert point.get(var_x) == 30.0


class TestVariableFunction:
    def test_variable_function_eval(self):
        """Test evaluation of VariableFunction"""
        var_x = Variable("x")
        func_x = VariableFunction(var_x)

        # Create test points
        points = []
        for i in range(3):
            point = VariablePoint()
            point.set(var_x, float(i))
            points.append(point)

        result = func_x.eval(points)

        expected = np.array([0.0, 1.0, 2.0], dtype=np.double)
        np.testing.assert_array_equal(result, expected)

    def test_variable_function_with_missing_value(self):
        """Test VariableFunction with missing variable"""
        var_x = Variable("x")
        var_y = Variable("y")

        func_x = VariableFunction(var_x)

        # Point has y but not x
        point = VariablePoint()
        point.set(var_y, 5.0)

        result = func_x.eval([point])

        # Should return NaN for missing value
        assert np.isnan(result[0])


class TestArithmeticFunctions:
    @pytest.fixture
    def setup_variables(self):
        """Setup common test variables"""
        self.var_x = Variable("x")
        self.var_y = Variable("y")

        self.func_x = VariableFunction(self.var_x)
        self.func_y = VariableFunction(self.var_y)

        # Create test points: (x, y) = (1, 2), (3, 4), (5, 6)
        self.points = []
        for i in range(3):
            point = VariablePoint()
            point.set(self.var_x, float(2 * i + 1))
            point.set(self.var_y, float(2 * i + 2))
            self.points.append(point)

    def test_addition_function(self, setup_variables):
        """Test AdditionFunction: f(x, y) = x + y"""
        add_func = AdditionFunction(self.func_x, self.func_y)
        result = add_func.eval(self.points)

        expected = np.array([3.0, 7.0, 11.0], dtype=np.double)  # 1+2, 3+4, 5+6
        np.testing.assert_array_almost_equal(result, expected)

    def test_subtraction_function(self, setup_variables):
        """Test SubstractionFunction: f(x, y) = x - y"""
        sub_func = SubstractionFunction(self.func_x, self.func_y)
        result = sub_func.eval(self.points)

        expected = np.array([-1.0, -1.0, -1.0], dtype=np.double)  # 1-2, 3-4, 5-6
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiplication_function(self, setup_variables):
        """Test MultiplicationFunction: f(x, y) = x * y"""
        mul_func = MultiplicationFunction(self.func_x, self.func_y)
        result = mul_func.eval(self.points)

        expected = np.array([2.0, 12.0, 30.0], dtype=np.double)  # 1*2, 3*4, 5*6
        np.testing.assert_array_almost_equal(result, expected)

    def test_division_function_normal(self, setup_variables):
        """Test DivisionFunction with normal values"""
        div_func = DivisionFunction(self.func_x, self.func_y)
        result = div_func.eval(self.points)

        expected = np.array([0.5, 0.75, 5.0 / 6.0], dtype=np.double)  # 1/2, 3/4, 5/6
        np.testing.assert_array_almost_equal(result, expected)

    def test_division_function_by_zero(self):
        """Test DivisionFunction with division by zero"""
        var_x = Variable("x")
        var_y = Variable("y")

        func_x = VariableFunction(var_x)
        func_y = VariableFunction(var_y)
        div_func = DivisionFunction(func_x, func_y)

        # Create points with y = 0 at position 1
        points = []
        for i in range(3):
            point = VariablePoint()
            point.set(var_x, float(i + 1))
            point.set(var_y, 1.0 if i != 1 else 0.0)
            points.append(point)

        result = div_func.eval(points)

        # Should have NaN at position 1
        assert not np.isnan(result[0])  # 1/1 = 1
        assert np.isnan(result[1])  # 2/0 = NaN
        assert not np.isnan(result[2])  # 3/1 = 3

    def test_division_function_with_nan_input(self):
        """Test DivisionFunction with NaN in input"""
        var_x = Variable("x")
        var_y = Variable("y")

        func_x = VariableFunction(var_x)
        func_y = VariableFunction(var_y)
        div_func = DivisionFunction(func_x, func_y)

        # Create points with NaN values
        points = []
        test_cases = [
            (1.0, 2.0),  # normal
            (np.nan, 3.0),  # nan in dividend
            (4.0, np.nan),  # nan in divisor
            (np.nan, 0.0),  # nan / 0
        ]

        for x_val, y_val in test_cases:
            point = VariablePoint()
            point.set(var_x, x_val)
            point.set(var_y, y_val)
            points.append(point)

        result = div_func.eval(points)

        # First case should be normal
        assert result[0] == 0.5

        # All others should be NaN
        assert np.all(np.isnan(result[1:]))

    def test_power_function(self, setup_variables):
        """Test PowerFunction: f(x, y) = x^y"""
        pow_func = PowerFunction(self.func_x, self.func_y)
        result = pow_func.eval(self.points)

        expected = np.array([1**2, 3**4, 5**6], dtype=np.double)  # 1  # 81  # 15625

        np.testing.assert_array_almost_equal(result, expected)

    def test_power_function_with_negative_base(self):
        """Test PowerFunction with negative base and non-integer exponent"""
        var_x = Variable("x")
        var_y = Variable("y")

        func_x = VariableFunction(var_x)
        func_y = VariableFunction(var_y)
        pow_func = PowerFunction(func_x, func_y)

        # Create points: (-4, 0.5) = sqrt(-4) = NaN
        point = VariablePoint()
        point.set(var_x, -4.0)
        point.set(var_y, 0.5)

        result = pow_func.eval([point])

        # Should be NaN for sqrt(-4)
        assert np.isnan(result[0])

    def test_composite_function(self, setup_variables):
        """Test composite function: f(x, y) = (x + y) * (x - y) / 2"""
        # Build: ((x + y) * (x - y)) / 2
        add = AdditionFunction(self.func_x, self.func_y)
        sub = SubstractionFunction(self.func_x, self.func_y)
        mul = MultiplicationFunction(add, sub)

        const_2 = ConstantFunction(2.0)
        div = DivisionFunction(mul, const_2)

        result = div.eval(self.points)

        # Expected: ((x+y)*(x-y))/2 = (x^2 - y^2)/2
        expected = np.array(
            [
                (1**2 - 2**2) / 2,  # (1-4)/2 = -1.5
                (3**2 - 4**2) / 2,  # (9-16)/2 = -3.5
                (5**2 - 6**2) / 2,  # (25-36)/2 = -5.5
            ],
            dtype=np.double,
        )

        np.testing.assert_array_almost_equal(result, expected)


class TestEdgeCases:
    def test_empty_points_list(self):
        """Test evaluation with empty points list"""
        var_x = Variable("x")
        func_x = VariableFunction(var_x)

        result = func_x.eval([])

        # Should return empty array
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_all_nan_points(self):
        """Test with all NaN values"""
        var_x = Variable("x")
        var_y = Variable("y")

        func_x = VariableFunction(var_x)
        func_y = VariableFunction(var_y)
        add_func = AdditionFunction(func_x, func_y)

        # Create points with NaN values
        points = []
        for _ in range(3):
            point = VariablePoint()
            point.set(var_x, np.nan)
            point.set(var_y, np.nan)
            points.append(point)

        result = add_func.eval(points)

        # All results should be NaN
        assert np.all(np.isnan(result))

    def test_large_arrays_performance(self):
        """Test performance with large number of points"""
        var_x = Variable("x")
        func_x = VariableFunction(var_x)

        # Create 10000 points
        n_points = 10000
        points = []
        for i in range(n_points):
            point = VariablePoint()
            point.set(var_x, float(i))
            points.append(point)

        result = func_x.eval(points)

        assert len(result) == n_points
        assert result[0] == 0.0
        assert result[-1] == float(n_points - 1)


# Parametrized tests for more comprehensive coverage
@pytest.mark.parametrize(
    "x_val,y_val,expected",
    [
        (4.0, 2.0, 2.0),  # 4/2 = 2
        (0.0, 5.0, 0.0),  # 0/5 = 0
        (-6.0, 3.0, -2.0),  # -6/3 = -2
        (1.0, 0.0, np.nan),  # 1/0 = NaN
        (0.0, 0.0, np.nan),  # 0/0 = NaN
    ],
)
def test_division_parametrized(x_val, y_val, expected):
    """Parametrized test for DivisionFunction"""
    var_x = Variable("x")
    var_y = Variable("y")

    func_x = VariableFunction(var_x)
    func_y = VariableFunction(var_y)
    div_func = DivisionFunction(func_x, func_y)

    point = VariablePoint()
    point.set(var_x, x_val)
    point.set(var_y, y_val)

    result = div_func.eval([point])[0]

    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)
