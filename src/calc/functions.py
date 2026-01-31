import numpy as np

from typing import Dict, Union


class Variable:
    '''
    Class to store distinct variable name

    Attributes:
        vname (str): Name of the variable
    '''

    def __init__(self, vname: str):
        '''
        Initialize a Variable object
        
        Args:
            vname (str): Name of variable
        '''
        self.vname = vname

    def name(self) -> str:
        '''
        Return name of variable

        Args:
            None

        Returns:
            str: Name of variable
        '''
        return self.vname


class VariablePoint:
    '''
    Class to store variable and it`s value

    Attributes:
        values (Dict[Variable, np.double]): Dictionary that stores pair (variable, value)
    '''
    def __init__(self):
        '''
        Initialize a VariablePoint object

        Args:
            None
        '''
        self.values: Dict[Variable, np.double] = {}

    def set(self, var: Variable, value: np.double) -> None:
        '''
        Set value for a variable

        Args:
            var (Variable): Variable to set value for
            value (np.double): Value of a variable
        '''
        self.values[var] = value

    def get(self, var: Variable) -> Union[np.double, None]:
        '''
        Get value of a variable
        
        Args:
            var (Variable): Variable to get value of
        
        Returns:
            Union[np.double, None]: Value of variable if exists; otherwise None
        '''
        return self.values.get(var, None)


class Function:
    '''
    Interface for functions

    All inheireted classes should implement `eval` method, which evaluates function over
    given list of VariablePoint
    '''
    def eval(self, evaluationPoints: list[VariablePoint]) -> np.ndarray[np.double]:
        '''
        Dummy function for evaluating values over list of VariablePoint. Inheireted
        functions should implement this method correctly. This means:
        * output array size should equal input list size
        * if any calculation errors occur, corresponding value should be set to np.nan
        This supports correctness of computations from leaves of computation tree all 
        way to the root

        Note that for correct behaviour, all variables used in function should be listed in
        every VariablePoint, otherwise computation tree will return np.nan for every 
        calculation because of VariableFunction behaviour

        Args:
            evaluationPoints (list[VariablePoint]): List of VariablePoint to calculate over
        
        Returns:
            np.ndarray[np.double]: One-dimensional array of values on same indexes as their
            ValuePoint used to calculate them
        '''
        return np.full_like(evaluationPoints, np.nan)


class ConstantFunction(Function):
    '''
    Function for constant function

    Attributes:
        value (np.double): Value for constant function
    '''
    def __init__(self, value: np.double):
        '''
        Initialize a ConstantFunction object

        Args:
            value (np.double): Value for constant function
        '''
        self.value = value

    def eval(self, evaluationPoints: list[VariablePoint]) -> np.ndarray[np.double]:
        '''
        Evaluate ConstantFunction

        Args:
            evaluationPoints (list[VariablePoint]): List of VariablePoint to calculate over
        
        Returns:
            np.ndarray[np.double]: One-dimensional array of constant values
        '''
        return np.full(len(evaluationPoints), self.value, dtype=np.double)


class VariableFunction(Function):
    '''
    Function for single variable projection functions

    Attributes:
        value (Variable): Variable to project to
    '''
    def __init__(self, variable: Variable):
        '''
        Initialize a VariableFunction object

        Args:
            variable (Variable):  Variable to project to
        '''
        self.variable = variable

    def eval(self, evaluationPoints: list[VariablePoint]) -> np.ndarray[np.double]:
        '''
        Evaluate VariableFunction

        Args:
            evaluationPoints (list[VariablePoint]): List of VariablePoint to calculate over
        
        Returns:
            np.ndarray[np.double]: One-dimensional array of variable values (projection from
            VariablePoint to given Variable)
        '''
        result = np.zeros(len(evaluationPoints), dtype=np.double)

        for i in range(len(evaluationPoints)):
            value = evaluationPoints[i].get(self.variable)
            result[i] = value if value is not None else np.nan

        return result


class AdditionFunction(Function):
    '''
    Function for addition

    Attributes:
        term1 (Function): First term of sum
        term2 (Function): Second term of sum
    '''
    def __init__(self, term1: Function, term2: Function):
        '''
        Initialize a AdditionFunction object

        Args:
            term1 (Function): First term of sum
            term2 (Function): Second term of sum
        '''
        self.term1 = term1
        self.term2 = term2

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        '''
        Evaluate AdditionFunction

        Args:
            evaluationPoints (list[VariablePoint]): List of VariablePoint to calculate over
        
        Returns:
            np.ndarray[np.double]: One-dimensional array of sum of two arrays
        '''
        term1Evaluations = self.term1.eval(variablePoints)
        term2Evaluations = self.term2.eval(variablePoints)

        return np.add(term1Evaluations, term2Evaluations)


class SubstractionFunction(Function):
    '''
    Function for substraction

    Attributes:
        minuend (Function): Minuend of substraction
        substrahend (Function): Substrahend of substraction
    '''
    def __init__(self, minuend: Function, subtrahend: Function):
        '''
        Initialize a SubstractionFunction object

        Args:
            minuend (Function): Minuend of substraction
            substrahend (Function): Substrahend of substraction
        '''
        self.minuend = minuend
        self.subtrahend = subtrahend

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        '''
        Evaluate SubstractionFunction

        Args:
            evaluationPoints (list[VariablePoint]): List of VariablePoint to calculate over
        
        Returns:
            np.ndarray[np.double]: One-dimensional array of difference of two arrays
        '''
        minuendEvaluations = self.minuend.eval(variablePoints)
        subtrahendEvaluations = self.subtrahend.eval(variablePoints)

        return np.subtract(minuendEvaluations, subtrahendEvaluations)


class MultiplicationFunction(Function):
    '''
    Function for multiplication

    Attributes:
        term1 (Function): First term of multiplication
        term2 (Function): Second term of multiplication
    '''
    def __init__(self, term1: Function, term2: Function):
        '''
        Initialize a MultiplicationFunction object

        Args:
            term1 (Function): First term of multiplication
            term2 (Function): Second term of multiplication
        '''
        self.term1 = term1
        self.term2 = term2

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        '''
        Evaluate MultiplicationFunction

        Args:
            evaluationPoints (list[VariablePoint]): List of VariablePoint to calculate over
        
        Returns:
            np.ndarray[np.double]: One-dimensional array of product of two arrays
        '''
        term1Evaluations = self.term1.eval(variablePoints)
        term2Evaluations = self.term2.eval(variablePoints)

        return np.multiply(term1Evaluations, term2Evaluations)


class DivisionFunction(Function):
    '''
    Function for division

    Attributes:
        dividend (Function): Dividend of division
        divisor (Function): Divisor of division
    '''
    def __init__(self, dividend: Function, divisor: Function):
        '''
        Initialize a DivisionFunction object

        Args:
            dividend (Function): Dividend of division
            divisor (Function): Divisor of division
        '''
        self.dividend = dividend
        self.divisor = divisor

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        '''
        Evaluate DivisionFunction

        Args:
            evaluationPoints (list[VariablePoint]): List of VariablePoint to calculate over
        
        Returns:
            np.ndarray[np.double]: One-dimensional array of quotient of two arrays
        '''
        dividendEvaluations = self.dividend.eval(variablePoints)
        divisorEvaluations = self.divisor.eval(variablePoints)

        return np.divide(
            dividendEvaluations,
            divisorEvaluations,
            where=(divisorEvaluations != 0),
            out=np.full_like(dividendEvaluations, np.nan),
        )


class PowerFunction(Function):
    '''
    Function for power

    Attributes:
        base (Function): Base of power
        index (Function): Index of power
    '''
    def __init__(self, base: Function, index: Function):
        '''
        Initialize a PowerFunction object

        Args:
            base (Function): Base of power
            index (Function): Index of power
        '''
        self.base = base
        self.index = index

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        '''
        Evaluate PowerFunction

        Args:
            evaluationPoints (list[VariablePoint]): List of VariablePoint to calculate over
        
        Returns:
            np.ndarray[np.double]: One-dimensional array of power of two arrays
        '''
        baseEvaluations = self.base.eval(variablePoints)
        indexEvaluations = self.index.eval(variablePoints)

        result = np.full_like(baseEvaluations, np.nan)
        for i in range(len(baseEvaluations)):
            baseValue = baseEvaluations[i]
            indexValue = indexEvaluations[i]

            if np.isnan(baseValue) or np.isnan(indexValue):
                continue

            try:
                if not (baseValue < 0 and not indexValue.is_integer()):
                    result[i] = np.power(baseValue, indexValue)
            except (ValueError, OverflowError):
                result[i] = np.nan

        return result
