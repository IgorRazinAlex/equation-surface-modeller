import numpy as np
from typing import Union


class Variable:
    def __init__(self, vname: str):
        self.vname = vname

    def name(self) -> str:
        return self.vname


class VariablePoint:
    def __init__(self):
        self.values = {}

    def set(self, var: Variable, value: np.double):
        self.values[var] = value

    def get(self, var: Variable) -> Union[np.double, None]:
        return self.values.get(var, None)


class Function:
    def eval(self, evaluationPoints: list[VariablePoint]) -> np.ndarray[np.double]:
        pass


class ConstantFunction(Function):
    def __init__(self, value: np.double):
        self.value = value

    def eval(self, evaluationPoints: list[VariablePoint]) -> np.ndarray[np.double]:
        return np.full(len(evaluationPoints), self.value, dtype=np.double)


class VariableFunction(Function):
    def __init__(self, variable: Variable):
        self.variable = variable

    def eval(self, evaluationPoints: list[VariablePoint]) -> np.ndarray[np.double]:
        result = np.zeros(len(evaluationPoints), dtype=np.double)

        for i in range(len(evaluationPoints)):
            value = evaluationPoints[i].get(self.variable)
            result[i] = value if value is not None else np.nan

        return result


class AdditionFunction(Function):
    def __init__(self, term1: Function, term2: Function):
        self.term1 = term1
        self.term2 = term2

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        term1Evaluations = self.term1.eval(variablePoints)
        term2Evaluations = self.term2.eval(variablePoints)

        return np.add(term1Evaluations, term2Evaluations)


class SubstractionFunction(Function):
    def __init__(self, minuend: Function, subtrahend: Function):
        self.minuend = minuend
        self.subtrahend = subtrahend

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        minuendEvaluations = self.minuend.eval(variablePoints)
        subtrahendEvaluations = self.subtrahend.eval(variablePoints)

        return np.subtract(minuendEvaluations, subtrahendEvaluations)


class MultiplicationFunction(Function):
    def __init__(self, term1: Function, term2: Function):
        self.term1 = term1
        self.term2 = term2

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        term1Evaluations = self.term1.eval(variablePoints)
        term2Evaluations = self.term2.eval(variablePoints)

        return np.multiply(term1Evaluations, term2Evaluations)


class DivisionFunction(Function):
    def __init__(self, dividend: Function, divisor: Function):
        self.dividend = dividend
        self.divisor = divisor

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
        dividendEvaluations = self.dividend.eval(variablePoints)
        divisorEvaluations = self.divisor.eval(variablePoints)

        return np.divide(
            dividendEvaluations,
            divisorEvaluations,
            where=(divisorEvaluations != 0),
            out=np.full_like(dividendEvaluations, np.nan),
        )


class PowerFunction(Function):
    def __init__(self, base: Function, index: Function):
        self.base = base
        self.index = index

    def eval(self, variablePoints: list[VariablePoint]) -> np.ndarray[np.double]:
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
