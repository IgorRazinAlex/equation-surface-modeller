import numpy as np

from typing import List, Tuple, Union


class GridMetadata:
    def __init__(self):
        self.x_markers = np.array([])
        self.y_markers = np.array([])

    def get_axis_markers(self) -> Tuple[np.array, np.array]:
        return self.x_markers, self.y_markers


class UniformGridMetadata(GridMetadata):
    '''
    равномерно по осям
    '''
    def __init__(
        self,
        x_range: Tuple[float, float], 
        y_range: Tuple[float, float],
        x_points: int, 
        y_points: int
    ):  
        super().__init__()
        self.x_markers = np.linspace(x_range[0], x_range[1], x_points)
        self.y_markers = np.linspace(y_range[0], y_range[1], y_points)


class CustomGridMetadata(GridMetadata):
    '''
    пользовательская
    '''
    def __init__(self, x_values: Union[List, np.ndarray], y_values: Union[List, np.ndarray]):
        super().__init__()
        self.x_markers = np.array(x_values)
        self.y_markers = np.array(y_values)


class SpaceMetadata:
    def __init__(
        self, grid_metadata: GridMetadata,
        x_min: Union[float, None] = None,
        x_max: Union[float, None] = None,
        y_min: Union[float, None] = None,
        y_max: Union[float, None] = None,
        z_min: Union[float, None] = None,
        z_max: Union[float, None] = None
    ):
        self.grid_metadata = grid_metadata
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    
    def get_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        x_axis, y_axis = self.grid_metadata.get_axis_markers()
        return np.meshgrid(x_axis, y_axis)
    
    @staticmethod
    def _check_bounds(number: float, min_number: Union[float, None], max_number: Union[float, None]) -> bool:
        if min_number is not None and min_number > number:
            return False
        if max_number is not None and max_number < number:
            return False
        return True

    def check_x_bounds(self, x: float):
        return self._check_bounds(x, self.x_min, self.x_max)

    def check_y_bounds(self, y: float):
        return self._check_bounds(y, self.y_min, self.y_max)

    def check_z_bounds(self, z: float):
        return self._check_bounds(z, self.z_min, self.z_max)
    
    def get_bounds(self):
        return self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max

class PlotMetadata:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
