import numpy as np
from typing import Tuple, Optional, Dict


class GridMetadata:
    def __init__(self):
        self.x_markers = np.array([])
        self.y_markers = np.array([])

    def get_axis_markers(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_markers, self.y_markers
    
    def get_resolution(self) -> Tuple[int, int]:
        return len(self.x_markers), len(self.y_markers)


class UniformGridMetadata(GridMetadata):
    def __init__(
        self,
        x_range: Tuple[float, float], 
        y_range: Tuple[float, float],
        x_points: int, 
        y_points: int
    ):  
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.x_points = x_points
        self.y_points = y_points
        self.x_markers = np.linspace(x_range[0], x_range[1], x_points)
        self.y_markers = np.linspace(y_range[0], y_range[1], y_points)


class SpaceMetadata:
    DEFAULT_PARAMETERS = {
        'x_min': -10,
        'x_max': 10,
        'y_min': -10,
        'y_max': 10,
        'z_min': -10,
        'z_max': 10,
        'z_points': 10,
    }

    def __init__(
        self, 
        grid_metadata: GridMetadata,
        x_min: float = DEFAULT_PARAMETERS['x_min'],
        x_max: float = DEFAULT_PARAMETERS['x_max'],
        y_min: float = DEFAULT_PARAMETERS['y_min'],
        y_max: float = DEFAULT_PARAMETERS['y_max'],
        z_min: float = DEFAULT_PARAMETERS['z_min'],
        z_max: float = DEFAULT_PARAMETERS['z_max'],
        z_points: int = DEFAULT_PARAMETERS['z_points']
    ):
        self.grid_metadata = grid_metadata
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.z_points = z_points
        
    def get_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.grid_metadata.get_axis_markers()
    
    def get_resolution_xy(self) -> Tuple[int, int]:
        return self.grid_metadata.get_resolution()
    
    def get_resolution_z(self) -> int:
        return self.z_points
    
    def get_z_axis(self) -> np.ndarray:
        return np.linspace(self.z_min, self.z_max, self.get_resolution_z())
    
    def get_grid_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        x_axis, y_axis = self.get_axes()
        return np.meshgrid(x_axis, y_axis, indexing='ij')
    
    def get_grid_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_axis, y_axis = self.get_axes()
        z_axis = self.get_z_axis()
        return np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
    
    def get_grid_3d_with_info(self) -> Dict:
        X, Y, Z = self.get_grid_3d()
        x_axis, y_axis = self.get_axes()
        z_axis = self.get_z_axis()
        
        return {
            'X': X, 'Y': Y, 'Z': Z,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis,
            'x_res': len(x_axis),
            'y_res': len(y_axis),
            'z_res': len(z_axis),
            'x_min': x_axis[0], 'x_max': x_axis[-1],
            'y_min': y_axis[0], 'y_max': y_axis[-1],
            'z_min': z_axis[0], 'z_max': z_axis[-1]
        }

    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        return (
            self.x_min, self.x_max,
            self.y_min, self.y_max,
            self.z_min, self.z_max
        )

    @staticmethod
    def _check_bounds(number: float, min_val: Optional[float], max_val: Optional[float]) -> bool:
        if min_val is not None and min_val > number:
            return False
        if max_val is not None and max_val < number:
            return False
        return True

    def check_x_bounds(self, x: float) -> bool:
        return self._check_bounds(x, self.x_min, self.x_max)

    def check_y_bounds(self, y: float) -> bool:
        return self._check_bounds(y, self.y_min, self.y_max)

    def check_z_bounds(self, z: float) -> bool:
        return self._check_bounds(z, self.z_min, self.z_max)


class PlotMetadata:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
