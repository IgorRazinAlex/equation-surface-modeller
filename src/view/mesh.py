import numpy as np
import mcubes
from abc import ABC, abstractmethod
from typing import Tuple

from src.view.plot import SpaceMetadata
from src.calc.equations import ExplicitSurfaceEquation, sanitize_equation


class SurfaceMesh(ABC):
    def __init__(self, equation: str, space_metadata: SpaceMetadata):
        self.equation = sanitize_equation(equation)
        self.space_metadata = space_metadata
    
    @abstractmethod
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_equation(self) -> str:
        return self.equation

    def get_space_metadata(self) -> SpaceMetadata:
        return self.space_metadata


class ExplicitSurfaceMesh(SurfaceMesh):
    def __init__(self, equation: str, space_metadata: SpaceMetadata):
        super().__init__(equation, space_metadata)
        self.equation_solver = ExplicitSurfaceEquation(equation)
    
    def _create_triangles(self, nx: int, ny: int) -> np.ndarray:
        triangles = []
        
        for i in range(nx - 1):
            for j in range(ny - 1):
                p0 = i * ny + j
                p1 = i * ny + (j + 1)
                p2 = (i + 1) * ny + j
                p3 = (i + 1) * ny + (j + 1)

                triangles.append([p0, p1, p2])
                triangles.append([p1, p3, p2])
        
        return np.array(triangles, dtype=np.int32)
    
    def _filter_valid_points(self, 
                            vertices: np.ndarray, 
                            triangles: np.ndarray,
                            valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nx, ny = valid_mask.shape
        old_to_new = np.full(nx * ny, -1, dtype=np.int32)

        valid_vertices = []
        new_index = 0
        
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                if valid_mask[i, j]:
                    old_to_new[idx] = new_index
                    valid_vertices.append(vertices[idx])
                    new_index += 1
        
        if len(valid_vertices) == 0:
            return np.array([]), np.array([])
        
        valid_vertices = np.array(valid_vertices)
        valid_triangles = []
        for tri in triangles:
            new_tri = [old_to_new[idx] for idx in tri]
            if all(idx != -1 for idx in new_tri):
                valid_triangles.append(new_tri)
        
        return valid_vertices, np.array(valid_triangles, dtype=np.int32)
    
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        plot_metadata = self.equation_solver.solve(self.space_metadata)

        if np.all(np.isnan(plot_metadata.z)):
            raise ValueError("No valid points generated for the surface")

        nx, ny = plot_metadata.z.shape
        x_axis, y_axis = self.space_metadata.get_axes()

        valid_mask = ~np.isnan(plot_metadata.z)
        
        if not np.any(valid_mask):
            raise ValueError("No valid points to generate mesh")

        vertices = np.zeros((nx * ny, 3), dtype=float)
        
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                vertices[idx] = [
                    x_axis[i],
                    y_axis[j],
                    plot_metadata.z[i, j]
                ]

        all_triangles = self._create_triangles(nx, ny)
        vertices, triangles = self._filter_valid_points(vertices, all_triangles, valid_mask)
        
        if len(vertices) == 0 or len(triangles) == 0:
            raise ValueError("Could not generate mesh: no valid triangles after filtering")
        
        return vertices, triangles


class ImplicitSurfaceMesh(SurfaceMesh):
    def _evaluate_volume(self) -> np.ndarray:
        X, Y, Z = self.space_metadata.get_grid_3d()

        env = {k: v for k, v in np.__dict__.items() if callable(v) or isinstance(v, (int, float, np.number))}
        env.update(
            {
                'x': X,
                'y': Y,
                'z': Z,
            }
        )

        try:
            volume = eval(self.equation, {"__builtins__": {}}, env)

            if volume.shape != X.shape:
                raise ValueError(f"Function returned array of shape {volume.shape}, expected {X.shape}")

            if not np.all(np.isfinite(volume)):
                volume = np.nan_to_num(volume, nan=1e6, posinf=1e6, neginf=-1e6)
            
            return volume
            
        except Exception as e:
            raise ValueError(f"Could not evaluate function '{self.equation}': {e}")
    
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        grid_info = self.space_metadata.get_grid_3d_with_info()

        volume = self._evaluate_volume()
        
        try:
            vertices_idx, triangles = mcubes.marching_cubes(volume, 0.0)
        except Exception as err:
            raise RuntimeError(f"Could not generate mesh: marching_vubes algorithm error: {err}")
        
        if len(vertices_idx) == 0:
            raise ValueError("Could not generate mesh: no surface found")

        vertices = np.zeros_like(vertices_idx, dtype=float)
        vertices[:, 0] = grid_info['x_min'] + (vertices_idx[:, 0] / (grid_info['x_res'] - 1)) * (grid_info['x_max'] - grid_info['x_min'])
        vertices[:, 1] = grid_info['y_min'] + (vertices_idx[:, 1] / (grid_info['y_res'] - 1)) * (grid_info['y_max'] - grid_info['y_min'])
        vertices[:, 2] = grid_info['z_min'] + (vertices_idx[:, 2] / (grid_info['z_res'] - 1)) * (grid_info['z_max'] - grid_info['z_min'])
        
        return vertices, triangles


def create_mesh(
    equation: str,
    equation_type: str,
    space_metadata: SpaceMetadata,
) -> SurfaceMesh:
    if equation_type.lower() == 'explicit':
        return ExplicitSurfaceMesh(equation, space_metadata)
    elif equation_type.lower() == 'implicit':
        return ImplicitSurfaceMesh(equation, space_metadata)
    else:
        raise ValueError(f"Unknown equation type: {equation_type}")
