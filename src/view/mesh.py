import numpy as np
import mcubes
from abc import ABC, abstractmethod
from typing import Tuple
import struct

from src.view.plot import SpaceMetadata
from src.calc.equations import ExplicitSurfaceEquation, sanitize_equation
from src.calc.dual_contouring import DualContouringMesh


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

    def _filter_valid_points(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        if not valid_vertices:
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
                vertices[i * ny + j] = [
                    x_axis[i], y_axis[j], plot_metadata.z[i, j]
                ]

        all_triangles = self._create_triangles(nx, ny)
        vertices, triangles = self._filter_valid_points(
            vertices, all_triangles, valid_mask
        )

        if len(vertices) == 0 or len(triangles) == 0:
            raise ValueError(
                "Could not generate mesh: no valid triangles after filtering"
            )

        return vertices, triangles


class ImplicitSurfaceMesh(SurfaceMesh):
    def _build_func(self):
        eq = self.equation
 
        def func(X, Y, Z):
            env = {
                k: v
                for k, v in np.__dict__.items()
                if callable(v) or isinstance(v, (int, float, np.number))
            }
            env.update({'x': X, 'y': Y, 'z': Z})
            volume = eval(eq, {"__builtins__": {}}, env)
            if not isinstance(volume, np.ndarray):
                volume = np.full(X.shape, float(volume))
            return volume
 
        return func

    def _evaluate_volume(self) -> np.ndarray:
        X, Y, Z = self.space_metadata.get_grid_3d()
        func = self._build_func()

        try:
            volume = func(X, Y, Z)
            if volume.shape != X.shape:
                raise ValueError(
                    f"Function returned array of shape {volume.shape}, "
                    f"expected {X.shape}"
                )
            if not np.all(np.isfinite(volume)):
                volume = np.nan_to_num(
                    volume, nan=1e6, posinf=1e6, neginf=-1e6
                )
            return volume
        except Exception as e:
            raise ValueError(
                f"Could not evaluate function '{self.equation}': {e}"
            )

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        grid_info = self.space_metadata.get_grid_3d_with_info()
        volume = self._evaluate_volume()

        try:
            vertices_idx, triangles = mcubes.marching_cubes(volume, 0.0)
        except Exception as err:
            raise RuntimeError(
                f"Could not generate mesh: marching_cubes error: {err}"
            )

        if len(vertices_idx) == 0:
            raise ValueError("Could not generate mesh: no surface found")

        vertices = np.zeros_like(vertices_idx, dtype=float)
        vertices[:, 0] = (
            grid_info['x_min']
            + (vertices_idx[:, 0] / (grid_info['x_res'] - 1))
            * (grid_info['x_max'] - grid_info['x_min'])
        )
        vertices[:, 1] = (
            grid_info['y_min']
            + (vertices_idx[:, 1] / (grid_info['y_res'] - 1))
            * (grid_info['y_max'] - grid_info['y_min'])
        )
        vertices[:, 2] = (
            grid_info['z_min']
            + (vertices_idx[:, 2] / (grid_info['z_res'] - 1))
            * (grid_info['z_max'] - grid_info['z_min'])
        )

        return vertices, triangles


class ImplicitSurfaceDualContouringMesh(SurfaceMesh):
    def __init__(
        self,
        equation: str,
        space_metadata: SpaceMetadata,
        discontinuity_threshold: float = 10.0,
    ):
        super().__init__(equation, space_metadata)
        self.discontinuity_threshold = discontinuity_threshold

    def _build_func(self):
        eq = self.equation

        def func(X, Y, Z):
            env = {
                k: v
                for k, v in np.__dict__.items()
                if callable(v) or isinstance(v, (int, float, np.number))
            }
            env.update({'x': X, 'y': Y, 'z': Z})
            result = eval(eq, {"__builtins__": {}}, env)
            if not isinstance(result, np.ndarray):
                result = np.full(X.shape, float(result))
            return result.astype(float)

        return func

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        x_axis, y_axis = self.space_metadata.get_axes()
        z_axis = self.space_metadata.get_z_axis()
        func = self._build_func()

        dc = DualContouringMesh(
            discontinuity_threshold=self.discontinuity_threshold,
        )
        vertices, triangles = dc.generate(func, x_axis, y_axis, z_axis)
        return vertices, triangles


def create_mesh(
    equation: str,
    equation_type: str,
    space_metadata: SpaceMetadata,
    algorithm: str = 'marching_cubes',
    discontinuity_threshold: float = 10.0,
) -> SurfaceMesh:
    eq_type = equation_type.lower()
    algo = algorithm.lower()

    if eq_type == 'explicit':
        return ExplicitSurfaceMesh(equation, space_metadata)

    elif eq_type == 'implicit':
        if algo == 'dual_contouring':
            return ImplicitSurfaceDualContouringMesh(
                equation,
                space_metadata,
                discontinuity_threshold=discontinuity_threshold,
            )
        elif algo == 'marching_cubes':
            return ImplicitSurfaceMesh(equation, space_metadata)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    else:
        raise ValueError(f"Unknown equation type: {equation_type}")


def export_to_obj(
    vertices: np.ndarray,
    triangles: np.ndarray,
    filename: str,
    metadata: dict = None
) -> None:
    if vertices is None or len(vertices) == 0:
        raise ValueError("No vertices to export")
    if triangles is None or len(triangles) == 0:
        raise ValueError("No triangles to export")

    with open(filename, 'w', encoding='utf-8') as f:
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")

        f.write("\n# Vertices\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n# Faces\n")
        for t in triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
        
        f.write(f"\n# Total vertices: {len(vertices)}, faces: {len(triangles)}\n")


def export_to_stl(
    vertices: np.ndarray,
    triangles: np.ndarray,
    filename: str,
    ascii_format: bool = False
) -> None:
    if vertices is None or len(vertices) == 0:
        raise ValueError("No vertices to export")
    if triangles is None or len(triangles) == 0:
        raise ValueError("No triangles to export")
    
    if ascii_format:
        _export_to_stl_ascii(vertices, triangles, filename)
    else:
        _export_to_stl_binary(vertices, triangles, filename)


def _export_to_stl_ascii(
    vertices: np.ndarray,
    triangles: np.ndarray,
    filename: str
) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("solid 3d_surface\n")
        
        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0])
            
            f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid 3d_surface\n")


def _export_to_stl_binary(
    vertices: np.ndarray,
    triangles: np.ndarray,
    filename: str
) -> None:
    with open(filename, 'wb') as f:
        header = b"Binary STL generated by 3D Surface Visualizer" + b"\x00" * 40
        f.write(header[:80])

        f.write(struct.pack('<I', len(triangles)))

        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0])

            f.write(struct.pack('<fff', normal[0], normal[1], normal[2]))

            f.write(struct.pack('<fff', v0[0], v0[1], v0[2]))
            f.write(struct.pack('<fff', v1[0], v1[1], v1[2]))
            f.write(struct.pack('<fff', v2[0], v2[1], v2[2]))

            f.write(struct.pack('<H', 0))
