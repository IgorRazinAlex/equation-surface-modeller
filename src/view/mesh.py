import numpy as np
import mcubes
from abc import ABC, abstractmethod
from typing import Tuple
import struct

from src.view.plot import SpaceMetadata, LineMetadata
from src.calc.equations import ExplicitSurfaceEquation, sanitize_equation
from src.calc.dual_contouring import DualContouringMesh


class SurfaceMesh(ABC):
    def __init__(self, equation: str, parameters_metadata):
        self.equation = sanitize_equation(equation)
        self.parameters_metadata = parameters_metadata
 
    @abstractmethod
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
 
    def get_equation(self) -> str:
        return self.equation
 
    def get_parameters_metadata(self):
        return self.parameters_metadata


class ExplicitSurfaceMesh(SurfaceMesh):
    def __init__(self, equation: str, parameters_metadata):
        super().__init__(equation, parameters_metadata)
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
        plot_metadata = self.equation_solver.solve(self.parameters_metadata)

        if np.all(np.isnan(plot_metadata.z)):
            raise ValueError("No valid points generated for the surface")

        nx, ny = plot_metadata.z.shape
        x_axis, y_axis = self.parameters_metadata.get_axes()
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
        X, Y, Z = self.parameters_metadata.get_grid_3d()
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
        grid_info = self.parameters_metadata.get_grid_3d_with_info()
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
        parameters_metadata,
        discontinuity_threshold: float = 10.0,
    ):
        super().__init__(equation, parameters_metadata)
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
        x_axis, y_axis = self.parameters_metadata.get_axes()
        z_axis = self.parameters_metadata.get_z_axis()
        func = self._build_func()

        dc = DualContouringMesh(
            discontinuity_threshold=self.discontinuity_threshold,
        )
        vertices, triangles = dc.generate(func, x_axis, y_axis, z_axis)
        return vertices, triangles


class ParametricCurveMesh(SurfaceMesh):
    def __init__(self, equation: str, parameters_metadata):
        super().__init__(equation, parameters_metadata)
        # Validate that parameters_metadata is LineMetadata
        if not isinstance(parameters_metadata, LineMetadata):
            raise TypeError(
                f"ParametricCurveMesh requires LineMetadata, got {type(parameters_metadata)}"
            )

    def _evaluate_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate parametric equations x(t), y(t), z(t)"""
        t_axis = self.parameters_metadata.get_t_axis()
        eq = self.equation
        
        # Parse equation - expected format: "x = expr_x, y = expr_y, z = expr_z"
        # or "expr_x, expr_y, expr_z" where each is a function of t
        import re
        # Simple parsing: split by commas and expect three expressions
        parts = [p.strip() for p in eq.split(',')]
        if len(parts) != 3:
            raise ValueError(
                "Parametric equation must have three components separated by commas: "
                "x(t), y(t), z(t) or x=..., y=..., z=..."
            )
        
        x_expr = parts[0]
        y_expr = parts[1]
        z_expr = parts[2]
        
        # Remove 'x=', 'y=', 'z=' if present
        for i, expr in enumerate([x_expr, y_expr, z_expr]):
            if '=' in expr:
                expr = expr.split('=')[1].strip()
                if i == 0:
                    x_expr = expr
                elif i == 1:
                    y_expr = expr
                else:
                    z_expr = expr
        
        # Evaluate each expression
        x_vals = np.zeros_like(t_axis)
        y_vals = np.zeros_like(t_axis)
        z_vals = np.zeros_like(t_axis)
        
        for i, t in enumerate(t_axis):
            env = {k: v for k, v in np.__dict__.items()
                   if callable(v) or isinstance(v, (int, float, np.number))}
            env.update({'t': t})
            
            try:
                x_vals[i] = eval(x_expr, {"__builtins__": {}}, env)
                y_vals[i] = eval(y_expr, {"__builtins__": {}}, env)
                z_vals[i] = eval(z_expr, {"__builtins__": {}}, env)
            except Exception as e:
                raise ValueError(
                    f"Error evaluating parametric equation at t={t}: {e}"
                )
        
        return x_vals, y_vals, z_vals

    def _generate_tube_mesh(
        self,
        curve_points: np.ndarray,
        tangents: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate tube mesh around curve points"""
        radius = self.parameters_metadata.radius
        segments = self.parameters_metadata.segments
        
        n_points = len(curve_points)
        if n_points < 2:
            raise ValueError("Curve must have at least 2 points")
        
        # Compute normals and binormals
        normals = np.zeros((n_points, 3))
        binormals = np.zeros((n_points, 3))
        
        # Initial normal (arbitrary perpendicular to first tangent)
        t0 = tangents[0]
        if abs(t0[0]) < 0.1 and abs(t0[1]) < 0.1:
            normals[0] = np.array([1, 0, 0])
        else:
            normals[0] = np.array([-t0[1], t0[0], 0])
            normals[0] /= np.linalg.norm(normals[0])
        
        binormals[0] = np.cross(tangents[0], normals[0])
        binormals[0] /= np.linalg.norm(binormals[0])
        
        # Propagate frame along curve
        for i in range(1, n_points):
            prev_t = tangents[i-1]
            curr_t = tangents[i]
            
            # Simple parallel transport: project previous normal onto plane perpendicular to current tangent
            # and renormalize
            prev_normal = normals[i-1]
            # Project onto plane perpendicular to curr_t
            proj = prev_normal - np.dot(prev_normal, curr_t) * curr_t
            proj_norm = np.linalg.norm(proj)
            if proj_norm > 1e-6:
                normals[i] = proj / proj_norm
            else:
                # If projection is zero, choose arbitrary perpendicular vector
                if abs(curr_t[0]) < 0.1 and abs(curr_t[1]) < 0.1:
                    normals[i] = np.array([1, 0, 0])
                else:
                    normals[i] = np.array([-curr_t[1], curr_t[0], 0])
                    normals[i] /= np.linalg.norm(normals[i])
            
            # Re-orthogonalize
            normals[i] -= np.dot(normals[i], curr_t) * curr_t
            if np.linalg.norm(normals[i]) > 1e-6:
                normals[i] /= np.linalg.norm(normals[i])
            else:
                # Fallback
                if abs(curr_t[0]) < 0.1 and abs(curr_t[1]) < 0.1:
                    normals[i] = np.array([1, 0, 0])
                else:
                    normals[i] = np.array([-curr_t[1], curr_t[0], 0])
                    normals[i] /= np.linalg.norm(normals[i])
            
            binormals[i] = np.cross(curr_t, normals[i])
            binormals[i] /= np.linalg.norm(binormals[i])
        
        # Generate vertices
        vertices = []
        for i in range(n_points):
            center = curve_points[i]
            for j in range(segments):
                angle = 2 * np.pi * j / segments
                offset = radius * (np.cos(angle) * normals[i] +
                                  np.sin(angle) * binormals[i])
                vertex = center + offset
                vertices.append(vertex)
        
        vertices = np.array(vertices)
        
        # Generate triangles
        triangles = []
        for i in range(n_points - 1):
            for j in range(segments):
                next_j = (j + 1) % segments
                
                v0 = i * segments + j
                v1 = i * segments + next_j
                v2 = (i + 1) * segments + j
                v3 = (i + 1) * segments + next_j
                
                triangles.append([v0, v1, v2])
                triangles.append([v1, v3, v2])
        
        return vertices, np.array(triangles, dtype=np.int32)

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        # Evaluate curve
        x_vals, y_vals, z_vals = self._evaluate_curve()
        curve_points = np.column_stack([x_vals, y_vals, z_vals])
        
        # Compute tangents (finite differences)
        tangents = np.zeros_like(curve_points)
        tangents[0] = curve_points[1] - curve_points[0]
        for i in range(1, len(curve_points) - 1):
            tangents[i] = curve_points[i+1] - curve_points[i-1]
        tangents[-1] = curve_points[-1] - curve_points[-2]
        
        # Normalize tangents
        norms = np.linalg.norm(tangents, axis=1)
        norms[norms == 0] = 1
        tangents = tangents / norms[:, np.newaxis]
        
        # Generate tube mesh
        vertices, triangles = self._generate_tube_mesh(curve_points, tangents)
        
        return vertices, triangles


def create_mesh(
    equation: str,
    equation_type: str,
    parameters_metadata,
    algorithm: str = 'marching_cubes',
    discontinuity_threshold: float = 10.0,
) -> SurfaceMesh:
    eq_type = equation_type.lower()
    algo = algorithm.lower()

    if eq_type == 'explicit':
        return ExplicitSurfaceMesh(equation, parameters_metadata)

    elif eq_type == 'implicit':
        if algo == 'dual_contouring':
            return ImplicitSurfaceDualContouringMesh(
                equation,
                parameters_metadata,
                discontinuity_threshold=discontinuity_threshold,
            )
        elif algo == 'marching_cubes':
            return ImplicitSurfaceMesh(equation, parameters_metadata)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    elif eq_type == 'parametric':
        # Parametric curve mesh
        return ParametricCurveMesh(equation, parameters_metadata)
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
