import numpy as np
from typing import Tuple, Optional
from scipy.linalg import lstsq
 
 
class DualContouringMesh:
    EDGES = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
 
    def __init__(
        self,
        discontinuity_threshold: float = 10.0,
        gradient_eps: float = 1e-4,
        qef_regularization: float = 1e-4,
        max_vertex_displacement: float = 2.0,
    ):
        self.discontinuity_threshold = discontinuity_threshold
        self.gradient_eps = gradient_eps
        self.qef_regularization = qef_regularization
        self.max_vertex_displacement = max_vertex_displacement

    def generate(
        self,
        func,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.func = func
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
 
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)

        X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
        self.F = self._safe_eval_grid(X, Y, Z)

        vertex_grid = np.full((nx - 1, ny - 1, nz - 1), -1, dtype=np.int32)
        vertices = []
 
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    v = self._process_cell(i, j, k)
                    if v is not None:
                        vertex_grid[i, j, k] = len(vertices)
                        vertices.append(v)
 
        if not vertices:
            raise ValueError("Dual Contouring: surface not found")

        triangles = self._build_faces(vertex_grid, nx, ny, nz)
 
        if len(triangles) == 0:
            raise ValueError("Dual Contouring: failed to build faces")
 
        return np.array(vertices, dtype=float), np.array(triangles, dtype=np.int32)
 
    def _safe_eval_grid(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        try:
            F = self.func(X, Y, Z)
            F = np.where(np.isfinite(F), F, np.sign(F + 1e-30) * 1e6)
        except Exception:
            F = np.full(X.shape, 1e6)
        return F.astype(float)
 
    def _eval_point(self, x: float, y: float, z: float) -> float:
        try:
            v = self.func(
                np.array([x]), np.array([y]), np.array([z])
            )
            val = float(v.flat[0])
            return val if np.isfinite(val) else 1e6
        except Exception:
            return 1e6
 
    def _gradient(self, x: float, y: float, z: float) -> np.ndarray:
        eps = self.gradient_eps
        gx = (self._eval_point(x + eps, y, z) - self._eval_point(x - eps, y, z)) / (2 * eps)
        gy = (self._eval_point(x, y + eps, z) - self._eval_point(x, y - eps, z)) / (2 * eps)
        gz = (self._eval_point(x, y, z + eps) - self._eval_point(x, y, z - eps)) / (2 * eps)
        g = np.array([gx, gy, gz])
        norm = np.linalg.norm(g)
        return g / norm if norm > 1e-10 else g
 
    def _cell_corners(self, i: int, j: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.array([
            [self.x_axis[i + dx], self.y_axis[j + dy], self.z_axis[k + dz]]
            for dx in range(2) for dy in range(2) for dz in range(2)
        ])
        values = np.array([
            self.F[i + dx, j + dy, k + dz]
            for dx in range(2) for dy in range(2) for dz in range(2)
        ])
        return coords, values
 
    def _is_discontinuous(self, values: np.ndarray) -> bool:
        if np.any(np.abs(values) > self.discontinuity_threshold):
            return True

        has_positive = np.any(values > 0)
        has_negative = np.any(values < 0)
        if has_positive and has_negative:
            value_range = np.max(values) - np.min(values)
            if value_range > self.discontinuity_threshold * 0.5:
                return True
 
        return False
 
    def _find_edge_crossing(
        self,
        p0: np.ndarray, f0: float,
        p1: np.ndarray, f1: float,
    ) -> Optional[np.ndarray]:
        if f0 * f1 >= 0:
            return None
        t = f0 / (f0 - f1)
        t = np.clip(t, 0.0, 1.0)
        return p0 + t * (p1 - p0)
 
    def _solve_qef(
        self,
        intersection_points: list,
        normals: list,
        cell_min: np.ndarray,
        cell_max: np.ndarray,
    ) -> np.ndarray:
        center = (cell_min + cell_max) * 0.5
 
        if len(intersection_points) == 0:
            return center
 
        A = np.array(normals, dtype=float)
        b = np.array([
            np.dot(n, p)
            for n, p in zip(normals, intersection_points)
        ], dtype=float)

        reg = self.qef_regularization
        A_reg = np.vstack([A, np.eye(3) * reg])
        b_reg = np.concatenate([b, center * reg])
 
        try:
            result, _, _, _ = lstsq(A_reg, b_reg)
        except Exception:
            result = center

        result = np.clip(result, cell_min, cell_max)
        return result
 
    def _process_cell(
        self, i: int, j: int, k: int
    ) -> Optional[np.ndarray]:
        coords, values = self._cell_corners(i, j, k)

        if self._is_discontinuous(values):
            return None

        intersection_points = []
        normals = []
 
        for e0, e1 in self.EDGES:
            pt = self._find_edge_crossing(
                coords[e0], values[e0],
                coords[e1], values[e1],
            )
            if pt is not None:
                g = self._gradient(pt[0], pt[1], pt[2])
                intersection_points.append(pt)
                normals.append(g)
 
        if len(intersection_points) == 0:
            return None

        cell_min = np.array([
            self.x_axis[i], self.y_axis[j], self.z_axis[k]
        ])
        cell_max = np.array([
            self.x_axis[i + 1], self.y_axis[j + 1], self.z_axis[k + 1]
        ])
 
        vertex = self._solve_qef(
            intersection_points, normals, cell_min, cell_max
        )
        return vertex
 
    def _build_faces(
        self,
        vertex_grid: np.ndarray,
        nx: int, ny: int, nz: int,
    ) -> list:
        triangles = []
        for i in range(nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    f0 = self.F[i, j, k]
                    f1 = self.F[i + 1, j, k]
                    if f0 * f1 >= 0:
                        continue
                    c = [
                        vertex_grid[i, j - 1, k - 1],
                        vertex_grid[i, j,     k - 1],
                        vertex_grid[i, j,     k    ],
                        vertex_grid[i, j - 1, k    ],
                    ]
                    if any(v == -1 for v in c):
                        continue
                    if f0 < 0:
                        triangles.append([c[0], c[1], c[2]])
                        triangles.append([c[0], c[2], c[3]])
                    else:
                        triangles.append([c[0], c[2], c[1]])
                        triangles.append([c[0], c[3], c[2]])

        for i in range(1, nx - 1):
            for j in range(ny - 1):
                for k in range(1, nz - 1):
                    f0 = self.F[i, j, k]
                    f1 = self.F[i, j + 1, k]
                    if f0 * f1 >= 0:
                        continue
                    c = [
                        vertex_grid[i - 1, j, k - 1],
                        vertex_grid[i - 1, j, k    ],
                        vertex_grid[i,     j, k    ],
                        vertex_grid[i,     j, k - 1],
                    ]
                    if any(v == -1 for v in c):
                        continue
                    if f0 < 0:
                        triangles.append([c[0], c[2], c[1]])
                        triangles.append([c[0], c[3], c[2]])
                    else:
                        triangles.append([c[0], c[1], c[2]])
                        triangles.append([c[0], c[2], c[3]])

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(nz - 1):
                    f0 = self.F[i, j, k]
                    f1 = self.F[i, j, k + 1]
                    if f0 * f1 >= 0:
                        continue
                    c = [
                        vertex_grid[i - 1, j - 1, k],
                        vertex_grid[i,     j - 1, k],
                        vertex_grid[i,     j,     k],
                        vertex_grid[i - 1, j,     k],
                    ]
                    if any(v == -1 for v in c):
                        continue
                    if f0 < 0:
                        triangles.append([c[0], c[1], c[2]])
                        triangles.append([c[0], c[2], c[3]])
                    else:
                        triangles.append([c[0], c[2], c[1]])
                        triangles.append([c[0], c[3], c[2]])

        return triangles
