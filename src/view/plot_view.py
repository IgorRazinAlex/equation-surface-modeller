import pyvista as pv
import numpy as np
from src.view.plot import SpaceMetadata


class MeshVisualizer:
    def __init__(self, title: str = "3D Surface Visualization"):
        self.title = title
        self.plotter = pv.Plotter()
        
    def prepare_mesh_data(self, vertices: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
        if len(vertices) == 0:
            raise ValueError("No vertices to visualize!")

        padding = np.full((triangles.shape[0], 1), 3, dtype=int)
        faces_padded = np.hstack((padding, triangles))
        faces_flat = faces_padded.flatten()

        return pv.PolyData(vertices, faces_flat)

    @staticmethod
    def clip(mesh: pv.PolyData, space_metadata: SpaceMetadata) -> pv.PolyData:
        mesh = mesh.clip(normal='z', origin=(0, 0, space_metadata.z_min), invert=False)
        mesh = mesh.clip(normal='-z', origin=(0, 0, space_metadata.z_max), invert=False)

        return mesh

    def visualize(self,
                  space_metadata: SpaceMetadata,
                  vertices: np.ndarray, 
                  triangles: np.ndarray,
                  equation_str: str = "",
                  color: str = 'orange',
                  opacity: float = 1.0,
                  show_edges: bool = False,
                  smooth_shading: bool = True,
                  specular: float = 0.5,
                  show_bounds: bool = True,
                  show_axes: bool = True) -> None:
        try:
            mesh = self.prepare_mesh_data(vertices, triangles)
        except ValueError as e:
            print(f"Error preparing mesh: {e}")
            return
        
        mesh = self.clip(mesh, space_metadata)

        self.plotter.add_mesh(
            mesh, 
            color=color, 
            specular=specular,
            smooth_shading=smooth_shading,
            opacity=opacity,
            show_edges=show_edges
        )

        if show_bounds:
            self.plotter.add_bounding_box()
        if show_axes:
            self.plotter.add_axes()
        if equation_str:
            self.plotter.add_text(equation_str, position='upper_left', font_size=10)

        self.plotter.show(title=self.title)

    def close(self):
        self.plotter.close()


class MultiMeshVisualizer:
    def __init__(self, title: str = "Multiple Surfaces Visualization"):
        self.title = title
        self.plotter = pv.Plotter()
        self.meshes = []
        
    def add_mesh(self,
                 space_metadata: SpaceMetadata,
                 vertices: np.ndarray, 
                 triangles: np.ndarray,
                 equation_str: str = "",
                 color: str = None,
                 opacity: float = 1.0,
                 show_edges: bool = False,
                 smooth_shading: bool = True,
                 specular: float = 0.5) -> None:
        visualizer = MeshVisualizer()
        try:
            mesh = visualizer.prepare_mesh_data(vertices, triangles)
            self.meshes.append((mesh, equation_str))

            mesh = self.clip(mesh, space_metadata)

            self.plotter.add_mesh(
                mesh, 
                color=color, 
                specular=specular,
                smooth_shading=smooth_shading,
                opacity=opacity,
                show_edges=show_edges
            )
        except ValueError as err:
            print(f"Error adding mesh: {err}")
    
    def show(self, show_bounds: bool = True, show_axes: bool = True) -> None:
        if show_bounds:
            self.plotter.add_bounding_box()
        if show_axes:
            self.plotter.add_axes()
        
        if self.meshes:
            for i, (_, eq_str) in enumerate(self.meshes):
                if eq_str:
                    self.plotter.add_text(f"{i+1}: {eq_str}", 
                                         position='upper_left', 
                                         font_size=8)

        self.plotter.show(title=self.title)
    
    def close(self):
        self.plotter.close()
