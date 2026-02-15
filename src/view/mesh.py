import numpy as np
import mcubes
import pyvista as pv

from src.view.plot import SpaceMetadata


class SurfaceMesh:
    def __init__(self, equation: str, space_metadata: SpaceMetadata):
        self.equation = equation
        self.space_metadata = space_metadata

    def generate_mesh(self):
        pass


class ExplicitSurfaceMesh(SurfaceMesh):
    '''
    решаем уравнение для кажого (x, y) из дискретной сетки, строим мэш
    '''
    def __init__(self, equation: str, space_metadata: SpaceMetadata):
        pass


class ImplicitSurfaceMesh(SurfaceMesh):
    '''
    не решаем уравнения для F(x, y, z) = 0 напрямую, а используем метод marching cubes
    '''
    def __init__(self, equation: str, space_metadata: SpaceMetadata, bounds=((-1, 1), (-1, 1), (-1, 1)), resolution=100):
        super().__init__(self._sanitize_equation(equation), space_metadata)
        self.bounds = bounds
        self.resolution = resolution

    @staticmethod
    def _sanitize_equation(eq):
        return eq.replace('^', '**')

    def _evaluate_volume(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self.space_metadata.get_bounds()

        x = np.linspace(x_min, x_max, self.resolution)
        y = np.linspace(y_min, y_max, self.resolution)
        z = np.linspace(z_min, z_max, self.resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Inject standard math functions
        env = {k: v for k, v in np.__dict__.items() if callable(v) or isinstance(v, (int, float, np.number))}
        env.update({'x': X, 'y': Y, 'z': Z})
        
        try:
            vol = eval(self.equation, {"__builtins__": None}, env)
        except Exception as e:
            raise ValueError(f"Could not evaluate function: {e}")
            
        return vol

    def generate_mesh(self):
        """ Returns vertices and triangles (N, 3) """
        volume = self._evaluate_volume()
        verts, triangles = mcubes.marching_cubes(volume, 0.0)

        # Rescaling to real world coordinates
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        z_min, z_max = self.bounds[2]
        
        verts[:, 0] = x_min + (verts[:, 0] / (self.resolution - 1)) * (x_max - x_min)
        verts[:, 1] = y_min + (verts[:, 1] / (self.resolution - 1)) * (y_max - y_min)
        verts[:, 2] = z_min + (verts[:, 2] / (self.resolution - 1)) * (z_max - z_min)
        
        return verts, triangles

    def visualize(self):
        """ Generates and visualizes the mesh using PyVista """
        print(f"Generating mesh for: {self.equation} ...")
        verts, triangles = self.generate_mesh()
        
        if len(verts) == 0:
            print("No surface found at this level!")
            return

        # --- PyVista Data Preparation ---
        # VTK/PyVista expects faces array to be flat: [n_points, p1, p2, p3, n_points, p4...]
        # Since we have triangles, we must prepend '3' to every face.
        
        # 1. Create a column of 3s
        padding = np.full((triangles.shape[0], 1), 3, dtype=int)
        
        # 2. Stack [3] + [v1, v2, v3] -> [[3, v1, v2, v3], ...]
        faces_padded = np.hstack((padding, triangles))
        
        # 3. Flatten to 1D array
        faces_flat = faces_padded.flatten()
        
        # --- Create PyVista Object ---
        mesh = pv.PolyData(verts, faces_flat)
        
        print(f"Mesh created: {mesh.n_points} vertices, {mesh.n_cells} faces.")

        # --- Plotting ---
        plotter = pv.Plotter()
        
        # Add the mesh
        plotter.add_mesh(
            mesh, 
            color='orange', 
            specular=0.5,           # Shininess
            smooth_shading=True,    # Gourad shading
            opacity=1.0,
            show_edges=False        # Set to True to see the wireframe
        )
        
        # Add some context helpers
        plotter.add_bounding_box()
        plotter.add_axes()
        plotter.add_text(self.equation, position='upper_left', font_size=10)
        
        print("Opening visualization window...")
        plotter.show()


if __name__ == "__main__":
    
    # гироид
    gyroid = ImplicitSurfaceMesh(
        equation_str="sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)",
        bounds=((-np.pi*2, np.pi*2), (-np.pi*2, np.pi*2), (-np.pi*2, np.pi*2)),
        resolution=120
    )
    gyroid.visualize()
