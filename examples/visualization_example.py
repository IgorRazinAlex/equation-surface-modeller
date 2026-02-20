import numpy as np
from src.view.plot import SpaceMetadata, UniformGridMetadata
from src.view.mesh import ImplicitSurfaceMesh, ExplicitSurfaceMesh
from src.view.plot_view import MeshVisualizer


def example_gyroid():
    grid = UniformGridMetadata(
        x_range=(-2*np.pi, 2*np.pi),
        y_range=(-2*np.pi, 2*np.pi),
        x_points=60,
        y_points=60
    )

    space = SpaceMetadata(
        grid_metadata=grid,
        z_min=-2*np.pi,
        z_max=2*np.pi,
        z_points=60
    )

    mesh = ImplicitSurfaceMesh(
        equation="sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)",
        space_metadata=space
    )

    vertices, triangles = mesh.generate_mesh()

    visualizer = MeshVisualizer(title="Gyroid")
    visualizer.visualize(space, vertices, triangles, equation_str="Gyroid")
    visualizer.close()


def example_sphere_with_custom_z_resolution():
    grid = UniformGridMetadata(
        x_range=(-3, 3),
        y_range=(-3, 3),
        x_points=40,
        y_points=40
    )

    space = SpaceMetadata(
        grid_metadata=grid,
        z_min=-3,
        z_max=3,
        z_points=80
    )
    
    mesh = ImplicitSurfaceMesh(
        equation="x**2 + y**2 + z**2 - 4",
        space_metadata=space
    )
    
    vertices, triangles = mesh.generate_mesh()
    
    visualizer = MeshVisualizer(title="Sphere with high Z resolution")
    visualizer.visualize(space, vertices, triangles, equation_str="Sphere", color='lightblue')
    visualizer.close()


def example_paraboloid():
    grid = UniformGridMetadata(
        x_range=(-2, 2),
        y_range=(-2, 2),
        x_points=50,
        y_points=50
    )
    
    space = SpaceMetadata(grid_metadata=grid)
    
    mesh = ExplicitSurfaceMesh(
        equation="x**2 + y**2",
        space_metadata=space
    )
    
    vertices, triangles = mesh.generate_mesh()
    
    visualizer = MeshVisualizer(title="Paraboloid")
    visualizer.visualize(
        space,
        vertices, 
        triangles, 
        equation_str="z = x^2 + y^2",
        color='lightgreen',
        show_edges=True
    )
    visualizer.close()

def example_saddle():
    grid = UniformGridMetadata(
        x_range=(-2, 2),
        y_range=(-2, 2),
        x_points=40,
        y_points=40
    )
    
    space = SpaceMetadata(grid_metadata=grid)
    
    mesh = ExplicitSurfaceMesh(
        equation="x**2 - y**2",
        space_metadata=space
    )
    
    vertices, triangles = mesh.generate_mesh()

    visualizer = MeshVisualizer(title="Hyperbolic Paraboloid")
    visualizer.visualize(
        space,
        vertices, 
        triangles, 
        equation_str="z = x^2 - y^2",
        color='coral'
    )
    visualizer.close()


def example_sphere_hemisphere():
    """Пример сферы, обрезанной до полусферы"""
    
    grid = UniformGridMetadata(
        x_range=(-3, 3),
        y_range=(-3, 3),
        x_points=50,
        y_points=50
    )

    space = SpaceMetadata(
        grid_metadata=grid,
        z_min=0,
        z_max=3,
        z_points=50
    )
    
    mesh = ImplicitSurfaceMesh(
        equation="x**2 + y**2 + z**2 - 4",
        space_metadata=space
    )
    
    vertices, triangles = mesh.generate_mesh()

    visualizer = MeshVisualizer(title="Hyperbolic Paraboloid")
    visualizer.visualize(
        space,
        vertices, 
        triangles, 
        equation_str="x**2 + y**2 + z**2 - 4",
        color='yellow'
    )
    visualizer.close()


def example_sin():
    grid = UniformGridMetadata(
        x_range=(-3, 3),
        y_range=(-3, 3),
        x_points=100,
        y_points=100
    )
    
    space = SpaceMetadata(grid_metadata=grid, z_min=-2, z_max=2, z_points=40)
    
    mesh = ExplicitSurfaceMesh(
        equation="sin(x * y)",
        space_metadata=space
    )
    
    vertices, triangles = mesh.generate_mesh()
    
    visualizer = MeshVisualizer(title="Hyperbolic Paraboloid")
    visualizer.visualize(
        space,
        vertices, 
        triangles, 
        equation_str="sin(x * y)",
        color='blue'
    )
    visualizer.close()


def example_inverse():
    grid = UniformGridMetadata(
        x_range=(-3, 3),
        y_range=(-3, 3),
        x_points=100,
        y_points=100
    )
    
    space = SpaceMetadata(grid_metadata=grid, z_min=-2, z_max=2, z_points=40)
    
    mesh = ExplicitSurfaceMesh(
        equation="1 / (x * y)",
        space_metadata=space
    )
    
    vertices, triangles = mesh.generate_mesh()
    
    visualizer = MeshVisualizer(title="Hyperbolic Paraboloid")
    visualizer.visualize(
        space,
        vertices, 
        triangles, 
        equation_str="1 / (x * y)",
        color='red'
    )
    visualizer.close()

def example_inverse_2():
    grid = UniformGridMetadata(
        x_range=(-3, 3),
        y_range=(-3, 3),
        x_points=100,
        y_points=100
    )
    
    space = SpaceMetadata(grid_metadata=grid, z_min=-4, z_max=4, z_points=100)
    
    mesh = ImplicitSurfaceMesh(
        equation="z - 1 / (x * y)",
        space_metadata=space
    )
    
    vertices, triangles = mesh.generate_mesh()
    
    visualizer = MeshVisualizer(title="Hyperbolic Paraboloid")
    visualizer.visualize(
        space,
        vertices, 
        triangles, 
        equation_str="z - 1 / (x * y)",
        color='gray'
    )
    visualizer.close()


if __name__ == "__main__":
    example_gyroid()
    example_sphere_with_custom_z_resolution()
    example_paraboloid()
    example_saddle()
    example_sphere_hemisphere()
    example_sin()
    example_inverse()
    example_inverse_2()
