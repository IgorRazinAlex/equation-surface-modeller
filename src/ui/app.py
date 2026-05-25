import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLineEdit, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QMessageBox, QLabel, QSizePolicy, QFileDialog,
)
from PyQt5.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor

from src.view.plot import SpaceMetadata, UniformGridMetadata, LineMetadata
from src.view.mesh import create_mesh, export_to_obj, export_to_stl
from src.view.plot_view import MeshVisualizer


class SurfaceVisualizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Surface Visualizer")
        self.setGeometry(100, 100, 1400, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, 1)

        viz_widget = self._create_visualization_widget()
        main_layout.addWidget(viz_widget, 3)

        self.current_vertices      = None
        self.current_triangles     = None
        self.current_equation      = ""


    def _create_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(370)
        layout = QVBoxLayout(panel)
 
        layout.addWidget(self._create_equation_group())
        layout.addWidget(self._create_algorithm_group())
        layout.addWidget(self._create_grid_group())
        layout.addWidget(self._create_display_group())
        layout.addWidget(self._create_export_group())
        layout.addWidget(self._create_buttons())
        layout.addStretch()

        return panel

    def _create_equation_group(self) -> QGroupBox:
        group = QGroupBox("Equation")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Type:"))
        self.equation_type = QComboBox()
        self.equation_type.addItems([
            "Explicit (z = f(x,y))",
            "Implicit (F(x,y,z) = 0)",
            "Parametric (x = x(t), y = y(t), z = z(t))",
        ])
        self.equation_type.currentIndexChanged.connect(
            self._on_equation_type_changed
        )
        row.addWidget(self.equation_type)
        layout.addLayout(row)

        layout.addWidget(QLabel("Equation:"))
        self.equation_input = QLineEdit()
        self.equation_input.setPlaceholderText(
            "e.g.  sin(x)*cos(y)  or  x**2 + y**2 - z - 1"
        )
        layout.addWidget(self.equation_input)

        group.setLayout(layout)
        return group

    def _create_algorithm_group(self) -> QGroupBox:
        self.algo_group = QGroupBox("Meshing Algorithm (Implicit only)")
        layout = QFormLayout()

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Marching Cubes",
            "Dual Contouring",
        ])
        self.algorithm_combo.setToolTip(
            "Dual Contouring корректно обрабатывает разрывные функции\n"
            "(например, z - 1/x = 0), отбрасывая артефакты вблизи разрывов."
        )
        layout.addRow("Algorithm:", self.algorithm_combo)

        self.disc_threshold = QDoubleSpinBox()
        self.disc_threshold.setRange(0.1, 1e6)
        self.disc_threshold.setValue(10.0)
        self.disc_threshold.setDecimals(1)
        self.disc_threshold.setToolTip(
            "Ячейки, где |F(x,y,z)| превышает этот порог,\n"
            "считаются артефактами разрыва и пропускаются.\n"
            "Уменьшите, если артефакты остаются;\n"
            "увеличьте, если поверхность имеет «дыры»."
        )
        layout.addRow("Discontinuity threshold:", self.disc_threshold)

        self.algo_group.setLayout(layout)
        return self.algo_group

    def _create_grid_group(self) -> QGroupBox:
        group = QGroupBox("Grid Parameters")
        layout = QFormLayout()

        self.x_min = self._make_dspin(-5)
        self.x_max = self._make_dspin(5)
        layout.addRow("X range:", self._range_widget(self.x_min, self.x_max))

        self.y_min = self._make_dspin(-5)
        self.y_max = self._make_dspin(5)
        layout.addRow("Y range:", self._range_widget(self.y_min, self.y_max))

        self.z_min = self._make_dspin(-5)
        self.z_max = self._make_dspin(5)
        self.z_range_row_label = QLabel("Z range:")
        layout.addRow(self.z_range_row_label,
                      self._range_widget(self.z_min, self.z_max))

        self.x_points = QSpinBox()
        self.x_points.setRange(10, 300)
        self.x_points.setValue(50)
        layout.addRow("X points:", self.x_points)

        self.y_points = QSpinBox()
        self.y_points.setRange(10, 300)
        self.y_points.setValue(50)
        layout.addRow("Y points:", self.y_points)

        self.z_points = QSpinBox()
        self.z_points.setRange(10, 300)
        self.z_points.setValue(50)
        self.z_points_label = QLabel("Z points:")
        layout.addRow(self.z_points_label, self.z_points)

        group.setLayout(layout)
        return group

    def _create_display_group(self) -> QGroupBox:
        group = QGroupBox("Display Options")
        layout = QFormLayout()

        self.mesh_color = QComboBox()
        self.mesh_color.addItems([
            "orange", "lightblue", "lightgreen",
            "white", "red", "yellow", "cyan", "magenta",
        ])
        layout.addRow("Color:", self.mesh_color)

        self.mesh_opacity = QDoubleSpinBox()
        self.mesh_opacity.setRange(0.0, 1.0)
        self.mesh_opacity.setSingleStep(0.1)
        self.mesh_opacity.setValue(1.0)
        layout.addRow("Opacity:", self.mesh_opacity)

        self.show_edges    = QCheckBox()
        self.smooth_shading = QCheckBox()
        self.smooth_shading.setChecked(True)
        layout.addRow("Show edges:",     self.show_edges)
        layout.addRow("Smooth shading:", self.smooth_shading)

        group.setLayout(layout)
        return group

    def _create_export_group(self) -> QGroupBox:
        """Create export settings group."""
        group = QGroupBox("Export Settings")
        layout = QFormLayout()

        self.export_format = QComboBox()
        self.export_format.addItems([
            "OBJ (Wavefront)",
            "STL (Binary)",
            "STL (ASCII)",
        ])
        self.export_format.setToolTip(
            "OBJ: текстовый формат с поддержкой метаданных\n"
            "STL Binary: компактный бинарный формат\n"
            "STL ASCII: читаемый текстовый формат"
        )
        layout.addRow("Format:", self.export_format)

        group.setLayout(layout)
        return group

    def _create_buttons(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_generate = QPushButton("Generate Surface")
        btn_generate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        btn_generate.clicked.connect(self.generate_surface)
        layout.addWidget(btn_generate)

        btn_update = QPushButton("Update Display")
        btn_update.clicked.connect(self.visualize_current_surface)
        layout.addWidget(btn_update)

        btn_clear = QPushButton("Clear Scene")
        btn_clear.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
        )
        btn_clear.clicked.connect(self.clear_surface)
        layout.addWidget(btn_clear)

        btn_export = QPushButton("Export Mesh")
        btn_export.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        btn_export.clicked.connect(self.export_mesh)
        layout.addWidget(btn_export)

        return widget

    def _create_visualization_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(widget)
        self.plotter.set_background("black")
        self.plotter.add_axes()
        layout.addWidget(self.plotter.interactor)

        return widget
 
    @staticmethod
    def _make_dspin(value: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(-1000, 1000)
        sb.setValue(value)
        return sb

    @staticmethod
    def _range_widget(lo: QDoubleSpinBox, hi: QDoubleSpinBox) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lo)
        lay.addWidget(QLabel("→"))
        lay.addWidget(hi)
        return w

    def _on_equation_type_changed(self, index: int):
        is_implicit = (index == 1)
        is_parametric = (index == 2)
        
        # Show/hide Z range and points for implicit surfaces only
        self.z_min.setVisible(is_implicit)
        self.z_max.setVisible(is_implicit)
        self.z_range_row_label.setVisible(is_implicit)
        self.z_points.setVisible(is_implicit)
        self.z_points_label.setVisible(is_implicit)
        
        # Algorithm group only for implicit surfaces
        self.algo_group.setVisible(is_implicit)
        
        # Update placeholder text based on equation type
        if is_parametric:
            self.equation_input.setPlaceholderText(
                "e.g.  sin(t), cos(t), t  or  x = t, y = t^2, z = sin(t)"
            )
        elif is_implicit:
            self.equation_input.setPlaceholderText(
                "e.g.  x**2 + y**2 - z - 1"
            )
        else:  # explicit
            self.equation_input.setPlaceholderText(
                "e.g.  sin(x)*cos(y)"
            )

    def _get_equation_type_str(self) -> str:
        index = self.equation_type.currentIndex()
        if index == 1:
            return "implicit"
        elif index == 2:
            return "parametric"
        else:
            return "explicit"

    def _get_algorithm_str(self) -> str:
        text = self.algorithm_combo.currentText()
        if "Dual" in text:
            return "dual_contouring"
        return "marching_cubes"

    def _get_parameters_metadata(self):
        """Return either SpaceMetadata or LineMetadata based on equation type."""
        eq_type = self._get_equation_type_str()
        
        if eq_type == "parametric":
            # For parametric curves, use x_min/x_max as t_min/t_max
            # Use y_min/y_max as radius? No, we'll use defaults for now
            # Actually we need to add UI controls for radius and segments
            # For now, use defaults from LineMetadata
            return LineMetadata(
                t_min=self.x_min.value(),
                t_max=self.x_max.value(),
                t_points=self.x_points.value(),  # Use x_points as t_points
                radius=0.1,  # Default
                segments=8,  # Default
            )
        else:
            # For explicit/implicit surfaces
            x_pts = self.x_points.value()
            y_pts = self.y_points.value()
            z_pts = self.z_points.value()

            grid = UniformGridMetadata(
                x_range=(self.x_min.value(), self.x_max.value()),
                y_range=(self.y_min.value(), self.y_max.value()),
                x_points=x_pts,
                y_points=y_pts,
            )
            return SpaceMetadata(
                grid_metadata=grid,
                x_min=self.x_min.value(), x_max=self.x_max.value(),
                y_min=self.y_min.value(), y_max=self.y_max.value(),
                z_min=self.z_min.value(), z_max=self.z_max.value(),
                z_points=z_pts,
            )

    def _get_clipping_metadata(self) -> SpaceMetadata:
        """Return SpaceMetadata for clipping based on UI bounds."""
        # For clipping we always need SpaceMetadata with x,y,z ranges
        # Use default grid with 2x2 points (not important for clipping)
        grid = UniformGridMetadata(
            x_range=(self.x_min.value(), self.x_max.value()),
            y_range=(self.y_min.value(), self.y_max.value()),
            x_points=2,
            y_points=2,
        )
        return SpaceMetadata(
            grid_metadata=grid,
            x_min=self.x_min.value(), x_max=self.x_max.value(),
            y_min=self.y_min.value(), y_max=self.y_max.value(),
            z_min=self.z_min.value(), z_max=self.z_max.value(),
            z_points=2,  # Not used for clipping
        )

    def generate_surface(self):
        equation = self.equation_input.text().strip()
        if not equation:
            QMessageBox.warning(self, "Warning", "Please enter an equation.")
            return

        try:
            eq_type   = self._get_equation_type_str()
            algorithm = self._get_algorithm_str()
            parameters_md  = self._get_parameters_metadata()

            mesh_obj = create_mesh(
                equation=equation,
                equation_type=eq_type,
                parameters_metadata=parameters_md,
                algorithm=algorithm,
                discontinuity_threshold=self.disc_threshold.value(),
            )
            vertices, triangles = mesh_obj.generate_mesh()

            self.current_vertices       = vertices
            self.current_triangles      = triangles
            self.current_equation       = equation
 
            self.visualize_current_surface()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to generate surface:\n{e}"
            )

    def visualize_current_surface(self):
        if self.current_vertices is None or self.current_triangles is None:
            return

        try:
            self.plotter.clear()

            viz = MeshVisualizer()
            mesh = viz.prepare_mesh_data(
                self.current_vertices, self.current_triangles
            )
            # Get clipping metadata from current UI values
            clipping_md = self._get_clipping_metadata()
            mesh = viz.clip(mesh, clipping_md)

            self.plotter.add_mesh(
                mesh,
                color=self.mesh_color.currentText(),
                opacity=self.mesh_opacity.value(),
                show_edges=self.show_edges.isChecked(),
                smooth_shading=self.smooth_shading.isChecked(),
                specular=0.5,
            )
            self.plotter.add_bounding_box()
            self.plotter.add_axes()
            self.plotter.add_text(
                f"Equation: {self.current_equation}",
                position="upper_left",
                font_size=10,
            )
            self.plotter.render()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to visualize surface:\n{e}"
            )

    def clear_surface(self):
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.render()

        self.current_vertices       = None
        self.current_triangles      = None
        self.current_equation       = ""

    def export_mesh(self):
        if self.current_vertices is None or self.current_triangles is None:
            QMessageBox.warning(
                self, "Warning", 
                "No mesh to export. Please generate a surface first."
            )
            return

        format_text = self.export_format.currentText()

        if "OBJ" in format_text:
            file_filter = "OBJ Files (*.obj);;All Files (*)"
            default_ext = ".obj"
        else:  # STL
            file_filter = "STL Files (*.stl);;All Files (*)"
            default_ext = ".stl"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Mesh to {format_text}",
            "",
            file_filter
        )
        
        if not filename:
            return

        if not filename.lower().endswith(default_ext):
            filename += default_ext

        try:
            metadata = None
            if "OBJ" in format_text:
                metadata = {
                    "Equation": self.current_equation,
                    "Equation Type": self._get_equation_type_str(),
                    "Algorithm": self._get_algorithm_str(),
                    "X Range": f"{self.x_min.value()} to {self.x_max.value()}",
                    "Y Range": f"{self.y_min.value()} to {self.y_max.value()}",
                    "Z Range": f"{self.z_min.value()} to {self.z_max.value()}",
                    "X Points": self.x_points.value(),
                    "Y Points": self.y_points.value(),
                    "Z Points": self.z_points.value(),
                    "Vertices": len(self.current_vertices),
                    "Faces": len(self.current_triangles),
                }

            if "OBJ" in format_text:
                export_to_obj(
                    self.current_vertices,
                    self.current_triangles,
                    filename,
                    metadata
                )
                format_name = "OBJ"
   
            elif "STL (Binary)" in format_text:
                export_to_stl(
                    self.current_vertices,
                    self.current_triangles,
                    filename,
                    ascii_format=False
                )
                format_name = "STL (Binary)"

            elif "STL (ASCII)" in format_text:
                export_to_stl(
                    self.current_vertices,
                    self.current_triangles,
                    filename,
                    ascii_format=True
                )
                format_name = "STL (ASCII)"
            else:
                raise ValueError(f"Unknown export format: {format_text}")

            QMessageBox.information(
                self, "Export Successful",
                f"Mesh exported successfully to:\n{filename}\n\n"
                f"Format: {format_name}\n"
                f"Vertices: {len(self.current_vertices)}\n"
                f"Faces: {len(self.current_triangles)}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export mesh:\n{e}"
            )


def main():
    app = QApplication(sys.argv)
    window = SurfaceVisualizerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
