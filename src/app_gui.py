"""
GUI application for 2D ray tracing with separate control window
"""

import tkinter as tk
from tkinter import ttk
import threading
import math
from config import ShapeType, Polarization
from scene_gui import GUIScene


class RayTracingController:
    """Control panel for ray tracing parameters"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ray Tracing Controls")
        self.root.geometry("350x600")

        # Scene parameters
        self.scene_params = {
            "shape_type": ShapeType.SQUARE,
            "position_x": 0.0,
            "position_y": 0.0,
            "rotation_deg": 30.0,
            "refractive_index_real": 1.31,
            "refractive_index_imag": 0.0,
            "num_rays": 10,
            "polarization": Polarization.PARALLEL,
            "max_recursion": 3,
            "plane_wave_offset": 0.0,
        }

        # Scene reference
        self.scene = None
        self.scene_thread = None
        self.running = False

        # Create controls
        self._create_controls()

        # Start scene automatically
        self._start_scene()

    def _create_controls(self):
        """Create all control widgets"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0

        # Title
        title = ttk.Label(
            main_frame, text="Scene Parameters", font=("Arial", 12, "bold")
        )
        title.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # Shape selection
        ttk.Label(main_frame, text="Shape:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.shape_var = tk.StringVar(value="SQUARE")
        shape_combo = ttk.Combobox(
            main_frame,
            textvariable=self.shape_var,
            values=["SQUARE"],
            state="readonly",
            width=15,
        )
        shape_combo.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        row += 1

        # Position X
        ttk.Label(main_frame, text="Position X:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.pos_x_var = tk.DoubleVar(value=0.0)
        pos_x_scale = ttk.Scale(
            main_frame,
            from_=-3,
            to=3,
            variable=self.pos_x_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self._update_label("pos_x", float(v)),
        )
        pos_x_scale.grid(row=row, column=1, pady=3)
        self.pos_x_label = ttk.Label(main_frame, text="0.00")
        self.pos_x_label.grid(row=row, column=2, pady=3)
        row += 1

        # Position Y
        ttk.Label(main_frame, text="Position Y:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.pos_y_var = tk.DoubleVar(value=0.0)
        pos_y_scale = ttk.Scale(
            main_frame,
            from_=-3,
            to=3,
            variable=self.pos_y_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self._update_label("pos_y", float(v)),
        )
        pos_y_scale.grid(row=row, column=1, pady=3)
        self.pos_y_label = ttk.Label(main_frame, text="0.00")
        self.pos_y_label.grid(row=row, column=2, pady=3)
        row += 1

        # Rotation
        ttk.Label(main_frame, text="Rotation (°):").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.rotation_var = tk.DoubleVar(value=30.0)
        rotation_scale = ttk.Scale(
            main_frame,
            from_=0,
            to=360,
            variable=self.rotation_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self._update_label("rotation", float(v)),
        )
        rotation_scale.grid(row=row, column=1, pady=3)
        self.rotation_label = ttk.Label(main_frame, text="30.0")
        self.rotation_label.grid(row=row, column=2, pady=3)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Refractive Index
        ttk.Label(main_frame, text="Refractive Index:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.n_var = tk.DoubleVar(value=1.31)
        n_scale = ttk.Scale(
            main_frame,
            from_=1.0,
            to=2.5,
            variable=self.n_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self._update_label("n", float(v)),
        )
        n_scale.grid(row=row, column=1, pady=3)
        self.n_label = ttk.Label(main_frame, text="1.31")
        self.n_label.grid(row=row, column=2, pady=3)
        row += 1

        # Number of rays
        ttk.Label(main_frame, text="Number of Rays:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.rays_var = tk.IntVar(value=10)
        rays_scale = ttk.Scale(
            main_frame,
            from_=1,
            to=50,
            variable=self.rays_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self._update_label("rays", int(float(v))),
        )
        rays_scale.grid(row=row, column=1, pady=3)
        self.rays_label = ttk.Label(main_frame, text="10")
        self.rays_label.grid(row=row, column=2, pady=3)
        row += 1

        # Plane wave offset
        ttk.Label(main_frame, text="Wave Y Offset:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.offset_var = tk.DoubleVar(value=0.0)
        offset_scale = ttk.Scale(
            main_frame,
            from_=-2,
            to=2,
            variable=self.offset_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self._update_label("offset", float(v)),
        )
        offset_scale.grid(row=row, column=1, pady=3)
        self.offset_label = ttk.Label(main_frame, text="0.00")
        self.offset_label.grid(row=row, column=2, pady=3)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Polarization
        ttk.Label(main_frame, text="Polarization:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        pol_frame = ttk.Frame(main_frame)
        pol_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        self.pol_var = tk.StringVar(value="PARALLEL")
        ttk.Radiobutton(
            pol_frame, text="Parallel", variable=self.pol_var, value="PARALLEL"
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            pol_frame,
            text="Perpendicular",
            variable=self.pol_var,
            value="PERPENDICULAR",
        ).pack(side=tk.LEFT, padx=(10, 0))
        row += 1

        # Max recursion
        ttk.Label(main_frame, text="Max Recursion:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.recursion_var = tk.IntVar(value=3)
        recursion_scale = ttk.Scale(
            main_frame,
            from_=0,
            to=10,
            variable=self.recursion_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self._update_label("recursion", int(float(v))),
        )
        recursion_scale.grid(row=row, column=1, pady=3)
        self.recursion_label = ttk.Label(main_frame, text="3")
        self.recursion_label.grid(row=row, column=2, pady=3)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        # Apply button
        self.apply_button = ttk.Button(
            main_frame, text="Apply Changes", command=self._apply_changes
        )
        self.apply_button.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1

        # Status label
        self.status_label = ttk.Label(
            main_frame, text="Scene running...", foreground="green"
        )
        self.status_label.grid(row=row, column=0, columnspan=3, pady=5)

    def _update_label(self, param, value):
        """Update label text when slider changes"""
        if param == "pos_x":
            self.pos_x_label.config(text=f"{value:.2f}")
        elif param == "pos_y":
            self.pos_y_label.config(text=f"{value:.2f}")
        elif param == "rotation":
            self.rotation_label.config(text=f"{value:.1f}")
        elif param == "n":
            self.n_label.config(text=f"{value:.2f}")
        elif param == "rays":
            self.rays_label.config(text=str(value))
        elif param == "offset":
            self.offset_label.config(text=f"{value:.2f}")
        elif param == "recursion":
            self.recursion_label.config(text=str(value))

    def _apply_changes(self):
        """Apply parameter changes to scene"""
        # Update parameters
        self.scene_params.update(
            {
                "shape_type": ShapeType[self.shape_var.get()],
                "position_x": self.pos_x_var.get(),
                "position_y": self.pos_y_var.get(),
                "rotation_deg": self.rotation_var.get(),
                "refractive_index_real": self.n_var.get(),
                "refractive_index_imag": 0.0,
                "num_rays": int(self.rays_var.get()),
                "polarization": Polarization[self.pol_var.get()],
                "max_recursion": int(self.recursion_var.get()),
                "plane_wave_offset": self.offset_var.get(),
            }
        )

        # Restart scene
        self.status_label.config(text="Applying changes...", foreground="orange")
        self._restart_scene()
        self.status_label.config(text="Scene running...", foreground="green")

    def _start_scene(self):
        """Start the pygame scene"""
        self.running = True
        self.scene_thread = threading.Thread(target=self._run_scene)
        self.scene_thread.daemon = True
        self.scene_thread.start()

    def _restart_scene(self):
        """Restart scene with new parameters"""
        # Signal current scene to stop
        self.running = False

        # Wait a moment for it to close
        if self.scene_thread:
            self.scene_thread.join(timeout=0.5)

        # Start new scene
        self._start_scene()

    def _run_scene(self):
        """Run the pygame scene in separate thread"""
        self.scene = GUIScene(self.scene_params)
        self.scene.run(lambda: self.running)

    def run(self):
        """Run the controller"""

        def on_close():
            self.running = False
            if self.scene_thread:
                self.scene_thread.join(timeout=1)
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_close)
        self.root.mainloop()


if __name__ == "__main__":
    app = RayTracingController()
    app.run()
