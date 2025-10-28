import tkinter as tk
from tkinter import ttk
import pygame
import threading
import math
import numpy as np
from typing import Optional
import os
import sys

# Set up pygame to use embedded window
os.environ["SDL_WINDOWID"] = str(0)
os.environ["SDL_VIDEODRIVER"] = "windib"

from config import ShapeType, ShapeConfig, Polarization, VACUUM_REFRACTIVE_INDEX
from scene import Scene


class RayTracingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("2D Ray Tracing Simulator")
        self.root.geometry("1200x700")

        # Scene parameters (will be updated by GUI)
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
            "plane_wave_offset": 0.0,  # Y-offset for plane wave
        }

        # Create the GUI layout
        self._create_layout()

        # Initialize scene in render panel
        self.scene = None
        self.scene_thread = None
        self.running = False

        # Start the scene after GUI is created
        self.root.after(100, self._init_pygame)

    def _create_layout(self):
        """Create the two-panel layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Pygame render (embedded)
        render_frame = ttk.LabelFrame(
            main_frame, text="Ray Tracing Visualization", padding=5
        )
        render_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Pygame embed frame
        self.pygame_frame = tk.Frame(render_frame, width=800, height=600, bg="black")
        self.pygame_frame.pack()

        # Right panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Scene Controls", padding=10)
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Add controls to the control panel
        self._create_controls(control_frame)

    def _create_controls(self, parent):
        """Create all control widgets"""
        row = 0

        # Shape selection
        ttk.Label(parent, text="Shape:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.shape_var = tk.StringVar(value="SQUARE")
        self.shape_combo = ttk.Combobox(
            parent,
            textvariable=self.shape_var,
            values=["SQUARE"],
            state="readonly",
            width=15,
        )
        self.shape_combo.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=2)
        self.shape_combo.bind("<<ComboboxSelected>>", self._on_update)
        row += 1

        # Position X slider
        ttk.Label(parent, text="Position X:").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.pos_x_var = tk.DoubleVar(value=self.scene_params["position_x"])
        self.pos_x_slider = ttk.Scale(
            parent,
            from_=-3,
            to=3,
            variable=self.pos_x_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda _: self._on_update(),
        )
        self.pos_x_slider.grid(row=row, column=1, pady=2)
        self.pos_x_label = ttk.Label(parent, text=f"{self.pos_x_var.get():.2f}")
        self.pos_x_label.grid(row=row, column=2, pady=2)
        row += 1

        # Position Y slider
        ttk.Label(parent, text="Position Y:").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.pos_y_var = tk.DoubleVar(value=self.scene_params["position_y"])
        self.pos_y_slider = ttk.Scale(
            parent,
            from_=-3,
            to=3,
            variable=self.pos_y_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda _: self._on_update(),
        )
        self.pos_y_slider.grid(row=row, column=1, pady=2)
        self.pos_y_label = ttk.Label(parent, text=f"{self.pos_y_var.get():.2f}")
        self.pos_y_label.grid(row=row, column=2, pady=2)
        row += 1

        # Rotation (in degrees)
        ttk.Label(parent, text="Rotation (deg):").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.rotation_var = tk.DoubleVar(value=self.scene_params["rotation_deg"])
        self.rotation_slider = ttk.Scale(
            parent,
            from_=0,
            to=360,
            variable=self.rotation_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda _: self._on_update(),
        )
        self.rotation_slider.grid(row=row, column=1, pady=2)
        self.rotation_label = ttk.Label(parent, text=f"{self.rotation_var.get():.1f}°")
        self.rotation_label.grid(row=row, column=2, pady=2)
        row += 1

        # Separator
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        # Refractive index (real part)
        ttk.Label(parent, text="Refractive Index:").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.n_real_var = tk.DoubleVar(value=self.scene_params["refractive_index_real"])
        self.n_real_slider = ttk.Scale(
            parent,
            from_=1.0,
            to=2.5,
            variable=self.n_real_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda _: self._on_update(),
        )
        self.n_real_slider.grid(row=row, column=1, pady=2)
        self.n_real_label = ttk.Label(parent, text=f"{self.n_real_var.get():.3f}")
        self.n_real_label.grid(row=row, column=2, pady=2)
        row += 1

        # Number of rays
        ttk.Label(parent, text="Number of Rays:").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.num_rays_var = tk.IntVar(value=self.scene_params["num_rays"])
        self.num_rays_slider = ttk.Scale(
            parent,
            from_=1,
            to=50,
            variable=self.num_rays_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda _: self._on_update(),
        )
        self.num_rays_slider.grid(row=row, column=1, pady=2)
        self.num_rays_label = ttk.Label(parent, text=f"{int(self.num_rays_var.get())}")
        self.num_rays_label.grid(row=row, column=2, pady=2)
        row += 1

        # Plane wave Y offset
        ttk.Label(parent, text="Plane Wave Offset:").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.wave_offset_var = tk.DoubleVar(
            value=self.scene_params["plane_wave_offset"]
        )
        self.wave_offset_slider = ttk.Scale(
            parent,
            from_=-2,
            to=2,
            variable=self.wave_offset_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda _: self._on_update(),
        )
        self.wave_offset_slider.grid(row=row, column=1, pady=2)
        self.wave_offset_label = ttk.Label(
            parent, text=f"{self.wave_offset_var.get():.2f}"
        )
        self.wave_offset_label.grid(row=row, column=2, pady=2)
        row += 1

        # Separator
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        # Polarization
        ttk.Label(parent, text="Polarization:").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.polarization_var = tk.StringVar(value="PARALLEL")
        pol_frame = ttk.Frame(parent)
        pol_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=2)
        ttk.Radiobutton(
            pol_frame,
            text="Parallel",
            variable=self.polarization_var,
            value="PARALLEL",
            command=self._on_update,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            pol_frame,
            text="Perpendicular",
            variable=self.polarization_var,
            value="PERPENDICULAR",
            command=self._on_update,
        ).pack(side=tk.LEFT, padx=(10, 0))
        row += 1

        # Max recursion depth
        ttk.Label(parent, text="Max Recursion:").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.recursion_var = tk.IntVar(value=self.scene_params["max_recursion"])
        self.recursion_slider = ttk.Scale(
            parent,
            from_=0,
            to=10,
            variable=self.recursion_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda _: self._on_update(),
        )
        self.recursion_slider.grid(row=row, column=1, pady=2)
        self.recursion_label = ttk.Label(
            parent, text=f"{int(self.recursion_var.get())}"
        )
        self.recursion_label.grid(row=row, column=2, pady=2)
        row += 1

        # Update button
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        self.update_button = ttk.Button(
            parent, text="Apply Changes", command=self._apply_changes
        )
        self.update_button.grid(row=row, column=0, columnspan=3, pady=10)

    def _on_update(self, event=None):
        """Update labels when sliders change"""
        self.pos_x_label.config(text=f"{self.pos_x_var.get():.2f}")
        self.pos_y_label.config(text=f"{self.pos_y_var.get():.2f}")
        self.rotation_label.config(text=f"{self.rotation_var.get():.1f}°")
        self.n_real_label.config(text=f"{self.n_real_var.get():.3f}")
        self.num_rays_label.config(text=f"{int(self.num_rays_var.get())}")
        self.wave_offset_label.config(text=f"{self.wave_offset_var.get():.2f}")
        self.recursion_label.config(text=f"{int(self.recursion_var.get())}")

    def _apply_changes(self):
        """Apply GUI changes to the scene"""
        # Update scene parameters from GUI
        self.scene_params.update(
            {
                "shape_type": ShapeType[self.shape_var.get()],
                "position_x": self.pos_x_var.get(),
                "position_y": self.pos_y_var.get(),
                "rotation_deg": self.rotation_var.get(),
                "refractive_index_real": self.n_real_var.get(),
                "num_rays": int(self.num_rays_var.get()),
                "polarization": Polarization[self.polarization_var.get()],
                "max_recursion": int(self.recursion_var.get()),
                "plane_wave_offset": self.wave_offset_var.get(),
            }
        )

        # Restart scene with new parameters
        if self.scene:
            self._restart_scene()

    def _init_pygame(self):
        """Initialize pygame in the embed frame"""
        # Get the window handle
        window_id = self.pygame_frame.winfo_id()

        # Set SDL to use the tkinter frame
        if sys.platform == "win32":
            os.environ["SDL_WINDOWID"] = str(window_id)
            os.environ["SDL_VIDEODRIVER"] = "windib"
        else:
            os.environ["SDL_WINDOWID"] = str(window_id)

        # Start the scene
        self._restart_scene()

    def _restart_scene(self):
        """Restart the scene with current parameters"""
        # Stop existing scene
        if self.running:
            self.running = False
            if self.scene_thread:
                self.scene_thread.join(timeout=1)

        # Create and start new scene in thread
        self.running = True
        self.scene_thread = threading.Thread(target=self._run_scene)
        self.scene_thread.daemon = True
        self.scene_thread.start()

    def _run_scene(self):
        """Run the pygame scene (in separate thread)"""
        # Import here to ensure pygame init happens in thread
        from scene_gui import GUIScene

        self.scene = GUIScene(self.scene_params, self.pygame_frame)
        self.scene.run(lambda: self.running)

    def run(self):
        """Start the GUI application"""

        def on_close():
            self.running = False
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_close)
        self.root.mainloop()


if __name__ == "__main__":
    app = RayTracingGUI()
    app.run()
