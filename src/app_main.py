"""
Main application that runs pygame in main thread with control panel
"""

import pygame
import tkinter as tk
from tkinter import ttk
import threading
import math
import time
import queue
from config import ShapeType, ShapeConfig, Polarization, VACUUM_REFRACTIVE_INDEX
from scene import Square
from ray import Ray, RayPath
from optics import (
    snells_law_refraction_angle,
    get_reflection_vector,
    get_refraction_vector,
    calculate_angle_of_incidence,
    fresnel_reflection_coefficient,
    fresnel_transmission_coefficient,
)
from colormap import (
    intensity_to_heat_color,
    calculate_intensity,
    draw_colorscale,
)
import numpy as np


class ControlPanel:
    """Tkinter control panel that runs in a separate thread"""

    def __init__(self, command_queue):
        self.command_queue = command_queue
        self.root = None

    def run(self):
        """Run the control panel"""
        self.root = tk.Tk()
        self.root.title("Ray Tracing Controls")
        self.root.geometry("350x550")

        self._create_controls()
        self.root.mainloop()

    def _create_controls(self):
        """Create control widgets"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0

        # Title
        title = ttk.Label(
            main_frame, text="Scene Parameters", font=("Arial", 12, "bold")
        )
        title.grid(row=row, column=0, columnspan=3, pady=10)
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
            command=lambda v: self._on_change(),
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
            command=lambda v: self._on_change(),
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
            command=lambda v: self._on_change(),
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
            command=lambda v: self._on_change(),
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
            command=lambda v: self._on_change(),
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
            command=lambda v: self._on_change(),
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
            pol_frame,
            text="Parallel",
            variable=self.pol_var,
            value="PARALLEL",
            command=self._on_change,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            pol_frame,
            text="Perpendicular",
            variable=self.pol_var,
            value="PERPENDICULAR",
            command=self._on_change,
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
            command=lambda v: self._on_change(),
        )
        recursion_scale.grid(row=row, column=1, pady=3)
        self.recursion_label = ttk.Label(main_frame, text="3")
        self.recursion_label.grid(row=row, column=2, pady=3)
        row += 1

        # Status
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5
        )
        row += 1
        self.status_label = ttk.Label(
            main_frame,
            text="Adjust parameters and they update live!",
            foreground="green",
        )
        self.status_label.grid(row=row, column=0, columnspan=3, pady=5)

    def _on_change(self, event=None):
        """Send parameter update to main thread"""
        # Update labels
        self.pos_x_label.config(text=f"{self.pos_x_var.get():.2f}")
        self.pos_y_label.config(text=f"{self.pos_y_var.get():.2f}")
        self.rotation_label.config(text=f"{self.rotation_var.get():.1f}")
        self.n_label.config(text=f"{self.n_var.get():.2f}")
        self.rays_label.config(text=str(int(self.rays_var.get())))
        self.offset_label.config(text=f"{self.offset_var.get():.2f}")
        self.recursion_label.config(text=str(int(self.recursion_var.get())))

        # Send update command
        params = {
            "position_x": self.pos_x_var.get(),
            "position_y": self.pos_y_var.get(),
            "rotation_deg": self.rotation_var.get(),
            "refractive_index": self.n_var.get(),
            "num_rays": int(self.rays_var.get()),
            "plane_wave_offset": self.offset_var.get(),
            "polarization": Polarization[self.pol_var.get()],
            "max_recursion": int(self.recursion_var.get()),
        }
        self.command_queue.put(("update", params))


class RayTracingApp:
    """Main application running pygame in main thread"""

    def __init__(self):
        # Initialize pygame
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2D Ray Tracing")
        self.font = pygame.font.Font(None, 16)

        # View parameters
        self.scale = 100
        self.offset = np.array([self.width / 2, self.height / 2])

        # Scene parameters
        self.params = {
            "position_x": 0.0,
            "position_y": 0.0,
            "rotation_deg": 30.0,
            "refractive_index": 1.31,
            "num_rays": 10,
            "plane_wave_offset": 0.0,
            "polarization": Polarization.PARALLEL,
            "max_recursion": 3,
        }

        # Scene objects
        self.shapes = []
        self.ray_paths = []
        self.plane_wave_x = -2.0
        self.bound_size = 10.0
        self.max_intensity = 0.5

        # Command queue for GUI updates
        self.command_queue = queue.Queue()

        # Start control panel in separate thread
        self.control_panel = ControlPanel(self.command_queue)
        self.control_thread = threading.Thread(target=self.control_panel.run)
        self.control_thread.daemon = True
        self.control_thread.start()

        # Initial scene setup
        self._update_scene()

    def _update_scene(self):
        """Update scene based on current parameters"""
        # Clear existing objects
        self.shapes = []
        self.ray_paths = []

        # Create shape
        shape = Square(
            center=(self.params["position_x"], self.params["position_y"]),
            size=1.0,
            rotation=math.radians(self.params["rotation_deg"]),
            refractive_index=complex(self.params["refractive_index"], 0.0),
        )
        self.shapes.append(shape)

        # Compute plane wave position
        min_x = float("inf")
        for shape in self.shapes:
            bounds = shape.get_bounds()
            min_x = min(min_x, bounds[0])
        self.plane_wave_x = min_x - 1.0  # 1 unit margin

        # Generate rays
        self._generate_rays()

    def _generate_rays(self):
        """Generate and trace rays"""
        num_rays = self.params["num_rays"]
        offset = self.params["plane_wave_offset"]

        if num_rays == 1:
            y_positions = [offset]
        else:
            y_positions = np.linspace(-1, 1, num_rays) + offset

        for y_pos in y_positions:
            ray = Ray(
                start=np.array([self.plane_wave_x, y_pos]),
                direction=np.array([1.0, 0.0]),
                electric_field=1.0,
                polarization=self.params["polarization"],
                refractive_index=VACUUM_REFRACTIVE_INDEX,
                recursion_level=0,
            )

            ray_path = RayPath(ray)
            self._trace_ray(ray, ray_path)
            self.ray_paths.append(ray_path)

    def _trace_ray(self, ray, ray_path):
        """Trace a ray through the scene"""
        from scene import Scene

        # Find intersection
        min_t = float("inf")
        min_intersection = None
        intersected_shape = None
        intersected_normal = None
        is_boundary = False

        # Check shape edges
        for shape in self.shapes:
            edges = shape.get_edges()
            normals = shape.get_edge_normals()
            for i, (edge_start, edge_end) in enumerate(edges):
                t, intersection = Scene._ray_edge_intersection(
                    None, ray.start, ray.direction, edge_start, edge_end
                )
                if t is not None and 0 < t < min_t:
                    min_t = t
                    min_intersection = intersection
                    intersected_shape = shape
                    intersected_normal = normals[i]

        # Check boundaries
        boundary_edges = self._get_boundary_edges()
        for edge_start, edge_end in boundary_edges:
            t, intersection = Scene._ray_edge_intersection(
                None, ray.start, ray.direction, edge_start, edge_end
            )
            if t is not None and 0 < t < min_t:
                min_t = t
                min_intersection = intersection
                is_boundary = True

        # Set endpoint and handle interaction
        if min_intersection is not None:
            ray.end = min_intersection

            if (
                not is_boundary
                and ray.recursion_level < self.params["max_recursion"]
                and intersected_shape
            ):
                self._handle_intersection(
                    ray,
                    ray_path,
                    intersected_shape,
                    intersected_normal,
                    min_intersection,
                )

    def _handle_intersection(self, ray, ray_path, shape, normal, intersection_point):
        """Handle ray-shape intersection"""
        # Determine indices
        is_entering = ray.refractive_index == VACUUM_REFRACTIVE_INDEX

        if is_entering:
            n1 = VACUUM_REFRACTIVE_INDEX
            n2 = shape.refractive_index
        else:
            n1 = shape.refractive_index
            n2 = VACUUM_REFRACTIVE_INDEX

        # Calculate angles
        theta_i = calculate_angle_of_incidence(ray.direction, normal)
        theta_t = snells_law_refraction_angle(theta_i, n1, n2)

        # Fresnel coefficients
        r_coeff = fresnel_reflection_coefficient(
            n1, n2, theta_i, theta_t, ray.polarization
        )

        # Reflected ray
        reflected_dir = get_reflection_vector(ray.direction, normal)
        reflected_ray = Ray(
            start=intersection_point,
            direction=reflected_dir,
            electric_field=ray.electric_field * abs(r_coeff),
            polarization=ray.polarization,
            refractive_index=n1,
            recursion_level=ray.recursion_level + 1,
        )
        self._trace_ray(reflected_ray, ray_path)
        ray_path.add_segment(reflected_ray)

        # Refracted ray
        if theta_t is not None:
            t_coeff = fresnel_transmission_coefficient(
                n1, n2, theta_i, theta_t, ray.polarization
            )
            refracted_dir = get_refraction_vector(
                ray.direction, normal, theta_i, theta_t
            )
            refracted_ray = Ray(
                start=intersection_point,
                direction=refracted_dir,
                electric_field=ray.electric_field * abs(t_coeff),
                polarization=ray.polarization,
                refractive_index=n2,
                recursion_level=ray.recursion_level + 1,
            )
            self._trace_ray(refracted_ray, ray_path)
            ray_path.add_segment(refracted_ray)

    def _get_boundary_edges(self):
        """Get scene boundary edges"""
        b = self.bound_size
        corners = [
            np.array([-b, -b]),
            np.array([b, -b]),
            np.array([b, b]),
            np.array([-b, b]),
        ]
        edges = []
        for i in range(4):
            edges.append((corners[i], corners[(i + 1) % 4]))
        return edges

    def world_to_screen(self, point):
        """Convert world to screen coordinates"""
        screen_point = point * np.array([1, -1]) * self.scale + self.offset
        return tuple(screen_point.astype(int))

    def render(self):
        """Render the scene"""
        self.screen.fill((0, 0, 0))

        # Draw axes
        pygame.draw.line(
            self.screen,
            (50, 50, 50),
            self.world_to_screen(np.array([-10, 0])),
            self.world_to_screen(np.array([10, 0])),
            1,
        )
        pygame.draw.line(
            self.screen,
            (50, 50, 50),
            self.world_to_screen(np.array([0, -10])),
            self.world_to_screen(np.array([0, 10])),
            1,
        )

        # Draw boundary
        for edge_start, edge_end in self._get_boundary_edges():
            pygame.draw.line(
                self.screen,
                (40, 40, 40),
                self.world_to_screen(edge_start),
                self.world_to_screen(edge_end),
                1,
            )

        # Draw plane wave
        pygame.draw.line(
            self.screen,
            (100, 100, 255),
            self.world_to_screen(
                np.array([self.plane_wave_x, -1 + self.params["plane_wave_offset"]])
            ),
            self.world_to_screen(
                np.array([self.plane_wave_x, 1 + self.params["plane_wave_offset"]])
            ),
            2,
        )

        # Draw shapes
        for shape in self.shapes:
            vertices = shape.transformed_vertices
            screen_points = [self.world_to_screen(v) for v in vertices]
            pygame.draw.polygon(self.screen, (255, 255, 255), screen_points, 2)

            # Draw center
            center_screen = self.world_to_screen(shape.center)
            pygame.draw.circle(self.screen, (255, 100, 100), center_screen, 3)

        # Draw rays
        for ray_path in self.ray_paths:
            for ray in ray_path.segments:
                if ray.end is not None:
                    intensity = calculate_intensity(ray.electric_field)
                    color = intensity_to_heat_color(intensity, self.max_intensity)
                    thickness = max(
                        1, min(3, int(1 + 2 * intensity / self.max_intensity))
                    )
                    pygame.draw.line(
                        self.screen,
                        color,
                        self.world_to_screen(ray.start),
                        self.world_to_screen(ray.end),
                        thickness,
                    )

        # Draw colorscale
        draw_colorscale(self.screen, self.font, 20, 40, 20, 100, self.max_intensity)

        pygame.display.flip()

    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Check for GUI updates
            try:
                while not self.command_queue.empty():
                    cmd, params = self.command_queue.get_nowait()
                    if cmd == "update":
                        self.params.update(params)
                        self._update_scene()
            except queue.Empty:
                pass

            self.render()
            clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    app = RayTracingApp()
    app.run()
