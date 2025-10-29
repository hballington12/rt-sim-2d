"""
Integrated GUI using pygame-gui for seamless control panel
"""

import pygame
import pygame_gui
import numpy as np
import math
from typing import Dict, Any
from config import ShapeType, Polarization, VACUUM_REFRACTIVE_INDEX
from scene import Square, Triangle, Hexagon, Octagon, Circle
from ray import Ray, RayPath
from optics import (
    snells_law_refraction_angle,
    get_reflection_vector,
    get_refraction_vector,
    calculate_angle_of_incidence,
    fresnel_reflection_coefficient,
    fresnel_transmission_coefficient,
    calculate_absorption_factor,
)
from colormap import (
    intensity_to_heat_color,
    calculate_intensity,
    draw_colorscale,
)


class IntegratedRayTracingApp:
    """Ray tracing app with integrated pygame-gui controls"""

    def __init__(self):
        pygame.init()

        # Window setup - wider to accommodate control panel
        self.width = 1200
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2D Ray Tracing with Controls")

        # Font
        self.font = pygame.font.Font(None, 16)

        # Rendering area (left side)
        self.render_width = 800
        self.render_surface = pygame.Surface((self.render_width, self.height))

        # View parameters
        self.scale = 100
        self.offset = np.array([self.render_width / 2, self.height / 2])

        # GUI Manager for controls (right side)
        self.ui_manager = pygame_gui.UIManager((self.width, self.height))

        # Scene parameters
        self.scene_params = {
            "position_x": 0.0,
            "position_y": 0.3,
            "rotation_deg": 0.0,
            "scale": 1.0,
            "refractive_index_real": 1.31,
            "refractive_index_imag": 0.0,
            "wavelength_nm": 532.0,  # Wavelength in nanometers
            "num_rays": 1,
            "plane_wave_offset": 0.0,
            "polarization": Polarization.PARALLEL,
            "max_recursion": 2,
            "shape_type": ShapeType.HEXAGON,
        }

        # Scene objects
        self.shapes = []
        self.ray_paths = []
        self.plane_wave_x = -2.0
        self.bound_size = 10.0
        self.max_intensity = 0.5

        # Scattering analysis data
        self.scattering_angles = np.linspace(
            0, 180, 181
        )  # 0 to 180 degrees in 1° steps
        self.scattering_intensity = np.zeros(181)  # Intensity at each angle bin
        self.initial_direction = np.array([1.0, 0.0])  # Initial ray direction (+x)

        # Create GUI controls
        self._create_gui_controls()

        # Initial scene update
        self._update_scene()

        # Clock for FPS
        self.clock = pygame.time.Clock()

    def _create_gui_controls(self):
        """Create all GUI control elements using pygame-gui"""
        panel_x = self.render_width + 20
        y_pos = 10
        label_width = 150
        control_width = 200
        height = 30
        spacing = 30  # Reduced from 40 to fit more controls

        # Title
        title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, 350, 30),
            text="Scene Controls",
            manager=self.ui_manager,
        )
        y_pos += 35  # Reduced from 50

        # Position X Slider
        self.pos_x_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Position X: {self.scene_params['position_x']:.2f}",
            manager=self.ui_manager,
        )
        self.pos_x_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["position_x"],
            value_range=(-3.0, 3.0),
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Position Y Slider
        self.pos_y_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Position Y: {self.scene_params['position_y']:.2f}",
            manager=self.ui_manager,
        )
        self.pos_y_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["position_y"],
            value_range=(-3.0, 3.0),
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Rotation Slider
        self.rotation_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Rotation: {self.scene_params['rotation_deg']:.1f}°",
            manager=self.ui_manager,
        )
        self.rotation_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["rotation_deg"],
            value_range=(0.0, 360.0),
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Scale Slider
        self.scale_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Scale: {self.scene_params['scale']:.2f}",
            manager=self.ui_manager,
        )
        self.scale_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["scale"],
            value_range=(0.1, 5.0),
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Refractive Index Real Part Slider
        self.n_real_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"n (real): {self.scene_params['refractive_index_real']:.3f}",
            manager=self.ui_manager,
        )
        self.n_real_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["refractive_index_real"],
            value_range=(1.0, 2.5),
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Refractive Index Imaginary Part Slider
        self.n_imag_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"n (imag): {self.scene_params['refractive_index_imag']:.3f}",
            manager=self.ui_manager,
        )
        self.n_imag_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["refractive_index_imag"],
            value_range=(0.0, 0.1),
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Wavelength Slider
        self.wavelength_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Wavelength: {self.scene_params['wavelength_nm']:.0f} nm",
            manager=self.ui_manager,
        )
        self.wavelength_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["wavelength_nm"],
            value_range=(380.0, 780.0),  # Visible light range
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Number of Rays Slider
        self.rays_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Number of Rays: {self.scene_params['num_rays']}",
            manager=self.ui_manager,
        )
        self.rays_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=float(self.scene_params["num_rays"]),
            value_range=(1.0, 50.0),
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Plane Wave Offset Slider
        self.offset_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Wave Y Offset: {self.scene_params['plane_wave_offset']:.2f}",
            manager=self.ui_manager,
        )
        self.offset_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["plane_wave_offset"],
            value_range=(-2.0, 2.0),
            manager=self.ui_manager,
        )
        y_pos += spacing

        # Max Recursion Slider
        self.recursion_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Max Recursion: {self.scene_params['max_recursion']}",
            manager=self.ui_manager,
        )
        self.recursion_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=float(self.scene_params["max_recursion"]),
            value_range=(0.0, 10.0),
            manager=self.ui_manager,
        )
        y_pos += spacing + 10  # Reduced extra spacing

        # Shape Dropdown
        shape_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text="Shape:",
            manager=self.ui_manager,
        )
        self.shape_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["Square", "Triangle", "Hexagon", "Octagon", "Circle"],
            starting_option="Square",
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            manager=self.ui_manager,
        )
        y_pos += spacing + 10  # Reduced extra spacing

        # Polarization Selection
        pol_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text="Polarization:",
            manager=self.ui_manager,
        )
        y_pos += 30  # Reduced from 35

        self.pol_parallel_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y_pos, 170, height),
            text="Parallel",
            manager=self.ui_manager,
        )

        self.pol_perp_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x + 180, y_pos, 170, height),
            text="Perpendicular",
            manager=self.ui_manager,
        )

        # Highlight the selected polarization
        if self.scene_params["polarization"] == Polarization.PARALLEL:
            self.pol_parallel_button.disable()
        else:
            self.pol_perp_button.disable()

        # Scattering plot area (below controls, 33% of panel height)
        y_pos += spacing + 10  # Reduced extra spacing
        plot_y_start = y_pos
        plot_height = int(self.height * 0.30)  # Reduced from 0.33 to fit better
        plot_width = 360

        # Store plot dimensions for rendering
        self.plot_rect = pygame.Rect(panel_x, plot_y_start, plot_width, plot_height)

        # Add title for plot
        plot_title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, plot_y_start, plot_width, 30),
            text="Normalised Scattering Intensity",
            manager=self.ui_manager,
        )

    def _update_scene(self):
        """Update scene based on current parameters"""
        # Clear existing
        self.shapes = []
        self.ray_paths = []

        # Create shape based on selected type
        shape_class_map = {
            ShapeType.SQUARE: Square,
            ShapeType.TRIANGLE: Triangle,
            ShapeType.HEXAGON: Hexagon,
            ShapeType.OCTAGON: Octagon,
            ShapeType.CIRCLE: Circle,
        }

        shape_class = shape_class_map.get(self.scene_params["shape_type"], Square)
        shape = shape_class(
            center=(self.scene_params["position_x"], self.scene_params["position_y"]),
            size=self.scene_params["scale"],
            rotation=math.radians(self.scene_params["rotation_deg"]),
            refractive_index=complex(
                self.scene_params["refractive_index_real"],
                self.scene_params["refractive_index_imag"],
            ),
        )
        self.shapes.append(shape)

        # Compute plane wave position
        min_x = float("inf")
        for shape in self.shapes:
            bounds = shape.get_bounds()
            min_x = min(min_x, bounds[0])
        self.plane_wave_x = min_x - 1.0

        # Generate rays
        self._generate_rays()

    def _generate_rays(self):
        """Generate and trace rays"""
        num_rays = self.scene_params["num_rays"]
        offset = self.scene_params["plane_wave_offset"]

        if num_rays == 1:
            y_positions = [offset]
        else:
            y_positions = np.linspace(-1, 1, num_rays) + offset

        for y_pos in y_positions:
            ray = Ray(
                start=np.array([self.plane_wave_x, y_pos]),
                direction=np.array([1.0, 0.0]),
                electric_field=1.0,
                polarization=self.scene_params["polarization"],
                refractive_index=VACUUM_REFRACTIVE_INDEX,
                recursion_level=0,
            )

            ray_path = RayPath(ray)
            self._trace_ray(ray, ray_path)
            self.ray_paths.append(ray_path)

        # Calculate scattering distribution after all rays are traced
        self._calculate_scattering_distribution()

    def _calculate_scattering_distribution(self):
        """Calculate angular scattering distribution from scattered rays"""
        # Reset scattering intensity array
        self.scattering_intensity = np.zeros(181)

        # Collect scattered rays that hit the boundary
        for ray_path in self.ray_paths:
            for ray in ray_path.segments:
                # Only count rays with recursion > 0 that hit the boundary
                if ray.recursion_level > 0 and ray.hit_boundary:
                    # Calculate scattering angle from initial direction
                    # angle = arccos(ray.direction · initial_direction)
                    cos_angle = np.dot(ray.direction, self.initial_direction)
                    # Clamp to [-1, 1] to handle numerical errors
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle_rad)

                    # Clamp angles away from singularities at 0° and 180°
                    # where sin(theta) = 0 causes division by zero issues
                    if angle_deg < 0.5:
                        angle_deg = 0.5
                    elif angle_deg > 179.5:
                        angle_deg = 179.5

                    # Find the bin index (0-180 degrees)
                    bin_index = int(np.round(angle_deg))
                    bin_index = np.clip(bin_index, 0, 180)

                    # Calculate intensity: I = E^2 / 2
                    intensity = (ray.electric_field**2) / 2.0

                    # Add intensity to bin (we'll divide by sin(theta) after accumulation)
                    self.scattering_intensity[bin_index] += intensity

        # Calculate differential scattering cross-section for each bin
        # by dividing by the solid angle width (proportional to sin(theta))
        differential_cross_section = np.zeros(181)
        for i in range(181):
            angle_deg = float(i)
            # Clamp away from singularities
            if angle_deg < 0.5:
                angle_deg = 0.1
            elif angle_deg > 179.5:
                angle_deg = 179.9

            angle_rad = np.radians(angle_deg)
            sin_theta = np.sin(angle_rad)

            if sin_theta > 0:
                differential_cross_section[i] = self.scattering_intensity[i] / sin_theta

        # Normalize: multiply back by sin(theta) before summing
        # This way normalization is by total intensity (sin factors cancel)
        total_intensity = 0.0
        for i in range(181):
            angle_deg = float(i)
            if angle_deg < 0.5:
                angle_deg = 0.1
            elif angle_deg > 179.5:
                angle_deg = 179.9
            angle_rad = np.radians(angle_deg)
            sin_theta = np.sin(angle_rad)
            total_intensity += differential_cross_section[i] * sin_theta

        if total_intensity > 0:
            differential_cross_section /= total_intensity

        # Store the differential cross-section (this is what we plot)
        self.scattering_intensity = differential_cross_section

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

        if min_intersection is not None:
            ray.end = min_intersection
            ray.hit_boundary = is_boundary

            if (
                not is_boundary
                and ray.recursion_level < self.scene_params["max_recursion"]
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
        """Handle ray-shape intersection with Fresnel equations and absorption"""
        is_entering = ray.refractive_index == VACUUM_REFRACTIVE_INDEX

        if is_entering:
            n1 = VACUUM_REFRACTIVE_INDEX
            n2 = shape.refractive_index
        else:
            n1 = shape.refractive_index
            n2 = VACUUM_REFRACTIVE_INDEX

        # Calculate absorption if ray traveled through absorbing medium
        absorption_factor = 1.0
        if ray.refractive_index.imag > 0 and ray.end is not None:
            # Ray traveled through absorbing medium
            distance = np.linalg.norm(ray.end - ray.start)
            absorption_factor = calculate_absorption_factor(
                distance, ray.refractive_index.imag, self.scene_params["wavelength_nm"]
            )

        theta_i = calculate_angle_of_incidence(ray.direction, normal)
        theta_t = snells_law_refraction_angle(theta_i, n1, n2)

        r_coeff = fresnel_reflection_coefficient(
            n1, n2, theta_i, theta_t, ray.polarization
        )

        # Reflected ray (with absorption applied if it was inside absorbing medium)
        reflected_dir = get_reflection_vector(ray.direction, normal)
        reflected_ray = Ray(
            start=intersection_point,
            direction=reflected_dir,
            electric_field=ray.electric_field * abs(r_coeff) * absorption_factor,
            polarization=ray.polarization,
            refractive_index=n1,
            recursion_level=ray.recursion_level + 1,
        )
        self._trace_ray(reflected_ray, ray_path)
        ray_path.add_segment(reflected_ray)

        # Refracted ray (with absorption applied)
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
                electric_field=ray.electric_field * abs(t_coeff) * absorption_factor,
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

    def _draw_absorbing_ray(self, ray):
        """Draw a ray with gradient to show absorption"""
        # Number of segments to draw gradient
        num_segments = 20

        start_pos = ray.start
        end_pos = ray.end
        total_distance = np.linalg.norm(end_pos - start_pos)

        if total_distance <= 0:
            return

        # Get starting field amplitude
        initial_field = ray.electric_field

        # Draw ray as multiple segments with decreasing intensity
        for i in range(num_segments):
            # Calculate segment positions
            t1 = i / num_segments
            t2 = (i + 1) / num_segments
            seg_start = start_pos + t1 * (end_pos - start_pos)
            seg_end = start_pos + t2 * (end_pos - start_pos)

            # Calculate absorption at midpoint of segment
            mid_distance = (t1 + t2) / 2 * total_distance
            absorption = calculate_absorption_factor(
                mid_distance,
                ray.refractive_index.imag,
                self.scene_params["wavelength_nm"],
            )

            # Apply absorption to get field at this segment
            seg_field = initial_field * absorption
            intensity = calculate_intensity(seg_field)

            # Get color and thickness
            color = intensity_to_heat_color(intensity, self.max_intensity)
            thickness = max(1, min(3, int(1 + 2 * intensity / self.max_intensity)))

            # Draw segment
            pygame.draw.line(
                self.render_surface,
                color,
                self.world_to_screen(seg_start),
                self.world_to_screen(seg_end),
                thickness,
            )

    def _draw_scattering_plot(self):
        """Draw the angular scattering distribution plot"""
        # Define plot area (with margins)
        margin = 10
        plot_x = self.plot_rect.x + margin
        plot_y = self.plot_rect.y + 40  # Below title
        plot_w = self.plot_rect.width - 2 * margin
        plot_h = self.plot_rect.height - 50  # Account for title and bottom margin

        # Draw plot background
        pygame.draw.rect(self.screen, (40, 40, 40), (plot_x, plot_y, plot_w, plot_h))

        # Draw plot border
        pygame.draw.rect(
            self.screen, (100, 100, 100), (plot_x, plot_y, plot_w, plot_h), 1
        )

        # Find max intensity for scaling
        max_intensity = np.max(self.scattering_intensity)
        if max_intensity <= 0:
            # No scattering data yet, just show empty plot
            return

        # Draw grid lines
        grid_color = (60, 60, 60)
        # Horizontal grid lines
        for i in range(5):
            y = plot_y + int(i * plot_h / 4)
            pygame.draw.line(
                self.screen, grid_color, (plot_x, y), (plot_x + plot_w, y), 1
            )

        # Vertical grid lines (every 30 degrees)
        for angle in [0, 30, 60, 90, 120, 150, 180]:
            x = plot_x + int((angle / 180.0) * plot_w)
            pygame.draw.line(
                self.screen, grid_color, (x, plot_y), (x, plot_y + plot_h), 1
            )

        # Draw the scattering data with logarithmic y-axis
        points = []
        min_intensity_log = max_intensity / 1000.0  # 3 orders of magnitude

        for i, intensity in enumerate(self.scattering_intensity):
            angle_deg = i  # Angle in degrees (0-180)
            x = plot_x + int((angle_deg / 180.0) * plot_w)

            # Use minimum log value for zero or very small intensities
            plot_intensity = max(intensity, min_intensity_log)

            # Logarithmic scaling for y-axis
            log_intensity = np.log10(plot_intensity)
            log_max = np.log10(max_intensity)
            log_min = np.log10(min_intensity_log)

            # Normalize to [0, 1] range
            normalized = (log_intensity - log_min) / (log_max - log_min)
            normalized = np.clip(normalized, 0.0, 1.0)

            # Invert y-axis so high values are at top
            y = plot_y + plot_h - int(normalized * plot_h)
            points.append((x, y))

        # Draw line connecting points
        if len(points) > 1:
            pygame.draw.lines(self.screen, (0, 255, 100), False, points, 2)

        # Draw axis labels
        label_color = (200, 200, 200)

        # X-axis labels (angles)
        for angle in [0, 30, 60, 90, 120, 150, 180]:
            x = plot_x + int((angle / 180.0) * plot_w)
            label = self.font.render(f"{angle}°", True, label_color)
            self.screen.blit(label, (x - 10, plot_y + plot_h + 5))

        # Y-axis labels (log scale - show max and min)
        y_label_max = self.font.render(f"{max_intensity:.2e}", True, label_color)
        self.screen.blit(y_label_max, (plot_x - 5, plot_y - 15))

        y_label_min = self.font.render(f"{min_intensity_log:.2e}", True, label_color)
        self.screen.blit(y_label_min, (plot_x - 5, plot_y + plot_h - 10))

    def render_scene(self):
        """Render the ray tracing scene to the render surface"""
        self.render_surface.fill((0, 0, 0))

        # Draw axes
        pygame.draw.line(
            self.render_surface,
            (50, 50, 50),
            self.world_to_screen(np.array([-10, 0])),
            self.world_to_screen(np.array([10, 0])),
            1,
        )
        pygame.draw.line(
            self.render_surface,
            (50, 50, 50),
            self.world_to_screen(np.array([0, -10])),
            self.world_to_screen(np.array([0, 10])),
            1,
        )

        # Draw boundary
        for edge_start, edge_end in self._get_boundary_edges():
            pygame.draw.line(
                self.render_surface,
                (40, 40, 40),
                self.world_to_screen(edge_start),
                self.world_to_screen(edge_end),
                1,
            )

        # Draw plane wave
        pygame.draw.line(
            self.render_surface,
            (100, 100, 255),
            self.world_to_screen(
                np.array(
                    [self.plane_wave_x, -1 + self.scene_params["plane_wave_offset"]]
                )
            ),
            self.world_to_screen(
                np.array(
                    [self.plane_wave_x, 1 + self.scene_params["plane_wave_offset"]]
                )
            ),
            2,
        )

        # Draw shapes
        for shape in self.shapes:
            vertices = shape.transformed_vertices
            screen_points = [self.world_to_screen(v) for v in vertices]
            pygame.draw.polygon(self.render_surface, (255, 255, 255), screen_points, 2)

            center_screen = self.world_to_screen(shape.center)
            pygame.draw.circle(self.render_surface, (255, 100, 100), center_screen, 3)

        # Draw rays with gradient for absorption
        for ray_path in self.ray_paths:
            for ray in ray_path.segments:
                if ray.end is not None:
                    # Skip rays that didn't hit any particle (recursion level 0 and hit boundary)
                    if ray.recursion_level == 0 and ray.hit_boundary:
                        continue

                    # Check if ray is in absorbing medium
                    if ray.refractive_index.imag > 0:
                        # Draw ray with gradient showing absorption
                        self._draw_absorbing_ray(ray)
                    else:
                        # Draw normal ray
                        intensity = calculate_intensity(ray.electric_field)
                        color = intensity_to_heat_color(intensity, self.max_intensity)
                        thickness = max(
                            1, min(3, int(1 + 2 * intensity / self.max_intensity))
                        )
                        pygame.draw.line(
                            self.render_surface,
                            color,
                            self.world_to_screen(ray.start),
                            self.world_to_screen(ray.end),
                            thickness,
                        )

        # Draw colorscale
        draw_colorscale(
            self.render_surface, self.font, 20, 40, 20, 100, self.max_intensity
        )

    def handle_ui_event(self, event):
        """Handle UI events and update parameters"""
        update_needed = False

        # Handle slider changes
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.pos_x_slider:
                self.scene_params["position_x"] = event.value
                self.pos_x_label.set_text(f"Position X: {event.value:.2f}")
                update_needed = True
            elif event.ui_element == self.pos_y_slider:
                self.scene_params["position_y"] = event.value
                self.pos_y_label.set_text(f"Position Y: {event.value:.2f}")
                update_needed = True
            elif event.ui_element == self.rotation_slider:
                self.scene_params["rotation_deg"] = event.value
                self.rotation_label.set_text(f"Rotation: {event.value:.1f}°")
                update_needed = True
            elif event.ui_element == self.scale_slider:
                self.scene_params["scale"] = event.value
                self.scale_label.set_text(f"Scale: {event.value:.2f}")
                update_needed = True
            elif event.ui_element == self.n_real_slider:
                self.scene_params["refractive_index_real"] = event.value
                self.n_real_label.set_text(f"n (real): {event.value:.3f}")
                update_needed = True
            elif event.ui_element == self.n_imag_slider:
                self.scene_params["refractive_index_imag"] = event.value
                self.n_imag_label.set_text(f"n (imag): {event.value:.3f}")
                update_needed = True
            elif event.ui_element == self.wavelength_slider:
                self.scene_params["wavelength_nm"] = event.value
                self.wavelength_label.set_text(f"Wavelength: {event.value:.0f} nm")
                update_needed = True
            elif event.ui_element == self.rays_slider:
                self.scene_params["num_rays"] = int(event.value)
                self.rays_label.set_text(f"Number of Rays: {int(event.value)}")
                update_needed = True
            elif event.ui_element == self.offset_slider:
                self.scene_params["plane_wave_offset"] = event.value
                self.offset_label.set_text(f"Wave Y Offset: {event.value:.2f}")
                update_needed = True
            elif event.ui_element == self.recursion_slider:
                self.scene_params["max_recursion"] = int(event.value)
                self.recursion_label.set_text(f"Max Recursion: {int(event.value)}")
                update_needed = True

        # Handle button clicks
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.pol_parallel_button:
                self.scene_params["polarization"] = Polarization.PARALLEL
                self.pol_parallel_button.disable()
                self.pol_perp_button.enable()
                update_needed = True
            elif event.ui_element == self.pol_perp_button:
                self.scene_params["polarization"] = Polarization.PERPENDICULAR
                self.pol_perp_button.disable()
                self.pol_parallel_button.enable()
                update_needed = True

        # Handle dropdown
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.shape_dropdown:
                # Map dropdown text to shape type
                shape_map = {
                    "Square": ShapeType.SQUARE,
                    "Triangle": ShapeType.TRIANGLE,
                    "Hexagon": ShapeType.HEXAGON,
                    "Octagon": ShapeType.OCTAGON,
                    "Circle": ShapeType.CIRCLE,
                }
                self.scene_params["shape_type"] = shape_map.get(
                    event.text, ShapeType.SQUARE
                )
                update_needed = True

        if update_needed:
            self._update_scene()

    def run(self):
        """Main application loop"""
        running = True

        while running:
            time_delta = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                # Process GUI events
                self.handle_ui_event(event)
                self.ui_manager.process_events(event)

            # Update GUI
            self.ui_manager.update(time_delta)

            # Render scene to surface
            self.render_scene()

            # Clear screen
            self.screen.fill((30, 30, 30))

            # Blit render surface to screen
            self.screen.blit(self.render_surface, (0, 0))

            # Draw vertical separator
            pygame.draw.line(
                self.screen,
                (100, 100, 100),
                (self.render_width, 0),
                (self.render_width, self.height),
                2,
            )

            # Draw GUI
            self.ui_manager.draw_ui(self.screen)

            # Draw scattering plot (after GUI so it appears on top)
            self._draw_scattering_plot()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    app = IntegratedRayTracingApp()
    app.run()
