"""
Integrated GUI using pygame-gui for seamless control panel
"""

import pygame
import pygame_gui
import numpy as np
import math
from typing import Dict, Any
from config import ShapeType, Polarization, VACUUM_REFRACTIVE_INDEX
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
            "position_y": 0.0,
            "rotation_deg": 30.0,
            "refractive_index": 1.31,
            "num_rays": 10,
            "plane_wave_offset": 0.0,
            "polarization": Polarization.PARALLEL,
            "max_recursion": 3,
            "shape_type": ShapeType.SQUARE,
        }

        # Scene objects
        self.shapes = []
        self.ray_paths = []
        self.plane_wave_x = -2.0
        self.bound_size = 10.0
        self.max_intensity = 0.5

        # Create GUI controls
        self._create_gui_controls()

        # Initial scene update
        self._update_scene()

        # Clock for FPS
        self.clock = pygame.time.Clock()

    def _create_gui_controls(self):
        """Create all GUI control elements using pygame-gui"""
        panel_x = self.render_width + 20
        y_pos = 20
        label_width = 150
        control_width = 200
        height = 30
        spacing = 40

        # Title
        title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, 350, 30),
            text="Scene Controls",
            manager=self.ui_manager,
        )
        y_pos += 50

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

        # Refractive Index Slider
        self.n_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text=f"Refractive Index: {self.scene_params['refractive_index']:.2f}",
            manager=self.ui_manager,
        )
        self.n_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            start_value=self.scene_params["refractive_index"],
            value_range=(1.0, 2.5),
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
        y_pos += spacing + 20

        # Shape Dropdown
        shape_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text="Shape:",
            manager=self.ui_manager,
        )
        self.shape_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["Square"],
            starting_option="Square",
            relative_rect=pygame.Rect(
                panel_x + label_width, y_pos, control_width, height
            ),
            manager=self.ui_manager,
        )
        y_pos += spacing + 20

        # Polarization Selection
        pol_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_pos, label_width, height),
            text="Polarization:",
            manager=self.ui_manager,
        )
        y_pos += 35

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

    def _update_scene(self):
        """Update scene based on current parameters"""
        # Clear existing
        self.shapes = []
        self.ray_paths = []

        # Create shape
        shape = Square(
            center=(self.scene_params["position_x"], self.scene_params["position_y"]),
            size=1.0,
            rotation=math.radians(self.scene_params["rotation_deg"]),
            refractive_index=complex(self.scene_params["refractive_index"], 0.0),
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
        """Handle ray-shape intersection with Fresnel equations"""
        is_entering = ray.refractive_index == VACUUM_REFRACTIVE_INDEX

        if is_entering:
            n1 = VACUUM_REFRACTIVE_INDEX
            n2 = shape.refractive_index
        else:
            n1 = shape.refractive_index
            n2 = VACUUM_REFRACTIVE_INDEX

        theta_i = calculate_angle_of_incidence(ray.direction, normal)
        theta_t = snells_law_refraction_angle(theta_i, n1, n2)

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
            elif event.ui_element == self.n_slider:
                self.scene_params["refractive_index"] = event.value
                self.n_label.set_text(f"Refractive Index: {event.value:.2f}")
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
                shape_map = {"Square": ShapeType.SQUARE}
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

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    app = IntegratedRayTracingApp()
    app.run()
