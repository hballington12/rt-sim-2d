import pygame
import numpy as np
from typing import Dict, Any, Callable
import math
from scene import Scene
from config import ShapeConfig, ShapeType
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


class GUIScene(Scene):
    """Scene class adapted for GUI control"""

    def __init__(self, params: Dict[str, Any]):
        """Initialize scene with GUI parameters"""
        # Store parameters
        self.gui_params = params

        # Initialize shapes list and other attributes
        self.shapes = []
        self.plane_wave_x = 0.0
        self.plane_wave_y_offset = params.get("plane_wave_offset", 0.0)
        self.num_rays = params.get("num_rays", 10)
        self.polarization = params.get("polarization")
        self.max_recursion_depth = params.get("max_recursion", 3)

        # Set up shape from parameters
        self._setup_shapes_from_params()
        self._compute_plane_wave_position()

        # Scene bounds
        self.bound_size = 10.0
        self._compute_scene_bounds()

        # Ray paths
        self.ray_paths = []
        self._generate_initial_rays_with_offset()

        # Pygame setup
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("2D Ray Tracing")

        # Font for UI text
        self.font = pygame.font.Font(None, 16)

        # View parameters
        self.scale = 100
        self.offset = np.array([self.width / 2, self.height / 2])

        # Track maximum intensity
        self.max_intensity = 0.5

    def _setup_shapes_from_params(self):
        """Create shapes from GUI parameters"""
        shape_type = self.gui_params.get("shape_type", ShapeType.SQUARE)

        if shape_type == ShapeType.SQUARE:
            from scene import Square

            shape = Square(
                center=(
                    self.gui_params.get("position_x", 0.0),
                    self.gui_params.get("position_y", 0.0),
                ),
                size=1.0,  # Fixed size for now
                rotation=math.radians(self.gui_params.get("rotation_deg", 0.0)),
                refractive_index=complex(
                    self.gui_params.get("refractive_index_real", 1.31),
                    self.gui_params.get("refractive_index_imag", 0.0),
                ),
            )
            self.shapes.append(shape)

    def _generate_initial_rays_with_offset(self):
        """Generate initial rays with Y offset from plane wave"""
        if self.num_rays <= 0:
            return

        # Linear spacing from -1 to 1 centered at y offset
        if self.num_rays == 1:
            y_positions = [self.plane_wave_y_offset]
        else:
            y_positions = np.linspace(-1, 1, self.num_rays) + self.plane_wave_y_offset

        # Create rays propagating in +x direction
        for y_pos in y_positions:
            start_point = np.array([self.plane_wave_x, y_pos])
            direction = np.array([1.0, 0.0])  # Propagating right

            ray = Ray(
                start=start_point,
                direction=direction,
                electric_field=1.0,
                polarization=self.polarization,
                refractive_index=complex(1.0, 0.0),  # Vacuum
                recursion_level=0,
            )

            # Create ray path
            ray_path = RayPath(ray)

            # Trace this ray
            self._trace_ray_gui(ray, ray_path)

            self.ray_paths.append(ray_path)

    def _trace_ray_gui(self, ray: Ray, ray_path: RayPath):
        """Trace ray with GUI-controlled max recursion"""
        # Find intersection
        min_t = float("inf")
        min_intersection = None
        intersected_shape = None
        intersected_normal = None
        is_boundary = False

        # Check all shape edges
        for shape in self.shapes:
            edges = shape.get_edges()
            normals = shape.get_edge_normals()
            for i, (edge_start, edge_end) in enumerate(edges):
                t, intersection = self._ray_edge_intersection(
                    ray.start, ray.direction, edge_start, edge_end
                )
                if t is not None and 0 < t < min_t:
                    min_t = t
                    min_intersection = intersection
                    intersected_shape = shape
                    intersected_normal = normals[i]

        # Check scene boundary edges
        boundary_edges = self._get_boundary_edges()
        for edge_start, edge_end in boundary_edges:
            t, intersection = self._ray_edge_intersection(
                ray.start, ray.direction, edge_start, edge_end
            )
            if t is not None and 0 < t < min_t:
                min_t = t
                min_intersection = intersection
                is_boundary = True
                intersected_shape = None
                intersected_normal = None

        # Set ray endpoint
        if min_intersection is not None:
            ray.end = min_intersection

            # Handle reflection/refraction if not at boundary and under recursion limit
            if (
                not is_boundary
                and ray.recursion_level < self.max_recursion_depth
                and intersected_shape
            ):
                self._handle_intersection_gui(
                    ray,
                    ray_path,
                    intersected_shape,
                    intersected_normal,
                    min_intersection,
                )

    def _handle_intersection_gui(
        self,
        ray: Ray,
        ray_path: RayPath,
        shape,
        normal: np.ndarray,
        intersection_point: np.ndarray,
    ):
        """Handle intersection with GUI parameters"""
        # Determine if ray is entering or exiting
        is_entering = ray.refractive_index.real == 1.0  # Coming from vacuum

        if is_entering:
            n1 = complex(1.0, 0.0)
            n2 = shape.refractive_index
        else:
            n1 = shape.refractive_index
            n2 = complex(1.0, 0.0)

        # Calculate angles
        theta_i = calculate_angle_of_incidence(ray.direction, normal)
        theta_t = snells_law_refraction_angle(theta_i, n1, n2)

        # Calculate Fresnel coefficients
        r_coeff = fresnel_reflection_coefficient(
            n1, n2, theta_i, theta_t, ray.polarization
        )

        # Create reflected ray
        reflected_dir = get_reflection_vector(ray.direction, normal)
        reflected_field = ray.electric_field * abs(r_coeff)

        reflected_ray = Ray(
            start=intersection_point,
            direction=reflected_dir,
            electric_field=reflected_field,
            polarization=ray.polarization,
            refractive_index=n1,
            recursion_level=ray.recursion_level + 1,
        )

        # Trace reflected ray
        self._trace_ray_gui(reflected_ray, ray_path)
        ray_path.add_segment(reflected_ray)

        # Create refracted ray if no TIR
        if theta_t is not None:
            t_coeff = fresnel_transmission_coefficient(
                n1, n2, theta_i, theta_t, ray.polarization
            )
            refracted_dir = get_refraction_vector(
                ray.direction, normal, theta_i, theta_t
            )
            refracted_field = ray.electric_field * abs(t_coeff)

            refracted_ray = Ray(
                start=intersection_point,
                direction=refracted_dir,
                electric_field=refracted_field,
                polarization=ray.polarization,
                refractive_index=n2,
                recursion_level=ray.recursion_level + 1,
            )

            # Trace refracted ray
            self._trace_ray_gui(refracted_ray, ray_path)
            ray_path.add_segment(refracted_ray)

    def run(self, running_check: Callable[[], bool]):
        """Run the scene with a callback to check if still running"""
        clock = pygame.time.Clock()

        while running_check():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return

            self.render()
            clock.tick(30)  # 30 FPS for GUI

        pygame.quit()
