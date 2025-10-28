import pygame
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from config import (
    NUM_RAYS,
    POLARIZATION,
    PLANE_WAVE_MARGIN,
    SHAPES,
    ShapeType,
    ShapeConfig,
    MAX_RECURSION,
    VACUUM_REFRACTIVE_INDEX,
)
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


class Shape:
    """Base class for 2D shapes"""

    def __init__(
        self,
        center: Tuple[float, float],
        size: float,
        rotation: float,
        refractive_index: complex,
    ):
        self.center = np.array(center)
        self.size = size
        self.rotation = rotation  # in radians
        self.refractive_index = refractive_index
        self.base_vertices = self._get_base_vertices()
        self.transformed_vertices = self._compute_transformed_vertices()

    def _get_base_vertices(self) -> np.ndarray:
        """Get base vertices before transformation"""
        raise NotImplementedError

    def _compute_transformed_vertices(self) -> np.ndarray:
        """Apply rotation and translation to base vertices"""
        # Create rotation matrix
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        # Scale, rotate, then translate
        scaled = self.base_vertices * self.size
        rotated = scaled @ rotation_matrix.T
        translated = rotated + self.center

        return translated

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, max_x, min_y, max_y)"""
        verts = self.transformed_vertices
        return (
            np.min(verts[:, 0]),
            np.max(verts[:, 0]),
            np.min(verts[:, 1]),
            np.max(verts[:, 1]),
        )

    def get_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get list of edges as (start_point, end_point) pairs"""
        edges = []
        verts = self.transformed_vertices
        n_verts = len(verts)
        for i in range(n_verts):
            edges.append((verts[i], verts[(i + 1) % n_verts]))
        return edges

    def get_edge_normals(self) -> List[np.ndarray]:
        """Get outward-pointing normals for each edge"""
        normals = []
        edges = self.get_edges()
        for start, end in edges:
            edge_vec = end - start
            # Rotate 90 degrees counterclockwise for outward normal
            normal = np.array([-edge_vec[1], edge_vec[0]])
            normal = normal / np.linalg.norm(normal)

            # Check if normal points outward by testing with center
            mid_point = (start + end) / 2
            if np.dot(normal, mid_point - self.center) < 0:
                normal = -normal
            normals.append(normal)
        return normals


class Square(Shape):
    """Square shape with unit size before scaling"""

    def _get_base_vertices(self) -> np.ndarray:
        """Unit square centered at origin"""
        return np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


class Scene:
    """Main scene containing shapes and handling ray tracing"""

    def __init__(self):
        self.shapes: List[Shape] = []
        self.plane_wave_x = 0.0
        self.num_rays = NUM_RAYS
        self.polarization = POLARIZATION
        self._setup_shapes()
        self._compute_plane_wave_position()

        # Scene bounds (will be computed after shapes are set up)
        self.bound_size = 10.0  # Default scene bound size
        self._compute_scene_bounds()

        # Ray paths
        self.ray_paths: List[RayPath] = []
        self._generate_initial_rays()

        # Pygame setup
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2D Ray Tracing")

        # Font for UI text
        self.font = pygame.font.Font(None, 16)

        # View parameters
        self.scale = 100  # pixels per unit
        self.offset = np.array([self.width / 2, self.height / 2])

        # Track maximum intensity for colorscale
        self.max_intensity = 0.5  # Will be updated based on actual rays

    def _setup_shapes(self):
        """Create shape instances from config"""
        for shape_config in SHAPES:
            if shape_config.shape_type == ShapeType.SQUARE:
                shape = Square(
                    center=shape_config.center,
                    size=shape_config.size,
                    rotation=shape_config.rotation,
                    refractive_index=shape_config.refractive_index,
                )
                self.shapes.append(shape)

    def _compute_plane_wave_position(self):
        """Determine x position for incident plane wave"""
        if not self.shapes:
            self.plane_wave_x = -2.0
            return

        # Find leftmost x coordinate of all shapes
        min_x = float("inf")
        for shape in self.shapes:
            bounds = shape.get_bounds()
            min_x = min(min_x, bounds[0])

        # Position plane wave to the left with margin
        self.plane_wave_x = min_x - PLANE_WAVE_MARGIN

    def _compute_scene_bounds(self):
        """Compute scene boundary size to contain all shapes and rays"""
        if not self.shapes:
            return

        # Find extent of all shapes
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")

        for shape in self.shapes:
            bounds = shape.get_bounds()
            min_x = min(min_x, bounds[0])
            max_x = max(max_x, bounds[1])
            min_y = min(min_y, bounds[2])
            max_y = max(max_y, bounds[3])

        # Add margin to bounds
        margin = 3.0
        self.bound_size = max(
            abs(min_x - margin),
            abs(max_x + margin),
            abs(min_y - margin),
            abs(max_y + margin),
            abs(self.plane_wave_x - margin),
        )

    def _generate_initial_rays(self):
        """Generate initial rays from plane wave with linear spacing"""
        if self.num_rays <= 0:
            return

        # Linear spacing from -1 to 1 centered at y=0
        if self.num_rays == 1:
            y_positions = [0.0]
        else:
            y_positions = np.linspace(-1, 1, self.num_rays)

        # Create rays propagating in +x direction
        for y_pos in y_positions:
            start_point = np.array([self.plane_wave_x, y_pos])
            direction = np.array([1.0, 0.0])  # Propagating right

            ray = Ray(
                start=start_point,
                direction=direction,
                electric_field=1.0,  # Unit amplitude initially
                polarization=self.polarization,
                refractive_index=VACUUM_REFRACTIVE_INDEX,
                recursion_level=0,
            )

            # Create ray path with this initial segment
            ray_path = RayPath(ray)

            # Trace this ray to find its first intersection and handle recursion
            self._trace_ray(ray, ray_path)

            self.ray_paths.append(ray_path)

    def _trace_ray(self, ray: Ray, ray_path: Optional[RayPath] = None):
        """Find intersection point for a ray and set its endpoint, handle reflection/refraction"""
        min_t = float("inf")
        min_intersection = None
        intersected_shape = None
        intersected_edge = None
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
                    intersected_edge = (edge_start, edge_end)
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
                intersected_edge = None
                intersected_normal = None

        # Set ray endpoint
        if min_intersection is not None:
            ray.end = min_intersection

            # Handle reflection/refraction if not at boundary and under recursion limit
            if (
                not is_boundary
                and ray.recursion_level < MAX_RECURSION
                and ray_path is not None
            ):
                self._handle_intersection(
                    ray,
                    ray_path,
                    intersected_shape,
                    intersected_normal,
                    min_intersection,
                )

    def _ray_edge_intersection(
        self,
        ray_start: np.ndarray,
        ray_dir: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Calculate ray-edge intersection using parametric equations"""
        # Ray: P = ray_start + t * ray_dir
        # Edge: P = edge_start + s * (edge_end - edge_start)
        # Solve: ray_start + t * ray_dir = edge_start + s * edge_vec

        edge_vec = edge_end - edge_start

        # Create matrix equation: [ray_dir, -edge_vec] * [t, s]^T = edge_start - ray_start
        A = np.column_stack([ray_dir, -edge_vec])
        b = edge_start - ray_start

        # Check if matrix is singular (parallel lines)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-10:
            return None, None

        # Solve for t and s
        params = np.linalg.solve(A, b)
        t, s = params[0], params[1]

        # Check if intersection is within edge bounds (0 <= s <= 1) and ray is forward (t > 0)
        if 0 <= s <= 1 and t > 1e-10:  # Small epsilon to avoid self-intersection
            intersection = ray_start + t * ray_dir
            return t, intersection

        return None, None

    def _get_boundary_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get edges of scene boundary box"""
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

    def _handle_intersection(
        self,
        ray: Ray,
        ray_path: RayPath,
        shape: Shape,
        normal: np.ndarray,
        intersection_point: np.ndarray,
    ):
        """Handle reflection and refraction at an intersection"""
        # Determine if ray is entering or exiting the shape
        # Ray is entering if it's coming from vacuum to shape
        is_entering = ray.refractive_index == VACUUM_REFRACTIVE_INDEX

        # Set refractive indices
        if is_entering:
            n1 = VACUUM_REFRACTIVE_INDEX
            n2 = shape.refractive_index
        else:
            n1 = shape.refractive_index
            n2 = VACUUM_REFRACTIVE_INDEX

        # Calculate angle of incidence
        theta_i = calculate_angle_of_incidence(ray.direction, normal)

        # Calculate refraction angle using Snell's law
        theta_t = snells_law_refraction_angle(theta_i, n1, n2)

        # Calculate Fresnel coefficients
        r_coeff = fresnel_reflection_coefficient(
            n1, n2, theta_i, theta_t, ray.polarization
        )

        # Create reflected ray (always happens)
        reflected_dir = get_reflection_vector(ray.direction, normal)
        reflected_field = ray.electric_field * abs(
            r_coeff
        )  # Apply Fresnel reflection coefficient

        reflected_ray = Ray(
            start=intersection_point,
            direction=reflected_dir,
            electric_field=reflected_field,
            polarization=ray.polarization,  # Same polarization as incident ray
            refractive_index=n1,  # Stays in same medium
            recursion_level=ray.recursion_level + 1,
        )

        # Trace reflected ray
        self._trace_ray(reflected_ray, ray_path)
        ray_path.add_segment(reflected_ray)

        # Create refracted ray if no total internal reflection
        if theta_t is not None:
            t_coeff = fresnel_transmission_coefficient(
                n1, n2, theta_i, theta_t, ray.polarization
            )
            refracted_dir = get_refraction_vector(
                ray.direction, normal, theta_i, theta_t
            )
            refracted_field = ray.electric_field * abs(
                t_coeff
            )  # Apply Fresnel transmission coefficient

            refracted_ray = Ray(
                start=intersection_point,
                direction=refracted_dir,
                electric_field=refracted_field,
                polarization=ray.polarization,  # Same polarization as incident ray
                refractive_index=n2,  # Enters new medium
                recursion_level=ray.recursion_level + 1,
            )

            # Trace refracted ray
            self._trace_ray(refracted_ray, ray_path)
            ray_path.add_segment(refracted_ray)

    def world_to_screen(self, point: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_point = point * np.array([1, -1]) * self.scale + self.offset
        return tuple(screen_point.astype(int))

    def render(self):
        """Render the scene"""
        self.screen.fill((0, 0, 0))  # Black background

        # Draw coordinate axes
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

        # Draw scene boundary
        boundary_edges = self._get_boundary_edges()
        for edge_start, edge_end in boundary_edges:
            pygame.draw.line(
                self.screen,
                (40, 40, 40),  # Dark gray
                self.world_to_screen(edge_start),
                self.world_to_screen(edge_end),
                1,
            )

        # Draw plane wave line
        plane_wave_color = (100, 100, 255)  # Light blue
        pygame.draw.line(
            self.screen,
            plane_wave_color,
            self.world_to_screen(np.array([self.plane_wave_x, -1])),
            self.world_to_screen(np.array([self.plane_wave_x, 1])),
            2,
        )

        # Draw shapes
        for shape in self.shapes:
            self._draw_shape(shape)

        # Draw rays
        for ray_path in self.ray_paths:
            self._draw_ray_path(ray_path)

        # Draw colorscale legend in top-left
        draw_colorscale(self.screen, self.font, 20, 40, 20, 100, self.max_intensity)

        pygame.display.flip()

    def _draw_shape(self, shape: Shape):
        """Draw a shape outline"""
        vertices = shape.transformed_vertices
        screen_points = [self.world_to_screen(v) for v in vertices]

        # Draw polygon outline
        pygame.draw.polygon(self.screen, (255, 255, 255), screen_points, 2)

        # Draw center point
        center_screen = self.world_to_screen(shape.center)
        pygame.draw.circle(self.screen, (255, 100, 100), center_screen, 3)

    def _draw_ray_path(self, ray_path: RayPath):
        """Draw all ray segments in a path"""
        for ray in ray_path.segments:
            if ray.end is not None:
                # Calculate intensity from electric field
                intensity = calculate_intensity(ray.electric_field)

                # Get heat color based on intensity
                ray_color = intensity_to_heat_color(intensity, self.max_intensity)

                # Draw ray line with thickness based on intensity (optional enhancement)
                thickness = max(1, min(3, int(1 + 2 * intensity / self.max_intensity)))

                pygame.draw.line(
                    self.screen,
                    ray_color,
                    self.world_to_screen(ray.start),
                    self.world_to_screen(ray.end),
                    thickness,
                )

    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            self.render()
            clock.tick(60)  # 60 FPS

        pygame.quit()


if __name__ == "__main__":
    scene = Scene()
    scene.run()
