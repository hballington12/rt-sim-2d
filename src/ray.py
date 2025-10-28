import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from config import Polarization


@dataclass
class Ray:
    """Single ray segment with position, direction, and field properties"""

    start: np.ndarray  # Starting point (x, y)
    direction: np.ndarray  # Unit vector for propagation direction
    electric_field: float  # Electric field amplitude (magnitude)
    polarization: Polarization  # PARALLEL (in x-y plane) or PERPENDICULAR (z direction)
    refractive_index: complex  # Refractive index of the medium the ray is in
    recursion_level: int = 0  # Recursion depth for this ray
    end: Optional[np.ndarray] = None  # End point after intersection
    hit_boundary: bool = False  # True if ray hit scene boundary rather than a shape

    def __post_init__(self):
        # Ensure direction is normalized
        self.direction = self.direction / np.linalg.norm(self.direction)

    def get_point_at_t(self, t: float) -> np.ndarray:
        """Get point along ray at parameter t"""
        return self.start + t * self.direction


class RayPath:
    """Complete path of a ray through the scene, including all segments"""

    def __init__(self, initial_ray: Ray):
        self.segments: List[Ray] = [initial_ray]

    def add_segment(self, ray: Ray):
        """Add a new ray segment to the path (after reflection/refraction)"""
        self.segments.append(ray)

    def get_current_ray(self) -> Ray:
        """Get the current (last) ray segment"""
        return self.segments[-1]
