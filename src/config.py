from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


class Polarization(Enum):
    PARALLEL = "parallel"  # In x-y plane, perpendicular to propagation
    PERPENDICULAR = "perpendicular"  # In z direction (out of plane)


class ShapeType(Enum):
    SQUARE = "square"


@dataclass
class ShapeConfig:
    """Configuration for a shape in the scene"""

    shape_type: ShapeType
    center: Tuple[float, float] = (0.0, 0.0)  # Center of mass position
    size: float = 1.0  # Unit size
    rotation: float = 0.0  # Rotation in radians
    refractive_index: complex = 1.31 + 0j  # Default refractive index


# Scene configuration
NUM_RAYS = 100  # Increased for better visualization
POLARIZATION = Polarization.PERPENDICULAR
PLANE_WAVE_MARGIN = 1.0  # Distance margin from leftmost shape
MAX_RECURSION = 3  # Maximum recursion depth for ray tracing
VACUUM_REFRACTIVE_INDEX = 1.0 + 0j  # Refractive index of vacuum/air

# Shape configurations
# Using config objects rather than instances for flexibility with GUI later
SHAPES = [
    ShapeConfig(
        shape_type=ShapeType.SQUARE,
        center=(0.0, 0.0),
        size=1.0,
        rotation=math.radians(30),  # 30 degrees in radians
        refractive_index=1.31 + 0j,
    )
]
