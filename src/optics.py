import numpy as np
from typing import Tuple, Optional
import math
from config import Polarization


def snells_law_refraction_angle(
    theta_i: float, n1: complex, n2: complex
) -> Optional[float]:
    """
    Calculate refraction angle using Snell's law (from Rust implementation).
    Returns None if total internal reflection occurs.

    Args:
        theta_i: Incident angle in radians
        n1: Complex refractive index of incident medium
        n2: Complex refractive index of transmitted medium

    Returns:
        Refraction angle in radians, or None if TIR
    """
    if n1 == n2:
        return theta_i

    # Real refractive index ratio
    n_ratio = n2.real / n1.real

    # Check for total internal reflection (only when going from higher to lower index)
    sin_theta_i = math.sin(theta_i)
    if n1.real > n2.real:
        critical_angle_sin = n_ratio
        if sin_theta_i > critical_angle_sin:
            return None  # Total internal reflection

    # Calculate transmitted angle (simplified for non-absorbing media)
    sin_theta_t = (n1.real / n2.real) * sin_theta_i

    # Check if result is valid
    if abs(sin_theta_t) > 1.0:
        return None  # Total internal reflection

    theta_t = math.asin(sin_theta_t)
    return theta_t


def get_reflection_vector(incident_dir: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Calculate reflection direction vector.

    Args:
        incident_dir: Normalized incident direction vector
        normal: Normalized surface normal (pointing outward)

    Returns:
        Normalized reflected direction vector
    """
    # Ensure normal points against incident direction (into the surface)
    if np.dot(normal, incident_dir) > 0:
        n = -normal
    else:
        n = normal

    # Reflection formula: r = d - 2(d·n)n
    cos_theta_i = -np.dot(incident_dir, n)
    reflected = incident_dir + 2.0 * cos_theta_i * n

    return reflected / np.linalg.norm(reflected)


def get_refraction_vector(
    incident_dir: np.ndarray, normal: np.ndarray, theta_i: float, theta_t: float
) -> np.ndarray:
    """
    Calculate refraction direction vector (adapted from Rust beam.rs).

    Args:
        incident_dir: Normalized incident direction vector
        normal: Normalized surface normal
        theta_i: Incident angle in radians
        theta_t: Transmitted angle in radians

    Returns:
        Normalized refracted direction vector
    """
    # Ensure normal points against incident direction
    if np.dot(normal, incident_dir) > 0:
        n = -normal
    else:
        n = normal

    # If angles are very small, propagation continues in same direction
    if abs(math.sin(theta_t)) < 1e-6:
        return incident_dir

    # Refraction formula adapted for 2D
    cos_theta_i = -np.dot(incident_dir, n)
    cos_theta_t = math.cos(theta_t)

    # Calculate refracted direction
    n_ratio = math.sin(theta_t) / math.sin(theta_i) if math.sin(theta_i) > 1e-6 else 1.0
    refracted = n_ratio * incident_dir + (n_ratio * cos_theta_i - cos_theta_t) * n

    return refracted / np.linalg.norm(refracted)


def calculate_angle_of_incidence(incident_dir: np.ndarray, normal: np.ndarray) -> float:
    """
    Calculate angle of incidence between ray and surface normal.

    Args:
        incident_dir: Normalized incident direction vector
        normal: Normalized surface normal

    Returns:
        Angle of incidence in radians (0 to π/2)
    """
    # Ensure normal points against incident direction
    if np.dot(normal, incident_dir) > 0:
        n = -normal
    else:
        n = normal

    # Calculate cosine of angle (absolute value)
    cos_theta = abs(np.dot(incident_dir, n))
    cos_theta = min(1.0, max(-1.0, cos_theta))  # Clamp to [-1, 1]

    return math.acos(cos_theta)


def fresnel_reflection_coefficient(
    n1: complex,
    n2: complex,
    theta_i: float,
    theta_t: Optional[float],
    polarization: Polarization,
) -> complex:
    """
    Calculate Fresnel reflection coefficient.

    Args:
        n1: Complex refractive index of incident medium
        n2: Complex refractive index of transmitted medium
        theta_i: Incident angle in radians
        theta_t: Transmitted angle in radians (None for TIR)
        polarization: Ray polarization (PARALLEL or PERPENDICULAR)

    Returns:
        Complex Fresnel reflection coefficient
    """
    cos_theta_i = math.cos(theta_i)

    # For total internal reflection, return 1 (perfect reflection)
    if theta_t is None:
        return complex(1.0, 0.0)

    cos_theta_t = math.cos(theta_t)

    if polarization == Polarization.PERPENDICULAR:
        # Perpendicular polarization (s-polarized, E field out of plane)
        # Corresponds to f11 in the Rust code
        r_perp = (n2 * cos_theta_i - n1 * cos_theta_t) / (
            n1 * cos_theta_t + n2 * cos_theta_i
        )
        return r_perp
    else:
        # Parallel polarization (p-polarized, E field in plane)
        # Corresponds to f22 in the Rust code
        r_para = (n1 * cos_theta_i - n2 * cos_theta_t) / (
            n1 * cos_theta_i + n2 * cos_theta_t
        )
        return r_para


def fresnel_transmission_coefficient(
    n1: complex, n2: complex, theta_i: float, theta_t: float, polarization: Polarization
) -> complex:
    """
    Calculate Fresnel transmission coefficient.

    Args:
        n1: Complex refractive index of incident medium
        n2: Complex refractive index of transmitted medium
        theta_i: Incident angle in radians
        theta_t: Transmitted angle in radians
        polarization: Ray polarization (PARALLEL or PERPENDICULAR)

    Returns:
        Complex Fresnel transmission coefficient
    """
    cos_theta_i = math.cos(theta_i)
    cos_theta_t = math.cos(theta_t)

    if polarization == Polarization.PERPENDICULAR:
        # Perpendicular polarization (s-polarized)
        # Corresponds to f11 in the Rust code
        t_perp = (2.0 * n1 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i)
        return t_perp
    else:
        # Parallel polarization (p-polarized)
        # Corresponds to f22 in the Rust code
        t_para = (2.0 * n1 * cos_theta_i) / (n1 * cos_theta_i + n2 * cos_theta_t)
        return t_para
