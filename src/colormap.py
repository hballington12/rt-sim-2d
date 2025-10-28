import numpy as np
from typing import Tuple


def intensity_to_heat_color(
    intensity: float, max_intensity: float = 1.0
) -> Tuple[int, int, int]:
    """
    Convert intensity value to heat-like color.
    Colormap: Black → Red → Orange → Yellow → White

    Args:
        intensity: Electric field intensity (E²/2)
        max_intensity: Maximum expected intensity for normalization

    Returns:
        RGB color tuple (0-255 for each channel)
    """
    # Normalize intensity to [0, 1]
    normalized = min(1.0, max(0.0, intensity / max_intensity))

    # Define color stops for heat map
    # Each stop is (position, R, G, B) where RGB are in [0, 1]
    stops = [
        (0.00, 0.0, 0.0, 0.0),  # Black
        (0.25, 0.5, 0.0, 0.0),  # Dark red
        (0.50, 1.0, 0.0, 0.0),  # Red
        (0.65, 1.0, 0.5, 0.0),  # Orange
        (0.80, 1.0, 1.0, 0.0),  # Yellow
        (1.00, 1.0, 1.0, 1.0),  # White
    ]

    # Find which two stops we're between
    for i in range(len(stops) - 1):
        pos1, r1, g1, b1 = stops[i]
        pos2, r2, g2, b2 = stops[i + 1]

        if pos1 <= normalized <= pos2:
            # Interpolate between the two colors
            if pos2 - pos1 > 0:
                t = (normalized - pos1) / (pos2 - pos1)
            else:
                t = 0

            r = r1 + t * (r2 - r1)
            g = g1 + t * (g2 - g1)
            b = b1 + t * (b2 - b1)

            # Convert to 0-255 range
            return (int(r * 255), int(g * 255), int(b * 255))

    # Fallback (should not reach here)
    return (255, 255, 255)


def calculate_intensity(electric_field: float) -> float:
    """
    Calculate light intensity from electric field amplitude.

    Args:
        electric_field: Electric field amplitude

    Returns:
        Intensity (E²/2)
    """
    return (electric_field**2) / 2.0


def draw_colorscale(
    screen, font, x: int, y: int, width: int, height: int, max_intensity: float = 0.5
):
    """
    Draw a colorscale legend on the screen.

    Args:
        screen: Pygame screen surface
        font: Pygame font for labels
        x, y: Top-left position
        width, height: Size of the colorscale bar
        max_intensity: Maximum intensity value for the scale
    """
    import pygame

    # Draw the gradient bar
    for i in range(height):
        # Calculate intensity for this position
        intensity = max_intensity * (1.0 - i / height)
        color = intensity_to_heat_color(intensity, max_intensity)
        pygame.draw.line(screen, color, (x, y + i), (x + width, y + i))

    # Draw border
    pygame.draw.rect(screen, (200, 200, 200), (x, y, width, height), 2)

    # Add labels
    label_color = (200, 200, 200)

    # Title
    title = font.render("Intensity", True, label_color)
    screen.blit(title, (x, y - 20))

    # Max value label
    max_label = font.render(f"{max_intensity:.2f}", True, label_color)
    screen.blit(max_label, (x + width + 5, y - 5))

    # Mid value label
    mid_label = font.render(f"{max_intensity / 2:.2f}", True, label_color)
    screen.blit(mid_label, (x + width + 5, y + height // 2 - 5))

    # Min value label
    min_label = font.render("0.00", True, label_color)
    screen.blit(min_label, (x + width + 5, y + height - 5))
