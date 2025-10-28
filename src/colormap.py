import numpy as np
from typing import Tuple
import math


def intensity_to_heat_color(
    intensity: float, max_intensity: float = 1.0, log_scale: bool = True
) -> Tuple[int, int, int]:
    """
    Convert intensity value to heat-like color with logarithmic scaling.
    Colormap: Black → Red → Orange → Yellow → White

    Args:
        intensity: Electric field intensity (E²/2)
        max_intensity: Maximum expected intensity for normalization
        log_scale: Use logarithmic scaling spanning 4 orders of magnitude

    Returns:
        RGB color tuple (0-255 for each channel)
    """
    if log_scale:
        # Logarithmic scaling: map intensities from max/10000 to max
        # This spans 4 orders of magnitude
        min_log_intensity = max_intensity / 10000.0  # 4 orders of magnitude down

        if intensity <= min_log_intensity:
            normalized = 0.0
        else:
            # Logarithmic normalization
            log_intensity = math.log10(intensity / min_log_intensity)
            log_max = 4.0  # log10(10000) = 4.0
            normalized = min(1.0, max(0.0, log_intensity / log_max))
    else:
        # Linear normalization (original)
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
    screen,
    font,
    x: int,
    y: int,
    width: int,
    height: int,
    max_intensity: float = 0.5,
    log_scale: bool = True,
):
    """
    Draw a colorscale legend on the screen with logarithmic scale.

    Args:
        screen: Pygame screen surface
        font: Pygame font for labels
        x, y: Top-left position
        width, height: Size of the colorscale bar
        max_intensity: Maximum intensity value for the scale
        log_scale: Use logarithmic scale spanning 4 orders of magnitude
    """
    import pygame

    if log_scale:
        # For logarithmic scale
        min_intensity = max_intensity / 10000.0  # 4 orders of magnitude

        # Draw the gradient bar
        for i in range(height):
            # Map position to logarithmic intensity
            t = 1.0 - i / height  # 0 to 1 from bottom to top
            if t <= 0:
                intensity = min_intensity
            else:
                # Logarithmic mapping
                log_range = 4.0  # 4 orders of magnitude
                log_val = t * log_range  # 0 to 4
                intensity = min_intensity * (10**log_val)

            color = intensity_to_heat_color(intensity, max_intensity, log_scale=True)
            pygame.draw.line(screen, color, (x, y + i), (x + width, y + i))
    else:
        # Linear scale (original)
        for i in range(height):
            intensity = max_intensity * (1.0 - i / height)
            color = intensity_to_heat_color(intensity, max_intensity, log_scale=False)
            pygame.draw.line(screen, color, (x, y + i), (x + width, y + i))

    # Draw border
    pygame.draw.rect(screen, (200, 200, 200), (x, y, width, height), 2)

    # Add labels
    label_color = (200, 200, 200)

    # Title
    title_text = "Log(I)" if log_scale else "Intensity"
    title = font.render(title_text, True, label_color)
    screen.blit(title, (x, y - 20))

    if log_scale:
        # Logarithmic labels
        # Max value label (10^0 relative to max)
        max_label = font.render(f"{max_intensity:.2e}", True, label_color)
        screen.blit(max_label, (x + width + 5, y - 5))

        # 10^-2 label
        mid_intensity = max_intensity / 100
        mid_label = font.render(f"{mid_intensity:.2e}", True, label_color)
        screen.blit(mid_label, (x + width + 5, y + height // 2 - 5))

        # Min value label (10^-4 relative to max)
        min_intensity = max_intensity / 10000
        min_label = font.render(f"{min_intensity:.2e}", True, label_color)
        screen.blit(min_label, (x + width + 5, y + height - 5))
    else:
        # Linear labels (original)
        max_label = font.render(f"{max_intensity:.2f}", True, label_color)
        screen.blit(max_label, (x + width + 5, y - 5))

        mid_label = font.render(f"{max_intensity / 2:.2f}", True, label_color)
        screen.blit(mid_label, (x + width + 5, y + height // 2 - 5))

        min_label = font.render("0.00", True, label_color)
        screen.blit(min_label, (x + width + 5, y + height - 5))
