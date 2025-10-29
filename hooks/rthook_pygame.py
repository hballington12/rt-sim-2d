"""
Runtime hook for pygame to ensure constants are available
This runs before the main application and ensures pygame constants are properly loaded
"""

import pygame
import pygame.constants

# Ensure all pygame constants are loaded into the pygame namespace
# This fixes the issue where pygame-gui tries to access pygame.DIRECTION_LTR
if hasattr(pygame.constants, "DIRECTION_LTR"):
    pygame.DIRECTION_LTR = pygame.constants.DIRECTION_LTR
if hasattr(pygame.constants, "DIRECTION_RTL"):
    pygame.DIRECTION_RTL = pygame.constants.DIRECTION_RTL
if hasattr(pygame.constants, "DIRECTION_TTB"):
    pygame.DIRECTION_TTB = pygame.constants.DIRECTION_TTB
if hasattr(pygame.constants, "DIRECTION_BTT"):
    pygame.DIRECTION_BTT = pygame.constants.DIRECTION_BTT

# Also ensure other commonly used constants are available
for attr_name in dir(pygame.constants):
    if not attr_name.startswith("_"):
        attr = getattr(pygame.constants, attr_name)
        if not hasattr(pygame, attr_name):
            setattr(pygame, attr_name, attr)
