# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Ray Tracing App
Builds a standalone executable with all dependencies bundled
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect pygame-gui theme files and data
datas = []
datas += collect_data_files('pygame_gui')

# Hidden imports that PyInstaller might miss
hiddenimports = []
hiddenimports += collect_submodules('pygame')
hiddenimports += collect_submodules('pygame_gui')
hiddenimports += ['numpy', 'numpy.core._methods', 'numpy.lib.format']

a = Analysis(
    ['src/integrated_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PIL', 'tkinter'],  # Exclude unused large packages
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RayTracingApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False to hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you create one
)

# For macOS, create an app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='RayTracingApp.app',
        icon=None,  # Add icon path here if you create one
        bundle_identifier='com.raytracing.app',
        info_plist={
            'CFBundleName': 'Ray Tracing App',
            'CFBundleDisplayName': 'Ray Tracing Simulator',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': 'True',
        },
    )
