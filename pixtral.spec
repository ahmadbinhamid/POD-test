# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# Collect all submodules for required packages
hiddenimports = (
    collect_submodules('fastapi')
    + collect_submodules('uvicorn')
    + collect_submodules('openai')
    + collect_submodules('PIL')
    + collect_submodules('requests')
    + collect_submodules('langsmith')
    + [
        'fastapi',
        'uvicorn',
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.logging',
        'openai',
        'PIL',
        'PIL.Image',
        'PIL._tkinter_finder',
        'requests',
        'langsmith',
        'asyncio',
        'json',
        'base64',
        'io',
        're',
        'datetime',
    ]
)

# Analysis configuration for PyInstaller
a = Analysis(
    # Main application entry point
    ['pixtral.py'],
    
    # Project root
    pathex=["/workspace/POD_OCR"],
    
    # No additional binaries required
    binaries=[],
    
    # Data files and dependencies to include
    datas=[
        # Project files
        ('./models.py', '.'),
        ('./pixtral.py', '.'),
    ]
    # Add package data
    + collect_data_files('openai')
    + collect_data_files('langsmith'),
    
    # Use the hiddenimports collected above
    hiddenimports=hiddenimports,
    
    # Custom hooks directory
    hookspath=['./hooks'],
    
    # Additional configuration
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Create the Python ZIP archive
pyz = PYZ(a.pure)

# Configure the executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='my_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Configure the collection of files
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='my_app',
)