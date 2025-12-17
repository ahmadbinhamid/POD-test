# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# Collect all submodules for required packages and also include extra packages
hiddenimports = (
    collect_submodules('lmdeploy')
    + collect_submodules('timm')
    + collect_submodules('fastapi')
    + [
        'fastapi',
        'uvicorn', 
        'transformers',
        'diffusers',
        'timm',
        'lmdeploy',
        'ultralytics',
        'doctr',             # corrected from 'python-doctr[torch]'
        'python-dotenv',
        'pypdfium2',
        'langsmith'
    ]
)

# Analysis configuration for PyInstaller
a = Analysis(
    # Main application entry point
    ['app.py'],
    
    # Project root and the actual venv site-packages (keep if you need it)
    pathex=["/workspace/POD_OCR", "/workspace/POD_OCR/venv/lib/python3.12/site-packages"],
    
    # No additional binaries required
    binaries=[],
    
    # Data files and dependencies to include (Models directory is in project)
    datas=[
        # Model files (from your project Models/ directory)
        ('./Models/page_orientation.pt', '.'),
        ('./Models/stamp_existence.pt', '.'),
        ('./Models/bol_regions_best.pt', '.'),
        ('./Models/doc_classify_best.pt', '.'),
        ('./Models/receipts_regions_best.pt', '.'),
        
        # Project files
        ('./requirements.txt', '.'),
        
        # Python source files
        ('./app.py', '.'),
        ('./gr_app.py', '.'),
        ('./internvl2.py', '.'),
        ('./main.py', '.'),
        ('./models.py', '.'),
        ('./pixtral.py', '.'),
        ('./prompts.py', '.'),
        ('./utils.py', '.'),
    ]
    # add package data discovered at runtime (do not repeat hard-coded site-packages paths)
    + collect_data_files('lmdeploy')
    + collect_data_files('timm'),
    
    # Use the hiddenimports collected above (don't overwrite)
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
