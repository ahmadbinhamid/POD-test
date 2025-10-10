# -*- mode: python ; coding: utf-8 -*-
import site
from PyInstaller.utils.hooks import collect_submodules

# Resolve current venv site-packages for pathex
pathex = [p for p in site.getsitepackages() if p.endswith('site-packages')]

hiddenimports = []
for pkg in ['timm', 'ultralytics']:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

hiddenimports += [
    'fastapi', 'uvicorn', 'pydantic',
    'transformers', 'diffusers',
    'httpx', 'openai',
    'torch', 'torchvision',
    'langsmith', 'pypdfium2', 'python_dotenv',
    'doctr', 'doctr.io', 'doctr.models', 'doctr.models._utils',
    'PIL', 'jinja2', 'cv2', 'requests'
]

a = Analysis(
    ['orchestrator.py'],   # entry script
    pathex=pathex,
    binaries=[],
    datas=[
        ('Models/*', 'Models'),          # include ML models
        ('requirements.txt', '.'),
        ('app.py', '.'),                 # FastAPI app
        ('lmdeploy_app.py', '.'),        # LMDeploy
        ('pixtral.py', '.'),             # Pixtral
        ('utils.py', '.'),               # utils
        ('prompts.py', '.'),             # prompts
        ('models.py', '.'),              # DB/ML models
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pod_ocr_suite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='pod_ocr_suite',
)
