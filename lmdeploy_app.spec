# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
import os
 
# Collect all submodules
hiddenimports = (
    collect_submodules('lmdeploy')
    + collect_submodules('PIL')
    + collect_submodules('torch')
    + collect_submodules('multiprocessing')
    + [
        'lmdeploy',
        'lmdeploy.api',
        'lmdeploy.serve',
        'lmdeploy.serve.openai',
        'lmdeploy.serve.openai.api_server',
        'lmdeploy.serve.vl_async_engine',
        'lmdeploy.serve.async_engine',
        'lmdeploy.pytorch',
        'lmdeploy.pytorch.engine',
        'lmdeploy.pytorch.engine.engine',
        'lmdeploy.pytorch.engine.mp_engine',
        'lmdeploy.pytorch.engine.mp_engine.zmq_engine',
        'lmdeploy.PytorchEngineConfig',
        'PIL',
        'PIL.Image',
        'torch',
        'torch.cuda',
        'torch.nn',
        'torch.distributed',
        'multiprocessing.spawn',
        'multiprocessing.managers',
        'multiprocessing.process',
        'multiprocessing.context',
        'zmq',
        'uvicorn',
        'fastapi',
        'pydantic',
    ]
)
 
# Collect data files
datas = (
    [('./lmdeploy_app.py', '.')]
    + collect_data_files('lmdeploy', include_py_files=True)
    + collect_data_files('torch', include_py_files=False)
    + collect_data_files('uvicorn')
    + collect_data_files('fastapi')
)
 
a = Analysis(
    ['lmdeploy_app.py'],
    pathex=[os.getcwd()],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['./hooks'] if os.path.exists('./hooks') else [],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'IPython',
        'notebook',
        'pytest',
    ],
    noarchive=False,
    optimize=0,
)
 
pyz = PYZ(a.pure)
 
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
 
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='my_app',
)