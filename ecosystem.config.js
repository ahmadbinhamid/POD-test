module.exports = {
  apps: [
    {
      name: 'pod-ocr-lmdeploy',
      // use /bin/bash to activate venv then exec python so environment matches interactive run
      script: '/bin/bash',
      args: '-lc "source /workspace/POD_OCR/venv/bin/activate && exec python lmdeploy_app.py"',
      cwd: '/workspace/POD_OCR',
      interpreter: 'none',
      out_file: '/workspace/POD_OCR/logs_lmdeploy.txt',
      error_file: '/workspace/POD_OCR/logs_lmdeploy.txt',
      combine_logs: true,
      env: {
        VIRTUAL_ENV: '/workspace/POD_OCR/venv',
        PATH: '/workspace/POD_OCR/venv/bin:' + process.env.PATH
      }
    },
    {
      name: 'pod-ocr-pixtral',
      script: '/bin/bash',
      args: '-lc "source /workspace/POD_OCR/venv/bin/activate && exec python pixtral.py"',
      cwd: '/workspace/POD_OCR',
      interpreter: 'none',
      out_file: '/workspace/POD_OCR/logs_pixtral.txt',
      error_file: '/workspace/POD_OCR/logs_pixtral.txt',
      combine_logs: true,
      env: {
        VIRTUAL_ENV: '/workspace/POD_OCR/venv',
        PATH: '/workspace/POD_OCR/venv/bin:' + process.env.PATH
      }
    },
    {
      name: 'pod-ocr-api',
      script: '/bin/bash',
      args: '-lc "source /workspace/POD_OCR/venv/bin/activate && exec python app.py"',
      cwd: '/workspace/POD_OCR',
      interpreter: 'none',
      out_file: '/workspace/POD_OCR/logs_api.txt',
      error_file: '/workspace/POD_OCR/logs_api.txt',
      combine_logs: true,
      env: {
        VIRTUAL_ENV: '/workspace/POD_OCR/venv',
        PATH: '/workspace/POD_OCR/venv/bin:' + process.env.PATH
      }
    }
  ]
};
