module.exports = {
  apps: [
    {
      name: "pod-ocr",
      script: "bash",
      args: "run_all.sh --tp 1 --venv /workspace/POD_OCR/venv",
      cwd: "/workspace/POD_OCR",
      autorestart: true,
      watch: false,
      max_memory_restart: "16G",
    },
  ],
};
