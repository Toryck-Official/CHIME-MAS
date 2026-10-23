#!/usr/bin/env bash
# 在 G-Memory 目录下运行其脚本；DeepSeek 密钥从 /home/claude/chime/.env 读取，不出现在命令行。
set -a
. /home/claude/chime/.env
set +a
export QWEN_EMB_FILE=model_int8.onnx
export OPENAI_API_BASE="$DEEPSEEK_BASE_URL" OPENAI_API_KEY="$DEEPSEEK_API_KEY" PYTHONPATH=.:tasks
cd /home/claude/refs/gmemory
exec .venv/bin/python "$@"
