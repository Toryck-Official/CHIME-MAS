#!/usr/bin/env bash
# 在 G-Memory 目录下运行其脚本；DeepSeek 密钥从 /home/claude/chime/.env 读取，不出现在命令行。
# DeepSeek 高峰时段（北京时间周一至周五 9:00–12:00、14:00–18:00）价格翻倍：默认拒绝启动，ALLOW_PEAK=1 才放行
bj_dow=$(TZ=Asia/Shanghai date +%u); bj_hm=$(TZ=Asia/Shanghai date +%H%M)
if [ "$bj_dow" -le 5 ] && { { [ "$bj_hm" -ge 0900 ] && [ "$bj_hm" -lt 1200 ]; } || { [ "$bj_hm" -ge 1400 ] && [ "$bj_hm" -lt 1800 ]; }; } \
   && [ "${ALLOW_PEAK:-0}" != "1" ]; then
  echo "PEAK_HOURS: 北京时间 $(TZ=Asia/Shanghai date '+%a %H:%M') 为高峰时段，未启动。设置 ALLOW_PEAK=1 可强制运行。" >&2
  exit 3
fi
set -a
. /home/claude/chime/.env
set +a
export QWEN_EMB_FILE=model_int8.onnx
export OPENAI_API_BASE="$DEEPSEEK_BASE_URL" OPENAI_API_KEY="$DEEPSEEK_API_KEY" PYTHONPATH=.:tasks
cd /home/claude/refs/gmemory
exec .venv/bin/python "$@"
