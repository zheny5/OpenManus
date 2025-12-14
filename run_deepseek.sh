#!/bin/bash
# 使用 DeepSeek LLM 运行 OpenManus

cd "$(dirname "$0")"

# 激活虚拟环境
source .venv/bin/activate

# 运行主程序
python main.py
