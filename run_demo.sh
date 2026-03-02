#!/bin/bash
# GAPartNet机器人操作示例运行脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 激活conda环境
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate gapartnet

# 设置环境变量（必须在conda activate之后）
export PYTHONPATH=/home/plote/isaacgym/python:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 进入manipulation目录
cd "$SCRIPT_DIR"

# 检查是否有--headless参数
if [[ "$*" == *"--headless"* ]]; then
    HEADLESS_FLAG="--headless"
else
    HEADLESS_FLAG=""
fi

# 检查模式参数
if [[ "$*" == *"--mode"* ]]; then
    # 如果用户指定了模式，使用用户指定的模式
    python run.py "$@"
else
    # 默认运行打开抽屉示例（带GUI）
    echo "运行模式: 打开抽屉 (GUI模式)"
    echo "如需headless模式，请添加 --headless 参数"
    echo "如需其他模式，请使用 --mode 参数，例如: --mode run_arti_free_control"
    echo ""
    python run.py --mode run_arti_open $HEADLESS_FLAG
fi

