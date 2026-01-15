#!/bin/bash

# 指定 Python 解释器路径, 请根据自己的python环境进行修改
PYTHON_PATH="/home/san/miniconda3/bin/python"

# 确保输入了两个参数：输入文件和输出文件
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.fasta output.fasta"
    exit 1
fi

# 赋值输入和输出文件变量
IN_FASTA=$1
OUTPUT_PATH=$2

# 使用指定的 Python 解释器运行您的脚本, 并传入输入和输出文件； 请根据自己的python文件名进行修改`main.py`以及传入参数的方法
$PYTHON_PATH main.py $IN_FASTA --output $OUTPUT_PATH

# 结束脚本
echo "运行完成"