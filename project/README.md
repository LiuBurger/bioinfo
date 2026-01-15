# 期末项目提交说明

## CHANGE LOG
 > 最终测试会测试多组数据，对每个数据文件限制时间为 10 分钟。



本项目在期末成绩中占比很高。请严格遵循以下指南来准备和提交您的期末项目。

## 提交指南

### 选择测评用户
- 每个队伍需指定一名用户作为提交用户。
- 我们会进入该用户的 `bioinfo/project/submission` 目录用于评估您团队的最终成绩。

### 预提交阶段
- 提交截止日期前两周为预提交阶段，用于测试代码运行和输出格式。
- ❗️❗️❗️❗️❗️ **由于期末项目截止后，需要及时测试提交成绩，所以不再接受任何提交，请大家重视预提交阶段！！** ❗️❗️❗️❗️❗️

### 预提交时间以及正式提交时间

- 预提交开放时间为十八周周末，即 2024 年 1 月 7 日 00:00:00。
- 正式提交截止时间为二十周周末晚24点， 即 2024 年 1 月 21 日 24:00:00。

### 我们如何测试(🔥提交后助教运行，大家不需要执行🔥)
- 在 `~/bioinfo/project/submission` 目录下，执行命令 `bash run.sh $infasta $output_path`。
- `$infasta` 和 `$output_path` 为输入的FASTA 文件路径以及输出文件路径

## 项目要求

### 1. 脚本文件
- `~/bioinfo/project/submission` 目录下必须包含名为 `run.sh` 的脚本文件。
- 此脚本将作为项目运行的主入口。

### 2. Python 环境
- 在 `run.sh` 中调用 Python 环境时，需指定具体的 Python 解释器位置。
- 例如：使用 `/home/san/miniconda3/bin/python`。

### 3. 运行时间限制
- 最终测试会测试多组数据，对每个数据文件限制时间为 10 分钟。
- 如果超过十分钟，程序会被强制终止，所以建议随着程序的运行，输出结果到输出文件中

## 联系方式
- 如遇任何问题或疑问，可以发邮件到 sdu_cs_bioinfo@163.com 联系课程助教

祝大家项目顺利！

---

### `run.sh` 文件示例

```bash
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

# 使用指定的 Python 解释器运行您的脚本, 并传入输入和输出文件； 请根据自己的python文件名进行修改`your_script.py`
$PYTHON_PATH your_script.py $IN_FASTA $OUTPUT_PATH

# 结束脚本
echo "运行完成"
```

请确保根据您的具体项目需求对上述 `run.sh` 脚本进行适当的调整。
