#!/bin/bash

# 系统配置脚本
# 用于安装依赖和初始化系统

echo "=========================================="
echo "医疗 RAG 系统 - 配置脚本"
echo "=========================================="

# 检查 Python 版本
echo "检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p data
mkdir -p hf_cache
echo "✓ 目录创建完成"

# 检查数据文件
if [ ! -f "data/processed_data.json" ]; then
    echo "创建示例数据文件..."
    echo "[]" > data/processed_data.json
    echo "⚠️  注意: data/processed_data.json 是空文件，请先准备数据并写入该文件"
fi

# 安装依赖
echo ""
echo "安装 Python 依赖..."
pip install -r requirements.txt

# 检查安装结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 依赖安装完成！"
    echo "=========================================="
    echo ""
    echo "下一步："
    echo "1. 准备 data/processed_data.json（或使用 convert_graphrag_multi.py 生成）"
    echo "2. 然后运行: streamlit run app_improved.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ 依赖安装失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi


