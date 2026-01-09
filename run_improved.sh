#!/bin/bash

# 改进版 RAG 系统启动脚本

echo "=========================================="
echo "医疗 RAG 系统 - 改进版"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

echo "✓ Python 版本: $(python3 --version)"
echo ""

# 检查依赖
echo "检查依赖..."
python3 -c "import streamlit, pymilvus, sentence_transformers, transformers, torch, pydantic" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ 所有依赖已安装"
else
    echo "⚠️  部分依赖缺失，正在安装..."
    pip3 install -r requirements.txt
fi

echo ""

# 检查配置文件
if [ ! -f "config.toml" ]; then
    echo "⚠️  警告: config.toml 不存在，将使用默认配置"
fi

# 检查数据文件
if [ ! -f "data/processed_data.json" ] || [ ! -s "data/processed_data.json" ]; then
    echo "⚠️  提示: data/processed_data.json 为空或不存在"
    echo "   首次运行需要先索引数据（在应用侧边栏点击'索引数据'）"
    echo ""
fi

# 启动应用
echo "=========================================="
echo "启动 Streamlit 应用..."
echo "=========================================="
echo ""
echo "应用将在浏览器中自动打开"
echo "如果未自动打开，请访问: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止应用"
echo ""

streamlit run app_improved.py


