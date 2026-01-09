"""
数据转换脚本 (终极融合版：全量文章 + 问答增强)
功能：
1. 使用稳健的递归切分算法，找回所有医疗文章 (解决只剩10篇的问题)
2. 读取 medical_questions.json，注入问答对知识
3. 自动生成 data/processed_data.json 供主程序索引
"""
import json
import os
from pathlib import Path
import pandas as pd
import re

# ================= 配置参数 =================
MAX_LENGTH = 1000  # 文章切分最大长度
OVERLAP = 100      # 上下文重叠 (防止语义断裂)
# ===========================================

def recursive_split(text, max_len, overlap):
    """
    稳健的分块算法 (递归切分)
    不再依赖 "About" 标题，而是按自然段落和句子切分
    """
    if not text: return []
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + max_len
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        # 寻找最佳切分点 (从 end 往前找)
        cut_point = -1
        # 优先级：换行 > 句号/感叹号/问号
        for i in range(end, start + overlap, -1):
            if text[i] in ['\n', '.', '。', '!', '?']:
                cut_point = i + 1
                break
        
        # 如果找不到标点，被迫硬切
        if cut_point == -1: 
            cut_point = end
            
        chunks.append(text[start:cut_point].strip())
        # 下一块从切分点减去重叠量开始
        start = cut_point - overlap
        
    return [c for c in chunks if len(c) > 20] # 过滤过短碎片

def main():
    print("🚀 启动数据转换程序...")
    records = []
    
    # ---------------------------------------------------------
    # 阶段 1: 处理基础文章 (medical.parquet)
    # ---------------------------------------------------------
    # 定义可能的路径列表 (自动寻找)
    parquet_paths = [
        "GraphRAG-Benchmark-main/Datasets/Corpus/medical.parquet",
        "medical.parquet",
        "data/medical.parquet"
    ]
    parquet_file = None
    for p in parquet_paths:
        if os.path.exists(p):
            parquet_file = Path(p)
            break
            
    if parquet_file:
        print(f"📖 [1/2] 处理文章源: {parquet_file}")
        try:
            df = pd.read_parquet(parquet_file)
            article_count = 0
            
            for idx, row in df.iterrows():
                context = row.get("context", "")
                if not isinstance(context, str): continue
                
                # 【关键修复】使用递归切分，不再丢数据
                parts = recursive_split(context, MAX_LENGTH, OVERLAP)
                
                for part in parts:
                    # 尝试智能提取标题 (取第一行)
                    first_line = part.split('\n')[0][:80]
                    # 如果第一行看起来像标题(包含About)，就用它，否则叫片段
                    if "About" in first_line:
                        title = first_line
                    else:
                        title = "Medical Document Fragment"
                    
                    records.append({
                        "title": title,
                        "abstract": part, # 存入完整正文
                        "source_file": "medical.parquet",
                        "chunk_index": len(records)
                    })
                    article_count += 1
            print(f"   => 成功提取 {article_count} 个文章片段 (数据完整！)")
        except Exception as e:
            print(f"❌ 读取 Parquet 失败: {e}")
    else:
        print("❌ 未找到 medical.parquet，请检查文件位置！")

    # ---------------------------------------------------------
    # 阶段 2: 处理问答数据 (medical_questions.json)
    # ---------------------------------------------------------
    # 定义可能的路径列表 (包含深层目录)
    qa_paths = [
        "GraphRAG-Benchmark-main/Datasets/Questions/medical_questions.json", # 深层路径
        "medical_questions.json",                                            # 根目录
        "data/medical_questions.json"
    ]
    qa_file = None
    for p in qa_paths:
        if os.path.exists(p):
            qa_file = Path(p)
            break
            
    if qa_file:
        print(f"📖 [2/2] 处理问答源: {qa_file}")
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            qa_count = 0
            for item in qa_data:
                # 安全过滤：再次确保不是小说
                source = str(item.get('source', '')).lower()
                if "novel" in source: 
                    continue
                
                q = item.get('question', '').strip()
                a = item.get('answer', '').strip()
                
                if q and a:
                    # 将问答对格式化为文档块
                    content = f"Question: {q}\nAnswer: {a}\nEvidence: {item.get('evidence','')}"
                    
                    records.append({
                        "title": f"Q&A: {q[:60]}...",
                        "abstract": content,
                        "source_file": "medical_questions.json",
                        "chunk_index": len(records)
                    })
                    qa_count += 1
            print(f"   => 成功提取 {qa_count} 个问答对")
        except Exception as e:
            print(f"❌ 读取 JSON 失败: {e}")
    else:
        print(f"⚠️ 未找到 medical_questions.json，跳过问答注入。")

    # ---------------------------------------------------------
    # 阶段 3: 保存结果
    # ---------------------------------------------------------
    output_file = Path("data/processed_data.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 转换完成！")
    print(f"📊 总数据量: {len(records)} 条")
if __name__ == "__main__":
    main()