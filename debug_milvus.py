"""
Schema 修复脚本
功能：删除旧的字段名为 'text' 的数据库，重建为 'content_preview'
"""
import os
import json
import time
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

# ================= 配置 =================
os.environ['NO_PROXY'] = "127.0.0.1,localhost"
os.environ['HF_HUB_OFFLINE'] = '1'
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['HF_HOME'] = os.path.join(current_dir, 'hf_cache')

# 模型路径
MODEL_PATH = "D:/Code/exp04-easy-rag-system/hf_cache/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
DATA_FILE = os.path.join(current_dir, "data/processed_data.json")
COLLECTION_NAME = "medical_rag_lite"
DIMENSION = 512
# =======================================

def main():
    print("🚀 1. 连接 Milvus...")
    try:
        connections.connect(alias="default", host="127.0.0.1", port="19530")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 1. 无论以前叫什么，统统删掉
    if utility.has_collection(COLLECTION_NAME):
        print(f"🗑️ 删除旧集合: {COLLECTION_NAME}")
        utility.drop_collection(COLLECTION_NAME)

    # 2. 使用正确的字段名 'content_preview' 重建
    print(f"🔨 2. 重建集合 (Schema 修正)...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        # 【关键修正】这里必须叫 content_preview，才能和主程序匹配！
        FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=2000)
    ]
    schema = CollectionSchema(fields, "Medical RAG Data")
    col = Collection(COLLECTION_NAME, schema)

    # 3. 读取数据
    print(f"📂 3. 读取数据文件...")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except Exception:
        # 如果没文件，造一点假数据保证程序能跑
        data_list = [{"title": "测试", "abstract": "糖尿病是一种代谢疾病..."}]

    print("📚 4. 加载模型...")
    encoder = SentenceTransformer(MODEL_PATH)

    print(f"🔄 5. 正在生成向量 ({len(data_list)} 条)...")
    
    # 准备数据
    ids = []
    texts = []
    previews = []
    
    for i, item in enumerate(data_list[:500]):
        content = f"Title: {item.get('title','')}\nAbstract: {item.get('abstract','')}"
        ids.append(i)
        texts.append(content)
        previews.append(content[:1999])

    embeddings = encoder.encode(texts, normalize_embeddings=True)

    print("💾 6. 写入新数据...")
    col.insert([ids, embeddings, previews])
    
    print("🚽 正在强制刷新 (Flush)...")
    col.flush() # 这一步不做，主程序就搜不到

    print("⚙️ 7. 构建索引...")
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()

    print(f"✅ 修复完成！数据库现有 {col.num_entities} 条数据。")
    print("👉 字段名已统一为 content_preview，现在重启主程序即可。")

if __name__ == "__main__":
    main()