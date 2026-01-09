"""
Milvus 服务模块 (最终修复版)
功能：
1. 适配 app_improved.py 的调用
2. 自动处理 512维 -> 384维 的切换 (自动重建库)
3. 解决 'str' object has no attribute 'insert' 报错
"""
import streamlit as st
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
import time

def get_milvus_client(data_path: str):
    """
    建立 Milvus 连接 (底层接口)
    """
    try:
        # 1. 尝试断开旧连接，防止 Streamlit 重载时报错
        try:
            connections.disconnect("default")
        except Exception:
            pass

        # 2. 解析 IP (去掉 http://)
        host = "127.0.0.1"
        port = "19530"
        
        # 简单的清洗逻辑
        if "://" in data_path:
            clean_path = data_path.split("://")[-1]
            if ":" in clean_path:
                host = clean_path.split(":")[0]
        elif ":" in data_path:
            host = data_path.split(":")[0]

        print(f"🔌 [Milvus] 正在连接 {host}:{port}...")
        connections.connect(alias="default", host=host, port=port, timeout=5)
        
        return "default" # 返回连接别名，不是对象
    except Exception as e:
        st.error(f"Milvus 连接失败: {e}")
        return None

def setup_milvus_collection(client, config):
    """
    初始化集合 (自动检测维度冲突)
    """
    if not client:
        return False
    
    collection_name = config.milvus.collection_name
    target_dim = config.model.embedding_dim # 从 config 读取 (384)
    
    try:
        # 1. 检查是否存在
        if utility.has_collection(collection_name):
            col = Collection(collection_name)
            col.load()
            
            # 2. 深度检查：维度是否匹配？
            existing_dim = -1
            for field in col.schema.fields:
                if field.name == "embedding":
                    existing_dim = field.params.get('dim')
                    break
            
            # 如果 Config 是 384，但库里是 512，必须删库重建！
            if existing_dim != -1 and existing_dim != target_dim:
                st.warning(f"⚠️ 维度冲突检测！数据库: {existing_dim}维 vs 配置: {target_dim}维")
                st.warning(f"🔄 正在自动删除旧集合 '{collection_name}' 并重建...")
                utility.drop_collection(collection_name)
                # 删完后，程序会继续往下走去创建新的
            else:
                print(f"📦 [Milvus] 集合已就绪: {collection_name}")
                return True

        # 3. 创建新集合
        st.info(f"🔨 创建新集合: {collection_name} (Dim={target_dim})")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=target_dim),
            # 【重要】增加长度以容纳完整文章
            FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=10000)
        ]
        schema = CollectionSchema(fields, description="Medical RAG Data")
        col = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": config.milvus.index_metric_type,
            "index_type": config.milvus.index_type,
            "params": config.milvus.index_params
        }
        col.create_index(field_name="embedding", index_params=index_params)
        col.load()
        st.success("✅ 集合初始化成功")
        return True

    except Exception as e:
        st.error(f"集合初始化失败: {e}")
        return False

def index_data_incremental(client, data, embedding_model, db, config):
    """
    增量索引 (UI 按钮点击后执行的逻辑)
    """
    if not client or not embedding_model:
        st.error("服务未就绪")
        return False
    
    collection_name = config.milvus.collection_name
    
    # 1. 获取已索引 ID
    indexed_ids = set(db.get_indexed_doc_ids())
    
    docs_to_process = []
    texts_to_embed = []
    
    # 进度条
    progress_bar = st.progress(0, text="正在分析数据...")
    
    # 2. 筛选
    limit = config.data.max_articles_to_index
    for i, doc in enumerate(data):
        if i >= limit: break
        
        doc_id = doc.get('chunk_index', i)
        
        # 这里的 abstract 其实是全文 (由 convert 脚本保证)
        content = doc.get('abstract', '')
        title = doc.get('title', 'No Title')
        
        # 无论是否已索引，先存入 SQLite 确保元数据完整
        # (因为之前可能被清空过)
        db.add_document(doc_id, title, content, content, doc.get('source_file'), i)
        
        # 检查是否需要向量化 (Milvus)
        if doc_id in indexed_ids:
            continue
            
        full_text = f"Title: {title}\nContent: {content}"
        
        docs_to_process.append({
            "id": doc_id,
            "preview": full_text[:9999] # 截断防止超长报错
        })
        texts_to_embed.append(full_text)

    # 3. 如果没新数据
    if not docs_to_process:
        progress_bar.progress(1.0, text="✅ 数据已是最新。")
        time.sleep(1)
        progress_bar.empty()
        return True
    
    st.info(f"发现 {len(docs_to_process)} 条数据待索引...")
    
    try:
        # 4. 批量处理
        batch_size = 50
        col = Collection(collection_name) # 获取集合对象
        
        total = len(docs_to_process)
        for i in range(0, total, batch_size):
            batch_docs = docs_to_process[i : i + batch_size]
            batch_texts = texts_to_embed[i : i + batch_size]
            
            # 进度
            progress = i / total
            progress_bar.progress(progress, text=f"正在向量化 {i}/{total}...")
            
            # Embedding
            embeddings = embedding_model.encode(batch_texts, normalize_embeddings=True)
            
            # Insert
            ids_col = [d['id'] for d in batch_docs]
            embeds_col = embeddings.tolist()
            previews_col = [d['preview'] for d in batch_docs]
            
            col.insert([ids_col, embeds_col, previews_col])
            
            # Update SQLite
            for d in batch_docs:
                db.mark_indexed(d['id'])

        # 5. 收尾：强制落盘
        progress_bar.progress(0.9, text="正在保存数据 (Flush)...")
        col.flush()
        
        progress_bar.progress(1.0, text="✅ 索引完成！")
        st.success(f"成功索引 {total} 条文档。")
        time.sleep(2)
        progress_bar.empty()
        return True
        
    except Exception as e:
        st.error(f"索引出错: {e}")
        return False