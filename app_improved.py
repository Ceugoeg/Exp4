"""
改进后的主应用
(精简版：专为 0.5B/1.5B/3B 等轻量级模型设计，移除量化逻辑)
"""
import streamlit as st
import os
import time
# 不需要 torch 和 bitsandbytes 了
from transformers import AutoTokenizer, AutoModelForCausalLM 

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'
os.environ['NO_PROXY'] = "127.0.0.1,localhost"
os.environ['no_proxy'] = "127.0.0.1,localhost"

# 导入模块
from config_manager import get_config, reload_config
from database import DocumentDatabase
from models_with_timeout import (
    load_embedding_model,
    _load_embedding_model_cached,
    _load_generation_model_cached,
)
from milvus_service import get_milvus_client, setup_milvus_collection, index_data_incremental
from services import VectorStoreService, EmbeddingService, GenerationService, DocumentService, RAGService
from security import InputValidator, RateLimiter, ResourceLimiter
from data_utils import load_data

# 页面配置
st.set_page_config(layout="wide", page_title="医疗 RAG 系统")

# 初始化配置
config = get_config()

# 初始化组件
@st.cache_resource
def init_components_cached(_config):
    return _init_components_internal(_config, None)

def _init_components_internal(_config, progress_placeholder):
    if progress_placeholder:
        progress_placeholder.progress(0.05, text="正在初始化数据库...")
    
    db = DocumentDatabase(_config.data.database_path)
    
    if progress_placeholder:
        progress_placeholder.progress(0.1, text="正在连接 Milvus...")
    
    milvus_client = get_milvus_client(_config.milvus.data_path)
    if not milvus_client:
        return None, None, None, None, None, None
    
    if not setup_milvus_collection(milvus_client, _config):
        return None, None, None, None, None, None
    
    def update_progress(value, message):
        if progress_placeholder:
            try:
                progress_placeholder.progress(value, text=message)
            except Exception:
                pass
    
    # 加载 Embedding
    if progress_placeholder:
        progress_placeholder.progress(0.2, text=f"正在加载嵌入模型...")
    embedding_model = load_embedding_model(_config.model.embedding_model_name, update_progress, timeout=600)
    
    if not embedding_model:
        return None, None, None, None, None, None
    
    # =========================================================================
    # 🚨【精简加载逻辑】🚨
    # 只针对 0.5B/1.5B/3B 模型，使用标准模式加载
    # =========================================================================
    model_name_path = _config.model.generation_model_name
    
    if progress_placeholder:
        progress_placeholder.progress(0.5, text=f"正在加载生成模型: {model_name_path}...")

    try:
        # 1. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, 
            trust_remote_code=True
        )
        
        # 2. 加载模型 (标准模式 - 速度最快)
        generation_model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto"  # 让 transformers 自动选择精度(通常是fp16或fp32)
        )
            
    except Exception as e:
        if progress_placeholder:
            progress_placeholder.progress(1.0, text=f"❌ 模型加载失败: {str(e)}")
        print(f"Model Load Error: {e}")
        return None, None, None, None, None, None
    # =========================================================================
    
    if progress_placeholder:
        progress_placeholder.progress(0.8, text="正在初始化服务层...")
    
    vector_service = VectorStoreService(milvus_client, _config)
    embedding_service = EmbeddingService(embedding_model)
    generation_service = GenerationService(generation_model, tokenizer, _config)
    document_service = DocumentService(db, vector_service, embedding_service)
    rag_service = RAGService(vector_service, embedding_service, generation_service, document_service, _config)
    
    if progress_placeholder:
        progress_placeholder.progress(1.0, text="✅ 初始化完成！")
        time.sleep(0.5)
        progress_placeholder.empty()
    
    return db, milvus_client, embedding_model, generation_model, tokenizer, rag_service

# 检查是否已初始化
if 'components_initialized' not in st.session_state or st.sidebar.button("🔄 重新初始化"):
    if 'components_initialized' in st.session_state:
        st.session_state['components_initialized'] = False
        try:
            _load_embedding_model_cached.clear()
        except Exception:
            pass
        config = reload_config()
    
    with st.status("🔄 系统初始化中...", expanded=True) as status:
        progress_bar = st.progress(0, text="开始初始化...")
        db, milvus_client, embedding_model, generation_model, tokenizer, rag_service = _init_components_internal(config, progress_bar)
        
        if all([db, milvus_client, embedding_model, generation_model, tokenizer, rag_service]):
            st.session_state['components_initialized'] = True
            st.session_state['db'] = db
            st.session_state['milvus_client'] = milvus_client
            st.session_state['embedding_model'] = embedding_model
            st.session_state['generation_model'] = generation_model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['rag_service'] = rag_service
        else:
            st.error("初始化失败，请重试")
else:
    db = st.session_state.get('db')
    milvus_client = st.session_state.get('milvus_client')
    embedding_model = st.session_state.get('embedding_model')
    generation_model = st.session_state.get('generation_model')
    tokenizer = st.session_state.get('tokenizer')
    rag_service = st.session_state.get('rag_service')

input_validator = InputValidator(config.security.max_query_length)
rate_limiter = RateLimiter(config.security.rate_limit_per_minute, 60)
resource_limiter = ResourceLimiter(config.security.max_concurrent_queries, config.security.query_timeout)

st.title("📄 医疗 RAG 系统 (0.5B 极速版)")
st.markdown(f"当前模型: `{config.model.generation_model_name}`")

if not all([db, milvus_client, embedding_model, generation_model, tokenizer, rag_service]):
    st.error("❌ 系统初始化失败")
    if st.button("🔄 重试"):
        st.session_state['components_initialized'] = False
        st.rerun()
    st.stop()
else:
    st.success("✅ 系统已就绪！")

st.sidebar.header("数据管理")
if st.sidebar.button("索引数据"):
    pubmed_data = load_data(config.data.data_file)
    if pubmed_data:
        with st.spinner("正在索引数据..."):
            index_data_incremental(milvus_client, pubmed_data, embedding_model, db, config)
    else:
        st.sidebar.warning("无法加载数据文件")

doc_count = db.get_document_count()
indexed_count = len(db.get_indexed_doc_ids())
st.sidebar.info(f"文档总数: {doc_count}\n已索引: {indexed_count}")

st.divider()

with st.expander("📜 查询历史", expanded=False):
    history = db.get_query_history(limit=10)
    if history:
        for item in history:
            st.text(f"查询: {item['query_text'][:50]}...")
            st.text(f"耗时: {item['response_time']:.2f}s")
            st.divider()
    else:
        st.info("暂无查询历史")

query = st.text_input("请输入问题:", key="query_input", placeholder="例如：什么是糖尿病？")

if st.button("获取答案", key="submit_button", type="primary"):
    if not query:
        st.warning("请输入问题")
    else:
        is_valid, error_msg = input_validator.validate(query)
        if not is_valid:
            st.error(f"验证失败: {error_msg}")
            st.stop()
        
        user_id = "default"
        if not rate_limiter.is_allowed(user_id):
            st.error("请求过于频繁")
            st.stop()
        
        if not resource_limiter.acquire():
            st.error("系统繁忙")
            st.stop()
        
        try:
            start_time = time.time()
            result = rag_service.query(query)
            
            st.subheader("生成的答案:")
            st.write(result["answer"])
            
            st.subheader("参考文档:")
            for i, doc in enumerate(result["retrieved_docs"]):
                dist = result['distances'][i] if result['distances'] else 0
                with st.expander(f"文档 {i+1} (相似度: {dist:.4f})"):
                    st.write(doc['abstract'])
            
            with st.expander("📊 性能"):
                perf = result["performance"]
                st.metric("总耗时", f"{perf['total_time']:.2f}s")
        
        except Exception as e:
            st.error(f"出错: {e}")
        finally:
            resource_limiter.release()

st.sidebar.divider()
st.sidebar.text(f"Model: {config.model.generation_model_name}")