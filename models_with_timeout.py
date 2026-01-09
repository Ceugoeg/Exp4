"""
带超时和进度检测的模型加载模块
"""
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import threading
import os
from pathlib import Path
from streamlit.runtime.scriptrunner import add_script_run_ctx

def check_download_progress(cache_dir, check_interval=60, timeout=600):
    """
    检查下载进度
    返回: (is_active, last_activity_time)
    """
    if not os.path.exists(cache_dir):
        return False, 0
    
    last_mtime = 0
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                mtime = os.path.getmtime(filepath)
                last_mtime = max(last_mtime, mtime)
            except:
                pass
    
    current_time = time.time()
    time_since_activity = current_time - last_mtime
    
    # 如果10分钟内没有活动，认为已停止
    is_active = time_since_activity < timeout
    
    return is_active, time_since_activity

@st.cache_resource
def _load_embedding_model_cached(model_name):
    """缓存的模型加载函数"""
    return SentenceTransformer(model_name)

def load_embedding_model(model_name, progress_callback=None, timeout=600):
    """Loads the sentence transformer model with timeout."""
    start_time = time.time()
    cache_dir = os.path.join(os.getenv('HF_HOME', './hf_cache'), 'models--' + model_name.replace('/', '--'))
    download_stopped = threading.Event()
    
    def check_progress():
        while not download_stopped.is_set():
            time.sleep(5)  # 每5秒检查一次，更频繁更新
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                if progress_callback:
                    progress_callback(1.0, f"⏱️ 超时（{timeout}秒）")
                download_stopped.set()
                return
            
            is_active, time_since = check_download_progress(cache_dir, timeout=300)
            if not is_active and elapsed > 120:  # 至少等待2分钟后才判断
                if progress_callback:
                    progress_callback(1.0, f"⚠️ 下载已停止（{time_since:.0f}秒无活动），请检查网络或重试")
                download_stopped.set()
                return
            
            if progress_callback and not download_stopped.is_set():
                # 检查缓存目录大小，估算下载进度
                cache_size = 0
                if os.path.exists(cache_dir):
                    for root, dirs, files in os.walk(cache_dir):
                        for f in files:
                            try:
                                cache_size += os.path.getsize(os.path.join(root, f))
                            except:
                                pass
                cache_size_mb = cache_size / (1024 * 1024)
                
                progress_value = min(0.2 + 0.15 * (elapsed / timeout), 0.9)
                progress_callback(progress_value, 
                               f"正在下载/加载模型...（已用时 {int(elapsed)}秒，已下载 {cache_size_mb:.1f}MB）")
    
    try:
        if progress_callback:
            progress_callback(0.1, f"开始加载嵌入模型: {model_name}")
        
        # 启动进度检查线程
        progress_thread = threading.Thread(target=check_progress, daemon=True)
        add_script_run_ctx(progress_thread)  # 让线程继承当前 context
        progress_thread.start()
        
        # 尝试加载模型（使用缓存）
        model = _load_embedding_model_cached(model_name)
        download_stopped.set()
        
        if progress_callback:
            progress_callback(0.3, "✅ 嵌入模型加载完成")
        
        return model
    except Exception as e:
        download_stopped.set()
        if progress_callback:
            progress_callback(1.0, f"❌ 加载失败: {e}")
        return None

@st.cache_resource
def _load_generation_model_cached(model_name):
    """缓存的生成模型加载函数"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_generation_model(model_name, progress_callback=None, timeout=1200):
    """Loads the Hugging Face generative model with timeout."""
    start_time = time.time()
    cache_dir = os.path.join(os.getenv('HF_HOME', './hf_cache'), 'models--' + model_name.replace('/', '--'))
    download_stopped = threading.Event()
    
    def check_progress():
        while not download_stopped.is_set():
            time.sleep(5)  # 每5秒检查一次，更频繁更新
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                if progress_callback:
                    progress_callback(1.0, f"⏱️ 超时（{timeout}秒）")
                download_stopped.set()
                return
            
            is_active, time_since = check_download_progress(cache_dir, timeout=300)
            if not is_active and elapsed > 120:
                if progress_callback:
                    progress_callback(1.0, f"⚠️ 下载已停止（{time_since:.0f}秒无活动），请检查网络或重试")
                download_stopped.set()
                return
            
            if progress_callback and not download_stopped.is_set():
                # 检查缓存目录大小，估算下载进度
                cache_size = 0
                if os.path.exists(cache_dir):
                    for root, dirs, files in os.walk(cache_dir):
                        for f in files:
                            try:
                                cache_size += os.path.getsize(os.path.join(root, f))
                            except:
                                pass
                cache_size_mb = cache_size / (1024 * 1024)
                
                progress_value = min(0.5 + 0.15 * (elapsed / timeout), 0.9)
                progress_callback(progress_value, 
                               f"正在下载/加载模型...（已用时 {int(elapsed)}秒，已下载 {cache_size_mb:.1f}MB）")
    
    try:
        if progress_callback:
            progress_callback(0.4, f"开始加载生成模型: {model_name}")
        
        # 启动进度检查线程
        progress_thread = threading.Thread(target=check_progress, daemon=True)
        add_script_run_ctx(progress_thread)  # 让线程继承当前 context
        progress_thread.start()
        
        if progress_callback:
            progress_callback(0.45, "正在加载 tokenizer...")
        
        # 尝试加载模型（使用缓存）
        model, tokenizer = _load_generation_model_cached(model_name)
        download_stopped.set()
        
        if progress_callback:
            progress_callback(0.7, "✅ 生成模型加载完成")
        
        return model, tokenizer
    except Exception as e:
        download_stopped.set()
        if progress_callback:
            progress_callback(1.0, f"❌ 加载失败: {e}")
        return None, None

