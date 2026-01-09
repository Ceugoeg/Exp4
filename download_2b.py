# download_3b.py
from modelscope import snapshot_download
import os

print("🚀 正在下载 Qwen2.5-3B-Instruct (约 6GB)...")
try:
    path = snapshot_download("Qwen/Qwen2.5-3B-Instruct", cache_dir="./hf_cache")
    print(f"✅ 下载成功！路径:\n{os.path.abspath(path)}")
except Exception as e:
    print(f"❌ 失败: {e}")