"""
配置管理模块
使用 TOML 配置文件和 pydantic 进行配置验证
"""
import os
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field

# 兼容不同 Python 版本的 TOML 库
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python 3.8-3.10
    except ImportError:
        raise ImportError("需要安装 tomli 库: pip install tomli")


class ModelConfig(BaseModel):
    """模型配置"""
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", description="嵌入模型名称")
    generation_model_name: str = Field(default="gpt2", description="生成模型名称")
    embedding_dim: int = Field(default=384, description="嵌入向量维度")


class MilvusConfig(BaseModel):
    """Milvus 配置"""
    data_path: str = Field(default="http://127.0.0.1:19530", description="Milvus 服务地址或 URI")
    collection_name: str = Field(default="medical_rag_lite", description="Collection 名称")
    index_metric_type: str = Field(default="L2", description="索引距离度量类型")
    index_type: str = Field(default="IVF_FLAT", description="索引类型")
    index_params: dict = Field(default={"nlist": 128}, description="索引参数")
    search_params: dict = Field(default={"nprobe": 16}, description="搜索参数")


class DataConfig(BaseModel):
    """数据配置"""
    data_file: str = Field(default="./data/processed_data.json", description="数据文件路径")
    max_articles_to_index: int = Field(default=500, description="最大索引文章数")
    database_path: str = Field(default="./document_metadata.db", description="文档元数据库路径")


class GenerationConfig(BaseModel):
    """生成参数配置"""
    max_new_tokens: int = Field(default=512, description="最大生成 token 数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 采样参数")
    repetition_penalty: float = Field(default=1.1, ge=1.0, description="重复惩罚")


class SearchConfig(BaseModel):
    """搜索配置"""
    top_k: int = Field(default=3, ge=1, description="返回 Top-K 结果")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度阈值")


class SecurityConfig(BaseModel):
    """安全性配置"""
    max_query_length: int = Field(default=1000, description="最大查询长度")
    max_concurrent_queries: int = Field(default=5, description="最大并发查询数")
    query_timeout: float = Field(default=60.0, description="查询超时时间（秒）")
    rate_limit_per_minute: int = Field(default=30, description="每分钟查询频率限制")


class AppConfig(BaseModel):
    """应用配置"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    @classmethod
    def from_toml(cls, config_path: Optional[str] = None) -> "AppConfig":
        """从 TOML 文件加载配置"""
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "./config.toml")
        
        config_path = Path(config_path)
        
        # 如果配置文件不存在，使用默认配置
        if not config_path.exists():
            return cls()
        
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
        
        # 支持从环境变量覆盖配置
        config_data = cls._override_with_env(config_data)
        
        return cls(**config_data)
    
    @staticmethod
    def _override_with_env(config_data: dict) -> dict:
        """使用环境变量覆盖配置"""
        # 模型配置
        if os.getenv("EMBEDDING_MODEL_NAME"):
            config_data.setdefault("model", {})["embedding_model_name"] = os.getenv("EMBEDDING_MODEL_NAME")
        if os.getenv("GENERATION_MODEL_NAME"):
            config_data.setdefault("model", {})["generation_model_name"] = os.getenv("GENERATION_MODEL_NAME")
        
        # 数据配置
        if os.getenv("DATA_FILE"):
            config_data.setdefault("data", {})["data_file"] = os.getenv("DATA_FILE")
        if os.getenv("MAX_ARTICLES_TO_INDEX"):
            config_data.setdefault("data", {})["max_articles_to_index"] = int(os.getenv("MAX_ARTICLES_TO_INDEX"))
        
        return config_data
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return self.dict()


# 全局配置实例
_config: Optional[AppConfig] = None


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """获取配置实例（单例模式）"""
    global _config
    if _config is None:
        _config = AppConfig.from_toml(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> AppConfig:
    """重新加载配置"""
    global _config
    _config = AppConfig.from_toml(config_path)
    return _config

