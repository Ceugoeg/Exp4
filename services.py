
import time
import torch
from typing import List, Dict, Any, Tuple
from pymilvus import Collection
from database import DocumentDatabase

class VectorStoreService:
    def __init__(self, client, config):
        self.client_alias = client 
        self.collection_name = config.milvus.collection_name
        # 容错处理
        if hasattr(config.milvus, 'search_params'):
            self.search_params = config.milvus.search_params
        else:
            self.search_params = {"nprobe": 10}
        self.top_k = config.search.top_k

    def search(self, query_vector: List[float], top_k: int = None) -> Tuple[List[int], List[float]]:
        if top_k is None: top_k = self.top_k
        try:
            collection = Collection(self.collection_name)
            collection.load() 
            search_param = {"metric_type": "L2", "params": self.search_params}
            results = collection.search(
                data=[query_vector], anns_field="embedding", param=search_param,
                limit=top_k, output_fields=["content_preview"] 
            )
            if not results or len(results[0]) == 0: return [], []
            doc_ids = [hit.id for hit in results[0]]
            distances = [hit.distance for hit in results[0]]
            return doc_ids, distances
        except Exception as e:
            print(f"向量搜索失败: {e}")
            return [], []

class EmbeddingService:
    def __init__(self, model):
        self.model = model
    def encode_single(self, text: str) -> List[float]:
        if hasattr(self.model, 'encode'):
            embedding = self.model.encode(text, normalize_embeddings=True)
            if hasattr(embedding, 'tolist'): return embedding.tolist()
            return embedding
        return []

class GenerationService:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def _post_process(self, text: str) -> str:
        """
        【核心修复】后处理函数
        强行切断模型的复读行为
        """
        # 1. 找到第一个【答案】的内容
        # Base模型有时候会把Prompt里的【答案】也重复一遍，我们只要有效内容
        
        # 定义停止符：如果模型试图再次生成这些标签，说明它在复读
        stop_markers = ["【答案】", "【回答】", "Question:", "User:", "Context:"]
        
        count = 0
        cut_index = len(text)
        
        if text.count("【答案】") > 1:
            # 找到第二个【答案】的位置
            first = text.find("【答案】")
            second = text.find("【答案】", first + 1)
            if second != -1:
                text = text[:second]
                
        if text.count("【参考关键词】") > 1:
            first = text.find("【参考关键词】")
            second = text.find("【参考关键词】", first + 1)
            if second != -1:
                text = text[:second]

        return text.strip()

    def generate(self, query: str, context_docs: List[Dict]) -> str:
        if not context_docs: return "无法找到相关文档来回答您的问题。"
        
        # 1. 组装上下文
        assembled = []
        for doc in context_docs:
            content = doc.get('content_preview') 
            if not content:
                t = doc.get('title') or ''
                a = doc.get('abstract') or ''
                content = f"Title: {t}\nContent: {a}"
            assembled.append(content)
        context = "\n\n---\n\n".join(assembled)
        
        # 2. Prompt (微调：统一使用 【答案】 标签，减少歧义)
        prompt = f"""你是专业的医学助手。基于以下资料回答问题。

资料：
{context}

问题：{query}

请严格按以下格式输出（不要重复）：
【答案】
(在此处填写入回答)
【要点】
(在此处列出要点)
【参考关键词】
"""
        
        try:
            # 3. 参数配置
            gen_cfg = getattr(self.config, 'generation', None)
            max_new = getattr(gen_cfg, 'max_new_tokens', 300) # 缩短最大长度，防止废话
            temp = 0.3 # 极低温度
            rep_pen = 1.1 # 高重复惩罚

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=temp,
                    top_p=0.9,
                    repetition_penalty=rep_pen,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # 4. 调用强力截断
            final_response = self._post_process(response)
            
            if not final_response: return "生成结果为空。"
            return final_response
            
        except Exception as e:
            print(f"Error: {e}")
            return "生成出错。"

class DocumentService:
    def __init__(self, db: DocumentDatabase, vector_service: VectorStoreService, embedding_service: EmbeddingService):
        self.db = db
        self.vector_service = vector_service
        self.embedding_service = embedding_service
    def get_documents_by_ids(self, doc_ids: List[int]) -> List[Dict]:
        return self.db.get_documents_by_ids(doc_ids)

class RAGService:
    def __init__(self, vector_service, embedding_service, generation_service, document_service, config):
        self.vector_service = vector_service
        self.embedding_service = embedding_service
        self.generation_service = generation_service
        self.document_service = document_service
        self.config = config

    def query(self, query_text: str) -> Dict:
        start_time = time.time()
        # 1. 编码
        query_embedding = self.embedding_service.encode_single(query_text)
        embed_time = time.time()
        # 2. 搜索
        doc_ids, distances = self.vector_service.search(query_embedding, top_k=self.config.search.top_k)
        search_time = time.time()
        # 3. 取文档
        retrieved_docs = self.document_service.get_documents_by_ids(doc_ids)
        retrieve_time = time.time()
        # 4. 生成
        answer = self.generation_service.generate(query_text, retrieved_docs)
        generation_time = time.time()
        
        performance = {
            "embedding_time": embed_time - start_time,
            "search_time": search_time - embed_time,
            "retrieve_time": retrieve_time - search_time,
            "generation_time": generation_time - retrieve_time,
            "total_time": generation_time - start_time
        }
        self.document_service.db.add_query_history(query_text, answer, doc_ids, performance["total_time"])
        
        return {
            "answer": answer, "retrieved_docs": retrieved_docs,
            "doc_ids": doc_ids, "distances": distances, "performance": performance
        }