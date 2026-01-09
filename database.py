"""
文档元数据数据库管理模块
使用 SQLite 存储文档元数据，替代内存中的 id_to_doc_map
"""
import sqlite3
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os


class DocumentDatabase:
    """文档元数据数据库管理类"""
    
    def __init__(self, db_path: str = "./document_metadata.db"):
        """初始化数据库连接"""
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # 返回字典格式
        
        cursor = self.conn.cursor()
        
        # 创建文档元数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT,
                content TEXT NOT NULL,
                source_file TEXT,
                chunk_index INTEGER,
                content_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引状态表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_status (
                doc_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                indexed_at TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES document_metadata(id)
            )
        """)
        
        # 创建查询历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                answer_text TEXT,
                retrieved_doc_ids TEXT,
                response_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引以提高查询性能
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON document_metadata(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_index_status ON index_status(status)")
        
        self.conn.commit()
    
    def _calculate_hash(self, content: str) -> str:
        """计算内容哈希值"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def add_document(self, doc_id: int, title: str, abstract: str, 
                     content: str, source_file: Optional[str] = None, 
                     chunk_index: Optional[int] = None) -> bool:
        """添加或更新文档"""
        try:
            content_hash = self._calculate_hash(content)
            cursor = self.conn.cursor()
            
            # 使用 INSERT OR REPLACE 处理重复
            cursor.execute("""
                INSERT OR REPLACE INTO document_metadata 
                (id, title, abstract, content, source_file, chunk_index, content_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, title, abstract, content, source_file, chunk_index, content_hash, datetime.now()))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        """根据 ID 获取文档"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM document_metadata WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_documents_by_ids(self, doc_ids: List[int]) -> List[Dict]:
        """批量获取文档"""
        if not doc_ids:
            return []
        
        placeholders = ','.join('?' * len(doc_ids))
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM document_metadata WHERE id IN ({placeholders})", doc_ids)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_all_document_ids(self) -> List[int]:
        """获取所有文档 ID（用于懒加载）"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM document_metadata")
        return [row[0] for row in cursor.fetchall()]
    
    def delete_document(self, doc_id: int) -> bool:
        """删除文档"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM document_metadata WHERE id = ?", (doc_id,))
            cursor.execute("DELETE FROM index_status WHERE doc_id = ?", (doc_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def update_document(self, doc_id: int, title: Optional[str] = None,
                        abstract: Optional[str] = None, content: Optional[str] = None) -> bool:
        """更新文档内容"""
        try:
            cursor = self.conn.cursor()
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            if abstract is not None:
                updates.append("abstract = ?")
                params.append(abstract)
            if content is not None:
                updates.append("content = ?")
                updates.append("content_hash = ?")
                params.append(content)
                params.append(self._calculate_hash(content))
            
            if not updates:
                return False
            
            updates.append("updated_at = ?")
            params.append(datetime.now())
            params.append(doc_id)
            
            query = f"UPDATE document_metadata SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def mark_indexed(self, doc_id: int, status: str = "completed"):
        """标记文档索引状态"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO index_status (doc_id, status, indexed_at)
            VALUES (?, ?, ?)
        """, (doc_id, status, datetime.now()))
        self.conn.commit()
    
    def get_indexed_doc_ids(self) -> List[int]:
        """获取已索引的文档 ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT doc_id FROM index_status WHERE status = 'completed'")
        return [row[0] for row in cursor.fetchall()]
    
    def get_unindexed_doc_ids(self) -> List[int]:
        """获取未索引的文档 ID"""
        indexed_ids = set(self.get_indexed_doc_ids())
        all_ids = set(self.get_all_document_ids())
        return list(all_ids - indexed_ids)
    
    def add_query_history(self, query_text: str, answer_text: Optional[str] = None,
                         retrieved_doc_ids: Optional[List[int]] = None,
                         response_time: Optional[float] = None) -> int:
        """添加查询历史"""
        cursor = self.conn.cursor()
        doc_ids_str = ','.join(map(str, retrieved_doc_ids)) if retrieved_doc_ids else None
        cursor.execute("""
            INSERT INTO query_history (query_text, answer_text, retrieved_doc_ids, response_time)
            VALUES (?, ?, ?, ?)
        """, (query_text, answer_text, doc_ids_str, response_time))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_query_history(self, limit: int = 50) -> List[Dict]:
        """获取查询历史"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM query_history 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_document_count(self) -> int:
        """获取文档总数"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_metadata")
        return cursor.fetchone()[0]
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


