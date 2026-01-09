"""
安全性模块：输入验证、频率限制、资源限制
"""
import time
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from functools import wraps
import threading


class InputValidator:
    """输入验证器"""
    
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length
        # 危险字符模式
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript 协议
            r'on\w+\s*=',  # 事件处理器
        ]
    
    def validate(self, text: str) -> Tuple[bool, Optional[str]]:
        """验证输入文本"""
        if not text or not isinstance(text, str):
            return False, "输入不能为空"
        
        if len(text) > self.max_length:
            return False, f"输入长度超过限制（最大 {self.max_length} 字符）"
        
        # 检查危险模式
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "输入包含不安全内容"
        
        return True, None
    
    def sanitize(self, text: str) -> str:
        """清理输入文本"""
        # 移除控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # 移除多余空白
        text = ' '.join(text.split())
        return text.strip()


class RateLimiter:
    """频率限制器"""
    
    def __init__(self, max_requests: int = 30, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        
        with self.lock:
            # 清理过期请求
            user_requests = self.requests[identifier]
            while user_requests and now - user_requests[0] > self.time_window:
                user_requests.popleft()
            
            # 检查是否超过限制
            if len(user_requests) >= self.max_requests:
                return False
            
            # 记录新请求
            user_requests.append(now)
            return True
    
    def get_remaining(self, identifier: str) -> int:
        """获取剩余请求数"""
        now = time.time()
        
        with self.lock:
            user_requests = self.requests[identifier]
            # 清理过期请求
            while user_requests and now - user_requests[0] > self.time_window:
                user_requests.popleft()
            
            return max(0, self.max_requests - len(user_requests))


class ResourceLimiter:
    """资源限制器"""
    
    def __init__(self, max_concurrent: int = 5, timeout: float = 60.0):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.active_requests = 0
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """获取资源"""
        with self.lock:
            if self.active_requests >= self.max_concurrent:
                return False
            self.active_requests += 1
            return True
    
    def release(self):
        """释放资源"""
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)


def rate_limit(max_requests: int = 30, time_window: int = 60):
    """频率限制装饰器"""
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 使用第一个参数或 kwargs 中的 identifier
            identifier = kwargs.get('user_id', 'default')
            if args:
                identifier = getattr(args[0], 'user_id', identifier)
            
            if not limiter.is_allowed(identifier):
                raise Exception(f"请求过于频繁，请稍后再试。剩余请求数: {limiter.get_remaining(identifier)}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def with_timeout(timeout: float):
    """超时装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"操作超时（{timeout}秒）")
            
            # 设置超时
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # 取消超时
            
            return result
        return wrapper
    return decorator

