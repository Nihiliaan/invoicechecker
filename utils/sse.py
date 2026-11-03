"""
Server-Sent Events (SSE) data formatting utilities.
"""

import json
from typing import Optional, Any


def make_step_event(step: str, status: Optional[bool] = None, data: Any = None) -> str:
    """构建标准纯英文三要素 SSE 事件数据帧：
    step: 纯英文步骤名
    status: 布尔值（仅在步骤完成或失败时提供）
    data: 纯英文键业务载荷
    """
    ev = {"step": step}
    if status is not None:
        ev["status"] = bool(status)
    if data is not None:
        ev["data"] = data
    status_str = "SUCCESS" if status is True else ("FAIL" if status is False else "RUNNING")
    print(f"[SSE][{step}][{status_str}]", flush=True)
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"


def make_event(data: dict) -> str:
    """向后兼容原始字典格式事件"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def log_event(msg: str, level: str = "info", step: str = "execute", status: str = "running") -> str:
    """记录控制台日志并生成对应的步骤推进事件"""
    print(f"[{level.upper()}][{step}][{status}] {msg}", flush=True)
    return make_step_event(step=step)
