"""
异步任务管理器（内存态）

用于追踪视频检测等长任务的进度与状态。
单进程内有效；生产多worker场景需换 Redis 等共享存储。
"""
import threading
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    PENDING = "pending"      # 已创建，待处理
    RUNNING = "running"      # 处理中
    COMPLETED = "completed"  # 成功完成
    FAILED = "failed"        # 失败


@dataclass
class Task:
    id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0          # 0~100
    message: str = ""
    total_frames: int = 0
    processed_frames: int = 0
    result: Optional[dict] = None  # 完成后的结果数据
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class TaskManager:
    """线程安全的任务管理器"""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def create(self) -> Task:
        task_id = uuid.uuid4().hex
        task = Task(id=task_id)
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            for k, v in kwargs.items():
                if hasattr(task, k):
                    setattr(task, k, v)

    def remove(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)


task_manager = TaskManager()
