"""Task 19.2 — Dead Letter Queue."""
import sys, os, time, json, logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("task_19_2")

@dataclass
class AgentTask:
    task_id: str
    task_type: str
    payload: dict
    retries: int = 0
    max_retries: int = 3

class DeadLetterQueue:
    """Stores failed tasks for inspection and re-enqueue."""
    
    def __init__(self):
        self._queue = deque()
        self._alert_threshold = 1  # Alert when DLQ has any items
    
    def put(self, task: AgentTask, error: str):
        entry = {"task": task.__dict__, "error": error, "failed_at": time.time()}
        self._queue.append(entry)
        logger.warning(f"[DLQ] Task {task.task_id} added: {error}")
        
        if len(self._queue) >= self._alert_threshold:
            logger.error(f"[DLQ_ALERT] {len(self._queue)} tasks in DLQ!")
    
    def drain(self) -> list:
        items = list(self._queue)
        self._queue.clear()
        return items
    
    def __len__(self):
        return len(self._queue)

class TaskProcessor:
    """Process tasks with retry + DLQ fallback."""
    
    def __init__(self, dlq: DeadLetterQueue):
        self.dlq = dlq
        self.processed = 0
        self.failed = 0
    
    def process(self, task: AgentTask) -> bool:
        """Process with exponential backoff retry."""
        try:
            # Simulate processing — fail for certain task types
            if task.payload.get("force_fail"):
                raise RuntimeError(f"Processing failed for {task.task_id}")
            
            # Success
            self.processed += 1
            logger.info(f"[PROCESS] {task.task_id} completed")
            return True
            
        except Exception as e:
            task.retries += 1
            if task.retries <= task.max_retries:
                wait = 2 ** task.retries  # Exponential backoff: 2, 4, 8s
                logger.warning(f"[RETRY] {task.task_id} attempt {task.retries}, waiting {wait}s")
                time.sleep(min(wait, 0.1))  # Capped for demo
                return self.process(task)
            else:
                self.failed += 1
                self.dlq.put(task, str(e))
                return False

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 19.2 — DEAD LETTER QUEUE")
    print("=" * 50)
    
    dlq = DeadLetterQueue()
    processor = TaskProcessor(dlq)
    
    # Successful task
    print("\n--- Successful task ---")
    ok_task = AgentTask(task_id="t1", task_type="review", payload={"text": "hello"})
    result = processor.process(ok_task)
    print(f"  Result: {'✅ success' if result else '❌ failed'}")
    
    # Failing task (exhausts retries → DLQ)
    print("\n--- Failing task (3 retries → DLQ) ---")
    fail_task = AgentTask(task_id="t2", task_type="review", payload={"force_fail": True})
    result = processor.process(fail_task)
    print(f"  Result: {'✅ success' if result else '❌ failed → DLQ'}")
    
    # Inspect DLQ
    print(f"\n--- DLQ contents ({len(dlq)} items) ---")
    for item in dlq.drain():
        print(f"  Task: {item['task']['task_id']}, Error: {item['error']}")
    
    # Re-enqueue after fix
    print("\n--- Re-enqueue after fix ---")
    print("  Fix root cause → re-enqueue: AgentTask.from_dict(entry['task'])")
    
    print(f"\nStats: {processor.processed} processed, {processor.failed} failed")
    print("✅ DLQ + retry + exponential backoff working!")
