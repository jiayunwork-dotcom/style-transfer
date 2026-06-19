import threading
import time
import queue
from datetime import datetime
from app.core.config import MAX_CONCURRENT_BATCH


class BatchTaskQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._active_count = 0
        self._lock = threading.Lock()
        self._cancel_flags = {}
        self._semaphore = threading.Semaphore(MAX_CONCURRENT_BATCH)
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def submit(self, task_id, process_func):
        with self._lock:
            self._cancel_flags[task_id] = False
        self._queue.put((task_id, process_func))
        return task_id

    def cancel(self, task_id):
        with self._lock:
            if task_id in self._cancel_flags:
                self._cancel_flags[task_id] = True
                return True
        return False

    def is_cancelled(self, task_id):
        with self._lock:
            return self._cancel_flags.get(task_id, False)

    def _worker(self):
        while True:
            try:
                task_id, process_func = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            with self._lock:
                if self._cancel_flags.get(task_id, False):
                    continue

            self._semaphore.acquire()
            try:
                if not self._cancel_flags.get(task_id, False):
                    process_func(task_id, self.is_cancelled)
            except Exception:
                pass
            finally:
                self._semaphore.release()

    def get_queue_size(self):
        return self._queue.qsize()


batch_queue = BatchTaskQueue()
