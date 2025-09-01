
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self, total_steps: int = 3, broadcast_status_function: callable = None):
        self.overall_progress = 0
        self.current_step = ""
        self.step_progress = 0
        self.total_steps = total_steps
        self.current_step_index = 0
        self.broadcast_status_function = broadcast_status_function
        self.message = ""

    def start_step(self, step_name: str, step_index: int = None):
        self.current_step = step_name
        if step_index is not None:
            self.current_step_index = step_index
        else:
            self.current_step_index += 1
        self.step_progress = 0.01
        self._publish()

    def set_message(self, message: str):
        self.message = message
        self._publish()

    def update_step_progress(self, progress: float):
        self.step_progress = progress
        self.overall_progress = (self.step_progress / self.total_steps) + (self.current_step_index / self.total_steps)
        self.overall_progress = min(100, int(self.overall_progress*100))
        self._publish()

    def update_step_progress_of_total(self, progress: int, total: int):
        self.step_progress = (progress / total)
        self.update_step_progress(self.step_progress)

    def _publish(self):
        if self.broadcast_status_function:
            status = "running"
            message = self.message
            if self.current_step not in ["queued", "finished", "failed"]:
                message += f" ({int(self.step_progress*100)}%)"
            # logger.info(f"ProgressTracker: {status}, {self.overall_progress}%, {message}")
            self.broadcast_status_function(
                status=status,
                progress=self.overall_progress,
                message=f"{self.current_step.capitalize()}: {message}"
            )
        # self.redis.set(f"detection:{self.report_id}:status", self.current_step)
        # self.redis.set(f"detection:{self.report_id}:progress", self.overall_progress)
        # self.redis.set(f"detection:{self.report_id}:message", self.message)