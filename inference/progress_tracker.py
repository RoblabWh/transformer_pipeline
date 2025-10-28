
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class Step:
#     def __init__(self, name: str, index: int, description: str = "", time_estimate: int = 0, model: str = None):
#         self.name = name
#         self.index = index
#         self.description = description
#         self.time_estimate = time_estimate  # in seconds
#         self.progress = 0.0  # 0.0 to 1.0
#         self.model = model
#         self.inference_time_per_image = None  # in seconds

class ProgressTracker:
    def __init__(self, total_steps: int = 3, broadcast_status_function: callable = None):
        self.overall_progress = 0
        self.current_step = ""
        self.step_progress = 0
        self.total_steps = total_steps
        self.current_step_index = 0
        self.broadcast_status_function = broadcast_status_function
        self.message = ""
        self.time_estimate = 180  # in seconds
        self.seconds_per_inference = 0.5  # in seconds
        self.inference_time_by_model = {"RoblabWhGe/rescuedet-deformable-detr": 0.4,
                                        "RoblabWhGe/rescuedet-yolos-small": 0.3,}
        self.time_estimate_per_step = []
        self.splitting_efficiency_factor = 1.0  # factor to reduce inference time when splitting is used
        
    def get_inference_time_for_model(self, model_name: str) -> float:
        return self.inference_time_by_model.get(model_name, self.seconds_per_inference)


    def start_step(self, step_name: str, step_index: int = None):
        self.current_step = step_name
        if step_index is not None:
            self.current_step_index = step_index
        else:
            self.current_step_index += 1
        self.step_progress = 0.01
        self.time_estimate = sum(self.time_estimate_per_step[self.current_step_index:])
        self._publish()

    def set_message(self, message: str):
        self.message = message
        self._publish()

    def update_step_progress(self, progress: float):
        self.step_progress = progress
        self.overall_progress = (self.step_progress / self.total_steps) + (self.current_step_index / self.total_steps)
        self.overall_progress = min(100, int(self.overall_progress*100))
        if self.current_step_index == 0:
            self.time_estimate = sum(self.time_estimate_per_step) - progress*self.time_estimate_per_step[0]
        self._publish()

    def update_time_estimate_of_current_step(self, time_remaining_seconds: int):
        self.time_estimate_per_step[self.current_step_index] = time_remaining_seconds
        self.time_estimate = sum(self.time_estimate_per_step[self.current_step_index:])
        self._publish()

    def update_step_progress_of_total(self, progress: int, total: int):
        self.step_progress = (progress / total)
        self.update_step_progress(self.step_progress)

    def _publish(self):
        if self.broadcast_status_function:
            status = "running"
            message = self.message
            if self.current_step not in ["queued", "finished", "failed"]:
                time_left = int(self.time_estimate)
                time_left_str = f"{int(time_left/60)}:{int(time_left%60):02d} min"
                message += f" ({int(self.step_progress*100)}%) | Time left: {time_left_str}"
            # logger.info(f"ProgressTracker: {status}, {self.overall_progress}%, {message}")
            self.broadcast_status_function(
                status=status,
                progress=self.overall_progress,
                message=f"{self.current_step.capitalize()}: {message}"
            )
        # self.redis.set(f"detection:{self.report_id}:status", self.current_step)
        # self.redis.set(f"detection:{self.report_id}:progress", self.overall_progress)
        # self.redis.set(f"detection:{self.report_id}:message", self.message)

    def estimate_total_time(self, images_count: int, splitting: bool, max_splitting_steps: int, models: list[str]) -> int:
        # step 1 model loading
        time_s1 = 20*len(models)  # 20 seconds per model load from the internet

        # splitting time
        time_s2 = images_count * 0.1
        if splitting:
            # splits =  sum([4**i for i in range(0, max_splitting_steps)][1:])+1
            # time_s2 += images_count * 0.05 * splits  # 0.05 seconds per image split
            # images_count = images_count * (4**max_splitting_steps)
            if max_splitting_steps == 1:
                time_s2 += images_count * 0.05   # 0.05 seconds per image split
                images_count = images_count * 4
            elif max_splitting_steps > 1:
                time_s2 += images_count * 0.2   # 0.2 seconds per image split
                images_count = images_count * 16

        # step 2 model inference
        time_s3 = self.estimate_time_inference(images_count, models, splitting)

        # step 3 merging & postprocessing
        time_s4 = self.estimate_time_postprocessing(images_count)

        total_time = time_s1 + time_s2 + time_s3 + time_s4
        self.time_estimate_per_step = [time_s1, time_s2, time_s3, time_s4]
        self.time_estimate = total_time
        logger.info(f"Estimated total processing time: {total_time} seconds for an estimated total of {images_count} images with steps: {self.time_estimate_per_step}")
        return total_time
    
    def estimate_time_inference(self, images_count: int, models: list[str], splitting: bool) -> int:
        # step 2 model inference
        time_s3 = 0
        for model in models:
            time_per_image = self.get_inference_time_for_model(model)
            time_s3 += images_count * time_per_image
        if splitting:
            self.splitting_efficiency_factor = 0.6
            time_s3 *= self.splitting_efficiency_factor  # assume some efficiency gain due to smaller images

        logger.info(f"Estimated inference time: {time_s3} seconds for {images_count} images and models: {models}")
        return time_s3
    
    def estimate_time_postprocessing(self, images_count: int) -> int:
        # step 3 merging & postprocessing
        time_s4 = images_count * 0.005 + 5 # 0.01 seconds per image for postprocessing + 4 seconds fixed overhead
        logger.info(f"Estimated postprocessing time: {time_s4} seconds for {images_count} images")
        return time_s4
    