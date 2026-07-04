import logging
import time
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, Field

import cv2
import numpy as np
import torch
# Patch torch.load to default weights_only to False to support YOLOv8 model loading in PyTorch 2.6+
import functools
original_torch_load = torch.load
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# --- Output Schemas ---

class BoundingBox(BaseModel):
    x_min: int = Field(..., description="Top-left X coordinate")
    y_min: int = Field(..., description="Top-left Y coordinate")
    x_max: int = Field(..., description="Bottom-right X coordinate")
    y_max: int = Field(..., description="Bottom-right Y coordinate")
    confidence: float = Field(..., description="Detection confidence score")

class DetectionOutput(BaseModel):
    people_count: int = Field(0, description="Total number of people detected")
    density_score: float = Field(0.0, description="Calculated crowd density score between 0 and 1")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list, description="List of bounding boxes for detected people")
    inference_time_ms: float = Field(0.0, description="Time taken for inference in milliseconds")

# --- Agent Implementation ---

class CrowdDetectionAgent:
    """
    Agent 1: Crowd Detection Agent
    Responsible for receiving an image, running YOLO inference to detect people,
    counting them, and calculating a crowd density score.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        """
        Initializes the agent, loads the YOLO model, and determines the best hardware device.
        
        Args:
            model_path: Path to the YOLO weights file. Defaults to YOLOv8 nano for speed.
            conf_threshold: Minimum confidence threshold to consider a detection valid.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Determine the optimal device (CUDA, MPS, or CPU)
        self.device = self._get_optimal_device()
        logger.info(f"Using compute device: {self.device}")
        
        # Load the model logic
        self.model = self._load_model()
        
    def _get_optimal_device(self) -> str:
        """Determines the best available compute device for PyTorch."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> YOLO:
        """Loads the YOLO model into memory and moves it to the appropriate device."""
        try:
            logger.info(f"Loading YOLO model from {self.model_path}...")
            # ultralytics handles downloading standard models if not found locally
            model = YOLO(self.model_path)
            model.to(self.device)
            logger.info("YOLO model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _calculate_density(self, boxes: List[BoundingBox], image_area: int) -> float:
        """
        Calculates a rudimentary density score based on the ratio of bounding box area
        to the total image area. Overlapping boxes are not perfectly accounted for in this 
        basic heuristic, but it serves as a solid baseline.
        
        Args:
            boxes: List of BoundingBox objects.
            image_area: Total area of the image in pixels.
            
        Returns:
            Float representing density score clamped between 0.0 and 1.0.
        """
        if not boxes or image_area <= 0:
            return 0.0
            
        total_bbox_area = 0
        for box in boxes:
            area = (box.x_max - box.x_min) * (box.y_max - box.y_min)
            total_bbox_area += area
            
        # The density score could exceed 1.0 if there's heavy overlap. Clamp to 1.0.
        # Alternatively, a more advanced approach calculates the union of all polygons.
        density = min(total_bbox_area / image_area, 1.0)
        return round(density, 4)

    def process_image(self, image: np.ndarray) -> DetectionOutput:
        """
        Main inference pipeline.
        
        Args:
            image: OpenCV image array (numpy ndarray) in BGR format.
            
        Returns:
            DetectionOutput schema containing count, density, and bounding boxes.
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid image input provided.")
            raise ValueError("Invalid image input. Must be a numpy ndarray.")
            
        height, width = image.shape[:2]
        image_area = height * width
        
        start_time = time.perf_counter()
        
        try:
            # Run inference
            # YOLO class 0 is 'person' in the COCO dataset
            results = self.model(image, conf=self.conf_threshold, classes=[0], verbose=False)
            
            bounding_boxes = []
            
            # Parse results (assuming single image batch)
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Extract coordinates and convert to standard integers
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                bounding_boxes.append(
                    BoundingBox(
                        x_min=int(x1),
                        y_min=int(y1),
                        x_max=int(x2),
                        y_max=int(y2),
                        confidence=round(conf, 4)
                    )
                )
                
            people_count = len(bounding_boxes)
            density_score = self._calculate_density(bounding_boxes, image_area)
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"Processed image in {inference_time_ms:.2f}ms. Count: {people_count}, Density: {density_score}")
            
            return DetectionOutput(
                people_count=people_count,
                density_score=density_score,
                bounding_boxes=bounding_boxes,
                inference_time_ms=round(inference_time_ms, 2)
            )
            
        except Exception as e:
            logger.error(f"Error during image processing inference: {str(e)}")
            raise RuntimeError(f"Inference pipeline failed: {str(e)}")

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy image for testing
    dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    try:
        agent = CrowdDetectionAgent(model_path="yolov8n.pt", conf_threshold=0.3)
        output = agent.process_image(dummy_image)
        print(output.model_dump_json(indent=2))
    except Exception as ex:
        print(f"Test failed: {ex}")
