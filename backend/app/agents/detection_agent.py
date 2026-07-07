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

import scipy.ndimage as ndimage
from app.models.csrnet import get_csrnet_model, CSRNet

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
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.015):
        """
        Initializes the agent, loads the CSRNet model, and determines the best hardware device.
        
        Args:
            model_path: Path to the CSRNet weights file. Defaults to backend/data/CSRNet.pth.
            conf_threshold: Minimum threshold in density map to consider a local peak a person.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.last_density_map = None
        
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

    def _load_model(self) -> CSRNet:
        """Loads the CSRNet model into memory and moves it to the appropriate device."""
        try:
            from app.models.csrnet import WEIGHTS_PATH
            path = self.model_path if self.model_path else WEIGHTS_PATH
            logger.info(f"Loading CSRNet model from {path}...")
            model = get_csrnet_model(weights_path=path, device=self.device)
            logger.info("CSRNet model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load CSRNet model: {str(e)}")
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
            # Preprocess the image for CSRNet
            # 1. Convert BGR to RGB
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Scale pixels to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # 3. ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            
            # 4. HWC to CHW format
            img = img.transpose((2, 0, 1))
            
            # 5. Convert to tensor and send to compute device
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
            
            # Run CSRNet inference
            with torch.no_grad():
                output = self.model(img_tensor)
                
            # Density map is the single-channel output
            density_map = output.squeeze().cpu().numpy()
            
            # Clip negative values to 0 since density cannot be negative
            density_map = np.clip(density_map, 0.0, None)
            
            # Store the density map on the instance so other agents (HotspotAgent) can access it
            self.last_density_map = density_map
            
            # Sum density map values to get total crowd count
            # CSRNet maps represent pixel density of crowd, sum(map) equals total people count
            predicted_count = float(np.sum(density_map))
            people_count = int(round(predicted_count))
            
            # Extract local peaks (heads) from density map to generate pseudo-bounding boxes
            # This maintains backward compatibility with downstream agents expecting bounding boxes
            bounding_boxes = []
            scale = 8 # CSRNet output is 1/8 of original image dimensions due to pooling layers
            
            # Find local maxima in a 3x3 neighborhood
            neighborhood_size = 3
            data_max = ndimage.maximum_filter(density_map, neighborhood_size)
            maxima = (density_map == data_max)
            
            # Filter background noise using threshold
            data_min = ndimage.minimum_filter(density_map, neighborhood_size)
            diff = ((data_max - data_min) > self.conf_threshold)
            maxima[diff == 0] = 0
            
            # Find peaks using labeling
            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            
            for dy, dx in slices:
                y_center = (dy.start + dy.stop - 1) / 2.0
                x_center = (dx.start + dx.stop - 1) / 2.0
                
                # Scale coordinates back to original image
                orig_x = int(x_center * scale)
                orig_y = int(y_center * scale)
                
                # Represent head as a 30x30 bounding box
                size = 15
                x_min = max(0, orig_x - size)
                y_min = max(0, orig_y - size)
                x_max = min(width - 1, orig_x + size)
                y_max = min(height - 1, orig_y + size)
                
                conf = float(density_map[int(y_center), int(x_center)])
                
                bounding_boxes.append(
                    BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        confidence=round(conf, 4)
                    )
                )
                
            # If the peak detection missed some heavily congested overlapping areas,
            # we want to ensure the list of boxes still correlates reasonably with the count.
            # However, people_count is the ground-truth estimate from the density map integral.
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
        agent = CrowdDetectionAgent()
        output = agent.process_image(dummy_image)
        print(output.model_dump_json(indent=2))
    except Exception as ex:
        print(f"Test failed: {ex}")
