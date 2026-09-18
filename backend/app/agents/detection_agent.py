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

from ultralytics import YOLO

class CrowdDetectionAgent:
    """
    Agent 1: Hybrid Crowd Detection Agent
    Combines YOLOv8 (Object Detection) for sparse & close-up scenes with 
    CSRNet (Dilated CNN Density Estimation) for dense & congested crowds.
    """
    
    def __init__(self, csrnet_path: str = None, yolo_path: str = "yolov8n.pt", conf_threshold: float = 0.015):
        """
        Initializes both YOLOv8 and CSRNet models.
        """
        self.csrnet_path = csrnet_path
        self.yolo_path = yolo_path
        self.conf_threshold = conf_threshold
        self.last_density_map = None
        self.last_mode = "HYBRID"
        
        # Determine the optimal device (CUDA, MPS, or CPU)
        self.device = self._get_optimal_device()
        logger.info(f"Using compute device: {self.device}")
        
        # Load models
        self.csrnet_model = self._load_csrnet_model()
        self.yolo_model = self._load_yolo_model()
        
    def _get_optimal_device(self) -> str:
        """Determines the best available compute device for PyTorch."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_csrnet_model(self) -> CSRNet:
        """Loads the CSRNet model into memory."""
        try:
            from app.models.csrnet import WEIGHTS_PATH
            path = self.csrnet_path if self.csrnet_path else WEIGHTS_PATH
            logger.info(f"Loading CSRNet model from {path}...")
            model = get_csrnet_model(weights_path=path, device=self.device)
            logger.info("CSRNet model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load CSRNet model: {str(e)}")
            raise RuntimeError(f"CSRNet initialization failed: {str(e)}")

    def _load_yolo_model(self) -> YOLO:
        """Loads the YOLOv8 model for sparse object detection."""
        try:
            logger.info(f"Loading YOLOv8 model from {self.yolo_path}...")
            model = YOLO(self.yolo_path)
            logger.info("YOLOv8 model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {str(e)}")
            raise RuntimeError(f"YOLOv8 initialization failed: {str(e)}")

    def _calculate_density_from_map(self, density_map: np.ndarray, people_count: int, image_area: int) -> float:
        """
        Calculates crowd density score (0.0 to 1.0) directly from continuous CSRNet density map.
        """
        if density_map is None or density_map.size == 0 or people_count <= 0:
            return 0.0
            
        active_threshold = 0.003
        active_pixels = np.count_nonzero(density_map > active_threshold)
        spatial_coverage = active_pixels / max(1, density_map.size)
        
        active_values = density_map[density_map > active_threshold]
        if len(active_values) > 0:
            mean_intensity = float(np.mean(active_values))
            intensity_score = min(1.0, mean_intensity / 0.035)
        else:
            intensity_score = 0.0
            
        volume_factor = min(1.0, (people_count * 5000.0) / max(1, image_area))
        raw_density = (0.45 * spatial_coverage) + (0.35 * intensity_score) + (0.20 * volume_factor)
        return round(float(min(1.0, max(0.0, raw_density))), 4)

    def _calculate_sparse_density(self, boxes: List[BoundingBox], image_area: int, people_count: int) -> float:
        """
        Calculates density score for sparse/medium scenes based on YOLO detections.
        """
        if not boxes or image_area <= 0 or people_count <= 0:
            return 0.0
        total_bbox_area = sum((b.x_max - b.x_min) * (b.y_max - b.y_min) for b in boxes)
        coverage = min(1.0, total_bbox_area / image_area)
        # Moderate weighting for sparse scenes
        density = (0.6 * coverage) + (0.4 * min(1.0, people_count / 30.0))
        return round(float(min(1.0, max(0.0, density))), 4)

    def process_image(self, image: np.ndarray) -> DetectionOutput:
        """
        Hybrid inference pipeline:
        1. Runs fast YOLOv8 person detection.
        2. If crowd is sparse (< 25 people), returns exact person detections and coordinates.
        3. If crowd is dense/occluded (>= 25 people or CSRNet integral >> YOLO), engages CSRNet.
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid image input provided.")
            raise ValueError("Invalid image input. Must be a numpy ndarray.")
            
        orig_height, orig_width = image.shape[:2]
        image_area = orig_height * orig_width
        start_time = time.perf_counter()
        
        # Downscale oversized images (e.g. 4K/DSLR photos > 1920px) for fast and accurate inference
        max_dim = 1920
        if max(orig_height, orig_width) > max_dim:
            scale_ratio = max_dim / float(max(orig_height, orig_width))
            proc_img = cv2.resize(image, (int(orig_width * scale_ratio), int(orig_height * scale_ratio)), interpolation=cv2.INTER_AREA)
        else:
            proc_img = image
            scale_ratio = 1.0
            
        proc_h, proc_w = proc_img.shape[:2]
        
        try:
            # -------------------------------------------------------------
            # Stage 1: Fast YOLOv8 Object Detection (Person class = 0)
            # -------------------------------------------------------------
            yolo_results = self.yolo_model(proc_img, classes=[0], conf=0.25, verbose=False)
            yolo_boxes = []
            
            if yolo_results and len(yolo_results[0].boxes) > 0:
                for box in yolo_results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    # Scale back to original image dimensions if resized
                    x_min = int(coords[0] / scale_ratio)
                    y_min = int(coords[1] / scale_ratio)
                    x_max = int(coords[2] / scale_ratio)
                    y_max = int(coords[3] / scale_ratio)
                    yolo_boxes.append(
                        BoundingBox(
                            x_min=max(0, x_min),
                            y_min=max(0, y_min),
                            x_max=min(orig_width - 1, x_max),
                            y_max=min(orig_height - 1, y_max),
                            confidence=round(conf, 4)
                        )
                    )
                    
            yolo_count = len(yolo_boxes)
            
            # -------------------------------------------------------------
            # Stage 2: CSRNet Density Map Estimation
            # -------------------------------------------------------------
            # Preprocess for CSRNet
            img_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = ((img_rgb - mean) / std).transpose((2, 0, 1))
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                csr_out = self.csrnet_model(img_tensor)
                
            raw_density_map = np.clip(csr_out.squeeze().cpu().numpy(), 0.0, None)
            csrnet_count = float(np.sum(raw_density_map))
            
            # -------------------------------------------------------------
            # Stage 3: Adaptive Hybrid Decision
            # -------------------------------------------------------------
            # Case A: Sparse / Medium Scene (e.g. 1 to 25 people in classroom/office/street)
            # YOLO provides 100% exact person bounding boxes and ignores background noise
            if yolo_count < 25 and csrnet_count < 40:
                self.last_mode = "YOLO_SPARSE_DETECTION"
                self.last_density_map = None # Hotspot agent will use exact YOLO person centroids
                people_count = yolo_count
                bounding_boxes = yolo_boxes
                density_score = self._calculate_sparse_density(bounding_boxes, image_area, people_count)
                
            # Case B: Dense / Congested Crowd Scene (e.g. 50 to 500+ people in concert/festival)
            # CSRNet continuous density integral provides true count where YOLO is occluded
            else:
                self.last_mode = "CSRNET_DENSE_ESTIMATION"
                self.last_density_map = raw_density_map
                people_count = int(round(max(csrnet_count, float(yolo_count))))
                density_score = self._calculate_density_from_map(raw_density_map, people_count, image_area)
                
                # Extract local peaks from density map for downstream compatibility
                bounding_boxes = []
                scale = (orig_width / raw_density_map.shape[1], orig_height / raw_density_map.shape[0])
                data_max = ndimage.maximum_filter(raw_density_map, 3)
                maxima = (raw_density_map == data_max)
                data_min = ndimage.minimum_filter(raw_density_map, 3)
                maxima[(data_max - data_min) <= self.conf_threshold] = 0
                labeled, num_objects = ndimage.label(maxima)
                slices = ndimage.find_objects(labeled)
                
                for dy, dx in slices:
                    yc = (dy.start + dy.stop - 1) / 2.0
                    xc = (dx.start + dx.stop - 1) / 2.0
                    orig_x = int(xc * scale[0])
                    orig_y = int(yc * scale[1])
                    size = 15
                    bounding_boxes.append(
                        BoundingBox(
                            x_min=max(0, orig_x - size),
                            y_min=max(0, orig_y - size),
                            x_max=min(orig_width - 1, orig_x + size),
                            y_max=min(orig_height - 1, orig_y + size),
                            confidence=round(float(raw_density_map[int(yc), int(xc)]), 4)
                        )
                    )
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Hybrid Mode: {self.last_mode} | Count: {people_count} (YOLO: {yolo_count}, CSRNet: {csrnet_count:.1f}) | Density: {density_score} | Time: {inference_time_ms:.1f}ms")
            
            return DetectionOutput(
                people_count=people_count,
                density_score=density_score,
                bounding_boxes=bounding_boxes,
                inference_time_ms=round(inference_time_ms, 2)
            )
            
        except Exception as e:
            logger.error(f"Error during hybrid detection inference: {str(e)}")
            raise RuntimeError(f"Hybrid inference pipeline failed: {str(e)}")

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
