import logging
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel, Field
import cv2
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# --- Schemas ---

from app.agents.detection_agent import BoundingBox

class HotspotInput(BaseModel):
    people_count: int = Field(..., description="Total number of people detected")
    bounding_boxes: List[BoundingBox] = Field(..., description="List of bounding boxes for detected people")
    image_width: int = Field(..., description="Width of the original image")
    image_height: int = Field(..., description="Height of the original image")
    density_map: Any = None

class HotspotOutput(BaseModel):
    zone_counts: Dict[str, int] = Field(..., description="Dictionary mapping zone names to people counts")
    hotspot_zone: str = Field(None, description="The zone name with the highest congestion/count")
    max_zone_count: int = Field(0, description="The number of people in the hotspot zone")

# --- Agent Implementation ---

class HotspotDetectionAgent:
    """
    Agent 2: Hotspot Detection Agent
    Responsible for taking bounding box outputs from the detection agent,
    dividing the space into grid zones, detecting the most congested hotspot,
    and generating heatmap visual data.
    """
    
    def __init__(self, grid_rows: int = 2, grid_cols: int = 2):
        """
        Initializes the spatial analytics agent.
        
        Args:
            grid_rows: Number of horizontal divisions (rows)
            grid_cols: Number of vertical divisions (columns)
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
    def _get_zone_name(self, row: int, col: int) -> str:
        """Helper to generate a readable zone name, e.g. A1, A2, B1"""
        # Map row to letter (A, B, C...) and col to number (1, 2, 3...)
        row_letter = chr(65 + row)  # 65 is 'A'
        col_number = col + 1
        return f"Zone_{row_letter}{col_number}"

    def analyze_zones(self, data: HotspotInput) -> HotspotOutput:
        """
        Analyzes the crowd density to count people in each grid zone.
        Supports both the direct CSRNet density map and fallback bounding boxes.
        
        Args:
            data: HotspotInput containing bounding boxes, image dimensions, and optional density map.
            
        Returns:
            HotspotOutput schema with zone counts and the identified hotspot.
        """
        # Initialize zone counts dictionary
        zone_counts: Dict[str, int] = {}
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                zone_counts[self._get_zone_name(r, c)] = 0

        # If density map is present, sum values in each grid cell directly (highly accurate)
        if data.density_map is not None:
            dm_height, dm_width = data.density_map.shape[:2]
            cell_width = dm_width / self.grid_cols
            cell_height = dm_height / self.grid_rows
            
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    x_start = int(c * cell_width)
                    x_end = int((c + 1) * cell_width)
                    y_start = int(r * cell_height)
                    y_end = int((r + 1) * cell_height)
                    
                    # Sum density values in this grid cell (pixels sum to total count)
                    zone_sum = np.sum(data.density_map[y_start:y_end, x_start:x_end])
                    zone_name = self._get_zone_name(r, c)
                    zone_counts[zone_name] = int(round(max(0.0, float(zone_sum))))
        
        # Fallback to bounding boxes if no density map is present
        elif data.bounding_boxes:
            cell_width = data.image_width / self.grid_cols
            cell_height = data.image_height / self.grid_rows
            
            for bbox in data.bounding_boxes:
                # Calculate the center point of the bounding box (pedestrian location)
                center_x = (bbox.x_min + bbox.x_max) / 2
                center_y = (bbox.y_min + bbox.y_max) / 2
                
                # Determine which grid column and row this center point falls into
                col_idx = int(center_x // cell_width)
                row_idx = int(center_y // cell_height)
                
                # Clamp indices in case the bounding box center is exactly on the image edge
                col_idx = max(0, min(col_idx, self.grid_cols - 1))
                row_idx = max(0, min(row_idx, self.grid_rows - 1))
                
                zone_name = self._get_zone_name(row_idx, col_idx)
                zone_counts[zone_name] += 1
                
        # Identify the hotspot
        hotspot_zone = None
        max_count = -1
        
        for zone, count in zone_counts.items():
            if count > max_count:
                max_count = count
                hotspot_zone = zone
                
        logger.info(f"Spatial Analysis Complete. Hotspot: {hotspot_zone} with {max_count} people.")
                
        return HotspotOutput(
            zone_counts=zone_counts,
            hotspot_zone=hotspot_zone,
            max_zone_count=max_count
        )

    def generate_heatmap_overlay(self, image: np.ndarray, data: HotspotInput) -> np.ndarray:
        """
        Generates a visual heatmap overlay using OpenCV.
        Supports both the direct CSRNet density map and fallback bounding boxes.
        
        Args:
            image: The original image numpy array (BGR).
            data: HotspotInput containing bounding boxes and optional density map.
            
        Returns:
            A new numpy array containing the image blended with the heatmap.
        """
        if image is None:
            raise ValueError("Invalid image provided for heatmap generation.")
            
        height, width = image.shape[:2]
        
        # If density map is present, resize and colorize it directly (CSRNet dense mode)
        if data.density_map is not None:
            density_map_resized = cv2.resize(data.density_map, (width, height), interpolation=cv2.INTER_CUBIC)
            density_map_resized = np.clip(density_map_resized, 0, None)
            
            max_val = np.max(density_map_resized)
            if max_val > 0:
                density_norm = (density_map_resized / max_val) * 255.0
            else:
                density_norm = density_map_resized
                
            density_uint8 = np.uint8(np.clip(density_norm, 0, 255))
            heatmap_color = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)
            
            # Use dynamic alpha mask so zero-density regions remain clean and unmasked
            alpha_mask = (density_uint8.astype(np.float32) / 255.0) * 0.55
            alpha_3ch = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
            
            overlay = (heatmap_color.astype(np.float32) * alpha_3ch + image.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)
            return overlay
            
        # Precise Person-Centric Heatmap (YOLO sparse mode)
        density_map = np.zeros((height, width), dtype=np.float32)
        
        for bbox in data.bounding_boxes:
            bw = bbox.x_max - bbox.x_min
            bh = bbox.y_max - bbox.y_min
            
            # Head/upper-body centroid
            cx = int(bbox.x_min + bw / 2.0)
            cy = int(bbox.y_min + min(bh / 3.0, 40.0))
            
            radius = max(15, int(min(bw, bh) * 0.6))
            sigma = max(5.0, radius / 2.0)
            
            # Draw local Gaussian blob around person
            y_min = max(0, cy - radius)
            y_max = min(height, cy + radius + 1)
            x_min = max(0, cx - radius)
            x_max = min(width, cx + radius + 1)
            
            if y_max > y_min and x_max > x_min:
                y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
                dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
                gaussian_blob = np.exp(-dist_sq / (2.0 * sigma**2))
                density_map[y_min:y_max, x_min:x_max] += gaussian_blob * 1.5
                
        # Normalize and colorize
        max_val = np.max(density_map)
        if max_val > 0:
            density_norm = (density_map / max_val) * 255.0
        else:
            density_norm = density_map
            
        density_uint8 = np.uint8(np.clip(density_norm, 0, 255))
        heatmap_color = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)
        
        # Smooth alpha mask: high heat glows brightly over people, zero heat leaves background natural
        alpha_mask = np.clip((density_uint8.astype(np.float32) / 255.0) * 0.65, 0.0, 0.65)
        alpha_3ch = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
        
        overlay = (heatmap_color.astype(np.float32) * alpha_3ch + image.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)
        return overlay

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate data
    img_h, img_w = 720, 1280
    
    simulated_input = HotspotInput(
        people_count=3,
        image_width=img_w,
        image_height=img_h,
        bounding_boxes=[
            BoundingBox(x_min=100, y_min=100, x_max=150, y_max=200, confidence=0.9), # Top-Left (A1)
            BoundingBox(x_min=120, y_min=110, x_max=160, y_max=210, confidence=0.8), # Top-Left (A1)
            BoundingBox(x_min=1000, y_min=600, x_max=1100, y_max=700, confidence=0.85) # Bottom-Right (B2)
        ]
    )
    
    agent = HotspotDetectionAgent(grid_rows=2, grid_cols=2)
    
    # 1. Spatial Analytics
    analysis_result = agent.analyze_zones(simulated_input)
    print(analysis_result.model_dump_json(indent=2))
    
    # 2. Heatmap Generation
    # Create a dummy dark image to test the overlay
    dummy_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    heatmap_result = agent.generate_heatmap_overlay(dummy_img, simulated_input)
    print(f"Heatmap overlay generated. Shape: {heatmap_result.shape}")
