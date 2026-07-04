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
        Analyzes the bounding boxes to count people in each grid zone.
        
        Args:
            data: HotspotInput containing bounding boxes and image dimensions.
            
        Returns:
            HotspotOutput schema with zone counts and the identified hotspot.
        """
        # Initialize zone counts dictionary
        zone_counts: Dict[str, int] = {}
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                zone_counts[self._get_zone_name(r, c)] = 0

        if not data.bounding_boxes:
            return HotspotOutput(zone_counts=zone_counts, hotspot_zone=None, max_zone_count=0)

        # Calculate width and height of each cell
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
        Creates a KDE-like gaussian density map based on bounding box centers.
        
        Args:
            image: The original image numpy array (BGR).
            data: HotspotInput containing bounding boxes.
            
        Returns:
            A new numpy array containing the image blended with the heatmap.
        """
        if image is None:
            raise ValueError("Invalid image provided for heatmap generation.")
            
        height, width = image.shape[:2]
        
        # Create an empty floating point accumulator map
        density_map = np.zeros((height, width), dtype=np.float32)
        
        # Parameters for the gaussian blob
        # The size of the blob represents the approximate physical area of a person
        sigma = min(width, height) // 20 
        
        for bbox in data.bounding_boxes:
            cx = int((bbox.x_min + bbox.x_max) / 2)
            cy = int((bbox.y_min + bbox.y_max) / 2)
            
            # Add a value at the center point. 
            # In a highly optimized system, we would add the 2D gaussian directly here.
            # For performance and simplicity, we increment points and blur later.
            if 0 <= cx < width and 0 <= cy < height:
                density_map[cy, cx] += 1.0
                
        # Apply a large gaussian blur to spread the density
        density_map = cv2.GaussianBlur(density_map, (0, 0), sigmaX=sigma, sigmaY=sigma)
        
        # Normalize the density map to [0, 255] for coloring
        if np.max(density_map) > 0:
            density_map = (density_map / np.max(density_map)) * 255
            
        density_map = np.uint8(density_map)
        
        # Apply a colormap (JET or INFERNO works well for heatmaps)
        heatmap_color = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
        
        # Blend the heatmap with the original image
        alpha = 0.5  # heatmap transparency
        overlay = cv2.addWeighted(heatmap_color, alpha, image, 1 - alpha, 0)
        
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
