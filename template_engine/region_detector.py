"""
Dynamic Region Detection
Supports YOLO, Coordinates, and Hybrid detection methods
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class RegionDetector:
    """
    Detects regions using template-specified method:
    - YOLO: Use YOLO model
    - Coordinates: Use fixed coordinates
    - Hybrid: Try YOLO first, fallback to coordinates
    """
    
    def __init__(self):
        self.yolo_models = {}  # Cache YOLO models
    
    def detect_regions(
        self,
        image: np.ndarray,
        template: Dict,
        category: str
    ) -> Dict[str, np.ndarray]:
        """
        Detect regions based on template configuration
        
        Args:
            image: Document image
            template: Template dict with region_config
            category: Document category (BOL, Receipt, etc.)
            
        Returns:
            Dict of {region_name: cropped_image}
        """
        region_config = template.get("region_config", {})
        detection_method = region_config.get("detection_method", "yolo").lower()
        
        if detection_method == "yolo":
            return self._detect_yolo(image, region_config, category)
        elif detection_method == "coordinates":
            return self._detect_coordinates(image, region_config)
        elif detection_method == "hybrid":
            return self._detect_hybrid(image, region_config, category)
        else:
            logger.warning(f"Unknown detection method: {detection_method}, using YOLO")
            return self._detect_yolo(image, region_config, category)
    
    def _detect_yolo(
        self,
        image: np.ndarray,
        region_config: Dict,
        category: str
    ) -> Dict[str, np.ndarray]:
        """Detect regions using YOLO model"""
        yolo_config = region_config.get("yolo_config", {})
        model_path = yolo_config.get("model_path")
        confidence_threshold = yolo_config.get("confidence_threshold", 0.25)
        
        # Use default model path based on category if not specified
        if not model_path:
            if category.lower() == "bol":
                model_path = "Models/bol_regions_best.pt"
            elif category.lower() == "receipt":
                model_path = "Models/receipts_regions_best.pt"
            else:
                logger.error(f"No YOLO model specified for category: {category}")
                return {}
        
        try:
            # Load model (with caching)
            if model_path not in self.yolo_models:
                logger.info(f"Loading YOLO model: {model_path}")
                self.yolo_models[model_path] = YOLO(model_path)
            
            model = self.yolo_models[model_path]
            
            # Run detection
            results = model(image, conf=confidence_threshold, verbose=False)
            
            # Extract regions
            regions = {}
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id] if hasattr(model, 'names') else f"region_{i}"
                    
                    # Crop region
                    cropped = image[y1:y2, x1:x2]
                    regions[class_name] = cropped
                    
                    logger.debug(f"Detected region: {class_name} at [{x1},{y1},{x2},{y2}]")
            
            return regions
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return {}
    
    def _detect_coordinates(
        self,
        image: np.ndarray,
        region_config: Dict
    ) -> Dict[str, np.ndarray]:
        """Detect regions using fixed coordinates"""
        coordinate_regions = region_config.get("coordinate_regions", [])
        
        if not coordinate_regions:
            logger.warning("No coordinate regions specified")
            return {}
        
        regions = {}
        img_height, img_width = image.shape[:2]
        
        for region_obj in coordinate_regions:
            try:
                region_name = region_obj.get("region_name", "")
                coords = region_obj.get("coordinates", {})
                
                # Get coordinates (can be absolute or relative)
                x = coords.get("x", 0)
                y = coords.get("y", 0)
                width = coords.get("width", 0)
                height = coords.get("height", 0)
                
                # If coordinates are relative (0-1), convert to absolute
                if x <= 1 and y <= 1 and width <= 1 and height <= 1:
                    x = int(x * img_width)
                    y = int(y * img_height)
                    width = int(width * img_width)
                    height = int(height * img_height)
                else:
                    x, y, width, height = int(x), int(y), int(width), int(height)
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, img_width))
                y = max(0, min(y, img_height))
                x2 = max(0, min(x + width, img_width))
                y2 = max(0, min(y + height, img_height))
                
                # Crop region
                if x2 > x and y2 > y:
                    cropped = image[y:y2, x:x2]
                    regions[region_name] = cropped
                    logger.debug(f"Detected region: {region_name} at [{x},{y},{x2},{y2}]")
                else:
                    logger.warning(f"Invalid coordinates for region: {region_name}")
                    
            except Exception as e:
                logger.error(f"Error detecting region {region_obj.get('region_name')}: {e}")
        
        return regions
    
    def _detect_hybrid(
        self,
        image: np.ndarray,
        region_config: Dict,
        category: str
    ) -> Dict[str, np.ndarray]:
        """Hybrid detection: Try YOLO first, fallback to coordinates"""
        # Try YOLO first
        logger.info("Attempting YOLO detection (hybrid mode)")
        yolo_regions = self._detect_yolo(image, region_config, category)
        
        if yolo_regions and len(yolo_regions) > 0:
            logger.info(f"YOLO detection successful: {len(yolo_regions)} regions found")
            return yolo_regions
        
        # Fallback to coordinates
        logger.info("YOLO detection failed or no regions found, falling back to coordinates")
        coord_regions = self._detect_coordinates(image, region_config)
        
        if coord_regions and len(coord_regions) > 0:
            logger.info(f"Coordinate detection successful: {len(coord_regions)} regions found")
            return coord_regions
        
        logger.warning("Both YOLO and coordinate detection failed")
        return {}
    
    def get_full_image_as_region(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Return full image as single region (fallback)"""
        return {"full_document": image}