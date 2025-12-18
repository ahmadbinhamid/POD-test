"""
Template Matching Algorithm
Scores templates based on text, visual, layout, and image similarity
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from PIL import Image
import imagehash

logger = logging.getLogger(__name__)


class TemplateMatcher:
    """
    Template matching with weighted scoring:
    - Text Patterns: 35%
    - Visual Features: 30%
    - Layout Similarity: 20%
    - Image Hash: 15%
    """
    
    THRESHOLD = 0.75
    TEXT_WEIGHT = 0.35
    VISUAL_WEIGHT = 0.30
    LAYOUT_WEIGHT = 0.20
    IMAGE_WEIGHT = 0.15
    
    def __init__(self):
        self.ocr_engine = None  # Will be set if needed
    
    def match_templates(
        self,
        image: np.ndarray,
        templates: List[Dict],
        yolo_results: Optional[Dict] = None,
        extracted_text: Optional[str] = None
    ) -> Tuple[Optional[Dict], float, List[Dict]]:
        """
        Match document image against templates
        
        Args:
            image: Document image (numpy array)
            templates: List of template dicts
            yolo_results: Optional YOLO detection results
            extracted_text: Optional pre-extracted text
            
        Returns:
            Tuple of (best_template, confidence, top_3_alternatives)
        """
        if not templates:
            logger.warning("No templates provided for matching")
            return None, 0.0, []
        
        scores = []
        
        for template in templates:
            try:
                score = self._calculate_template_score(
                    image, template, yolo_results, extracted_text
                )
                scores.append({
                    "template": template,
                    "score": score
                })
            except Exception as e:
                logger.error(f"Error scoring template {template.get('template_id')}: {e}")
                scores.append({
                    "template": template,
                    "score": 0.0
                })
        
        # Sort by score descending
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Get best match
        best = scores[0] if scores else {"template": None, "score": 0.0}
        best_template = best["template"] if best["score"] >= self.THRESHOLD else None
        best_score = best["score"]
        
        # Get top 3 for suggestions
        top_3 = [
            {
                "template_id": s["template"].get("template_id"),
                "template_name": s["template"].get("name", "Unknown"),
                "match_score": round(s["score"], 2),
                "priority": idx + 1
            }
            for idx, s in enumerate(scores[:3])
        ]
        
        return best_template, best_score, top_3
    
    def _calculate_template_score(
        self,
        image: np.ndarray,
        template: Dict,
        yolo_results: Optional[Dict],
        extracted_text: Optional[str]
    ) -> float:
        """Calculate weighted score for a template"""
        
        # 1. Text Pattern Score (35%)
        text_score = self._calculate_text_score(extracted_text, template)
        
        # 2. Visual Feature Score (30%)
        visual_score = self._calculate_visual_score(yolo_results, template)
        
        # 3. Layout Score (20%)
        layout_score = self._calculate_layout_score(yolo_results, template)
        
        # 4. Image Hash Score (15%)
        image_score = self._calculate_image_score(image, template)
        
        # Weighted sum
        final_score = (
            text_score * self.TEXT_WEIGHT +
            visual_score * self.VISUAL_WEIGHT +
            layout_score * self.LAYOUT_WEIGHT +
            image_score * self.IMAGE_WEIGHT
        )
        
        logger.debug(
            f"Template {template.get('template_id')}: "
            f"text={text_score:.2f}, visual={visual_score:.2f}, "
            f"layout={layout_score:.2f}, image={image_score:.2f}, "
            f"final={final_score:.2f}"
        )
        
        return final_score
    
    def _calculate_text_score(self, extracted_text: Optional[str], template: Dict) -> float:
        """Calculate text pattern matching score (35%)"""
        if not extracted_text or not template.get("identification_patterns"):
            return 0.0
        
        patterns = template.get("identification_patterns", [])
        if not patterns:
            return 0.0
        
        extracted_text = extracted_text.lower()
        
        matched_weight = 0.0
        total_weight = 0.0
        
        for pattern_obj in patterns:
            pattern = pattern_obj.get("pattern", "").lower()
            weight = pattern_obj.get("weight", 1.0)
            total_weight += weight
            
            # Check if pattern exists in text
            if pattern in extracted_text:
                matched_weight += weight
            # Also try regex matching
            elif re.search(re.escape(pattern), extracted_text, re.IGNORECASE):
                matched_weight += weight
        
        score = matched_weight / total_weight if total_weight > 0 else 0.0
        return min(score, 1.0)
    
    def _calculate_visual_score(self, yolo_results: Optional[Dict], template: Dict) -> float:
        """Calculate visual feature score (30%)"""
        if not yolo_results or not template.get("region_config"):
            return 0.5  # Neutral score if no YOLO results
        
        region_config = template.get("region_config", {})
        expected_regions = region_config.get("regions", [])
        
        if not expected_regions:
            return 0.5
        
        # Get confidence scores for detected regions
        confidences = []
        for region in expected_regions:
            region_name = region.get("name", "")
            # Check if this region was detected by YOLO
            if region_name in yolo_results.get("detected_regions", {}):
                conf = yolo_results["detected_regions"][region_name].get("confidence", 0.0)
                confidences.append(conf)
        
        # Average confidence
        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.3  # Low score if no regions matched
    
    def _calculate_layout_score(self, yolo_results: Optional[Dict], template: Dict) -> float:
        """Calculate layout similarity score (20%)"""
        if not yolo_results or not template.get("region_config"):
            return 0.5  # Neutral score
        
        region_config = template.get("region_config", {})
        
        # If using coordinate-based detection, compare positions
        if region_config.get("detection_method") == "coordinates":
            coordinate_regions = region_config.get("coordinate_regions", [])
            detected_regions = yolo_results.get("detected_regions", {})
            
            if not coordinate_regions or not detected_regions:
                return 0.5
            
            position_diffs = []
            
            for coord_region in coordinate_regions:
                region_name = coord_region.get("region_name", "")
                expected_coords = coord_region.get("coordinates", {})
                
                if region_name in detected_regions:
                    detected_box = detected_regions[region_name].get("bbox", {})
                    
                    # Calculate position difference (normalized)
                    x_diff = abs(expected_coords.get("x", 0) - detected_box.get("x", 0))
                    y_diff = abs(expected_coords.get("y", 0) - detected_box.get("y", 0))
                    
                    # Normalize by image size (assume 1000x1000 for now)
                    normalized_diff = (x_diff + y_diff) / 2000.0
                    position_diffs.append(normalized_diff)
            
            if position_diffs:
                avg_diff = sum(position_diffs) / len(position_diffs)
                return max(0.0, 1.0 - avg_diff)
        
        return 0.5
    
    def _calculate_image_score(self, image: np.ndarray, template: Dict) -> float:
        """Calculate image similarity score (15%)"""
        reference_images = template.get("reference_images", [])
        
        if not reference_images:
            return 0.5  # Neutral score if no reference images
        
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Calculate hash of document
            doc_hash = imagehash.phash(pil_image)
            
            # Compare with reference images
            similarities = []
            
            for ref_img_obj in reference_images[:3]:  # Check top 3 reference images
                ref_url = ref_img_obj.get("image_url", "")
                
                # In production, you'd fetch and hash the reference image
                # For now, we'll use a placeholder score
                # TODO: Implement actual image fetching and hashing
                
                # Placeholder: assume moderate similarity
                similarities.append(0.6)
            
            if similarities:
                return max(similarities)  # Return best similarity
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating image hash: {e}")
            return 0.5
    
    def extract_text_from_image(self, image: np.ndarray) -> str:
        """Extract text from image using OCR"""
        try:
            import pytesseract
            
            # Convert to PIL Image
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract text
            text = pytesseract.image_to_string(pil_image)
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""