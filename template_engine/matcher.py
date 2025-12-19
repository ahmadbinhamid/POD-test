"""
Template Matching Algorithm with DETAILED SCORE LOGGING
Uses 4-component weighted scoring with comprehensive debugging
"""

from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from PIL import Image
import imagehash

logger = logging.getLogger("template_engine.matcher")


class TemplateMatcher:
    """
    Intelligent template matching using multiple algorithms
    """
    
    def __init__(self):
        logger.info("🎯 TemplateMatcher initialized")
        # Weights for scoring components
        self.weights = {
            "text": 0.35,      # Text pattern matching
            "visual": 0.30,    # Visual feature similarity
            "layout": 0.20,    # Layout similarity
            "image_hash": 0.15 # Perceptual image hashing
        }
        
        logger.info(f"  Scoring weights: {self.weights}")
    
    def match_templates(
        self, 
        image: Image.Image, 
        templates: List[Dict]
    ) -> Tuple[Optional[Dict], float, List[Dict]]:
        """
        Match document image against templates
        
        Args:
            image: PIL Image of document
            templates: List of template dicts
            
        Returns:
            (matched_template, confidence_score, suggested_templates)
        """
        logger.info("\n" + "=" * 80)
        logger.info("🎯 TEMPLATE_MATCHING: Starting template matching algorithm")
        logger.info("=" * 80)
        logger.info(f"  Input image size: {image.size}")
        logger.info(f"  Input image mode: {image.mode}")
        logger.info(f"  Templates to evaluate: {len(templates)}")
        
        if not templates:
            logger.warning("  ⚠️  No templates provided for matching!")
            return None, 0.0, []
        
        # Calculate scores for each template
        scored_templates = []
        
        for idx, template in enumerate(templates, 1):
            template_id = template.get("template_id", "Unknown")
            template_name = template.get("template_name", "Unknown")
            threshold = template.get("identification", {}).get("confidence_threshold", 0.75)
            
            logger.info("\n" + "-" * 80)
            logger.info(f"📋 EVALUATING TEMPLATE {idx}/{len(templates)}")
            logger.info("-" * 80)
            logger.info(f"  Template ID: {template_id}")
            logger.info(f"  Template Name: {template_name}")
            logger.info(f"  Confidence Threshold: {threshold}")
            
            # Calculate component scores with detailed logging
            scores = {}
            
            # 1. Text Pattern Matching (35%)
            logger.info("\n  🔍 COMPONENT 1: TEXT PATTERN MATCHING (Weight: 35%)")
            logger.info("  " + "-" * 76)
            scores["text"] = self._calculate_text_score(image, template)
            weighted_text = scores["text"] * self.weights["text"]
            logger.info(f"  ✓ Text Score: {scores['text']:.4f}")
            logger.info(f"  ✓ Weighted Contribution: {scores['text']:.4f} × {self.weights['text']} = {weighted_text:.4f}")
            
            # 2. Visual Feature Comparison (30%)
            logger.info("\n  🎨 COMPONENT 2: VISUAL FEATURE COMPARISON (Weight: 30%)")
            logger.info("  " + "-" * 76)
            scores["visual"] = self._calculate_visual_score(image, template)
            weighted_visual = scores["visual"] * self.weights["visual"]
            logger.info(f"  ✓ Visual Score: {scores['visual']:.4f}")
            logger.info(f"  ✓ Weighted Contribution: {scores['visual']:.4f} × {self.weights['visual']} = {weighted_visual:.4f}")
            
            # 3. Layout Similarity (20%)
            logger.info("\n  📐 COMPONENT 3: LAYOUT SIMILARITY (Weight: 20%)")
            logger.info("  " + "-" * 76)
            scores["layout"] = self._calculate_layout_score(image, template)
            weighted_layout = scores["layout"] * self.weights["layout"]
            logger.info(f"  ✓ Layout Score: {scores['layout']:.4f}")
            logger.info(f"  ✓ Weighted Contribution: {scores['layout']:.4f} × {self.weights['layout']} = {weighted_layout:.4f}")
            
            # 4. Image Hash Comparison (15%)
            logger.info("\n  #️⃣ COMPONENT 4: IMAGE HASH COMPARISON (Weight: 15%)")
            logger.info("  " + "-" * 76)
            scores["image_hash"] = self._calculate_image_hash_score(image, template)
            weighted_hash = scores["image_hash"] * self.weights["image_hash"]
            logger.info(f"  ✓ Image Hash Score: {scores['image_hash']:.4f}")
            logger.info(f"  ✓ Weighted Contribution: {scores['image_hash']:.4f} × {self.weights['image_hash']} = {weighted_hash:.4f}")
            
            # Calculate weighted overall score
            overall_score = (
                scores["text"] * self.weights["text"] +
                scores["visual"] * self.weights["visual"] +
                scores["layout"] * self.weights["layout"] +
                scores["image_hash"] * self.weights["image_hash"]
            )
            
            # Detailed calculation breakdown
            logger.info("\n  📊 FINAL SCORE CALCULATION:")
            logger.info("  " + "-" * 76)
            logger.info(f"  Text:       {scores['text']:.4f} × {self.weights['text']} = {weighted_text:.4f}")
            logger.info(f"  Visual:     {scores['visual']:.4f} × {self.weights['visual']} = {weighted_visual:.4f}")
            logger.info(f"  Layout:     {scores['layout']:.4f} × {self.weights['layout']} = {weighted_layout:.4f}")
            logger.info(f"  Image Hash: {scores['image_hash']:.4f} × {self.weights['image_hash']} = {weighted_hash:.4f}")
            logger.info("  " + "-" * 76)
            logger.info(f"  🎯 OVERALL SCORE: {overall_score:.4f} ({overall_score * 100:.2f}%)")
            logger.info(f"  📏 THRESHOLD:     {threshold:.4f} ({threshold * 100:.2f}%)")
            
            # Determine pass/fail
            passes_threshold = overall_score >= threshold
            status = "✅ PASSES" if passes_threshold else "❌ FAILS"
            logger.info(f"  🏁 STATUS: {status}")
            
            scored_templates.append({
                "template": template,
                "template_id": template_id,
                "template_name": template_name,
                "match_score": overall_score,
                "score_breakdown": scores,
                "weighted_scores": {
                    "text": weighted_text,
                    "visual": weighted_visual,
                    "layout": weighted_layout,
                    "image_hash": weighted_hash
                },
                "confidence_breakdown": {
                    "text_similarity": scores["text"],
                    "visual_similarity": scores["visual"],
                    "layout_similarity": scores["layout"],
                    "image_hash_similarity": scores["image_hash"]
                },
                "passes_threshold": passes_threshold
            })
        
        # Sort by score (highest first)
        scored_templates.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Get best template
        best_template = scored_templates[0]
        threshold = best_template["template"].get("identification", {}).get("confidence_threshold", 0.75)
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 MATCHING RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  🥇 Best Match: {best_template['template_name']}")
        logger.info(f"  📈 Score: {best_template['match_score']:.4f} ({best_template['match_score'] * 100:.2f}%)")
        logger.info(f"  📏 Threshold: {threshold:.4f} ({threshold * 100:.2f}%)")
        
        if best_template['match_score'] >= threshold:
            logger.info(f"  ✅ STATUS: TEMPLATE MATCHED!")
            logger.info(f"  ✨ Margin: +{(best_template['match_score'] - threshold) * 100:.2f}% above threshold")
        else:
            logger.info(f"  ❌ STATUS: NO MATCH (Below threshold)")
            logger.info(f"  ⚠️  Gap: -{(threshold - best_template['match_score']) * 100:.2f}% below threshold")
        
        # Log all template rankings
        logger.info(f"\n  📋 Template Rankings:")
        for i, scored in enumerate(scored_templates, 1):
            status_icon = "✅" if scored['passes_threshold'] else "❌"
            logger.info(f"    {i}. {status_icon} {scored['template_name']}: {scored['match_score']:.4f}")
        
        # Prepare suggestions (top 5)
        suggestions = []
        for i, scored in enumerate(scored_templates[:5], 1):
            suggestions.append({
                "template_id": scored["template_id"],
                "template_name": scored["template_name"],
                "match_score": round(scored["match_score"], 2),
                "priority": i,
                "matched_patterns": self._get_matched_patterns(image, scored["template"]),
                "confidence_breakdown": scored["confidence_breakdown"]
            })
        
        logger.info("=" * 80 + "\n")
        
        # Return matched template if above threshold
        if best_template["match_score"] >= threshold:
            return best_template["template"], best_template["match_score"], suggestions
        else:
            return None, best_template["match_score"], suggestions
    
    def _calculate_text_score(self, image: Image.Image, template: Dict) -> float:
        """Calculate text pattern matching score"""
        patterns = template.get("identification", {}).get("text_patterns", [])
        
        if not patterns:
            logger.info("    ⚠️  No text patterns configured")
            return 0.5
        
        logger.info(f"    Configured patterns: {len(patterns)}")
        for i, pattern in enumerate(patterns, 1):
            logger.info(f"      {i}. \"{pattern}\"")
        
        try:
            import pytesseract
            text = pytesseract.image_to_string(image).lower()
            logger.info(f"    Extracted text length: {len(text)} characters")
            
            # Count pattern matches
            matches = 0
            for pattern in patterns:
                if pattern.lower() in text:
                    matches += 1
                    logger.info(f"      ✅ MATCH: \"{pattern}\" found in document")
                else:
                    logger.info(f"      ❌ MISS:  \"{pattern}\" not found")
            
            score = matches / len(patterns)
            logger.info(f"    Result: {matches}/{len(patterns)} patterns matched = {score:.4f}")
            return score
            
        except ImportError:
            logger.warning("    ⚠️  pytesseract not available, using placeholder score")
            return 0.5
        except Exception as e:
            logger.error(f"    ❌ Error: {e}")
            return 0.0
    
    def _calculate_visual_score(self, image: Image.Image, template: Dict) -> float:
        """Calculate visual feature similarity"""
        ref_images = template.get("identification", {}).get("reference_images", [])
        
        logger.info(f"    Reference images configured: {len(ref_images)}")
        
        if not ref_images:
            logger.info("    ⚠️  No reference images, using placeholder score")
            return 0.5
        
        for i, ref in enumerate(ref_images, 1):
            logger.info(f"      {i}. {ref.get('file_path', 'N/A')}")
        
        # TODO: Implement actual histogram comparison
        logger.info("    ℹ️  Visual comparison not yet implemented")
        logger.info("    Using placeholder score: 0.7000")
        return 0.7
    
    def _calculate_layout_score(self, image: Image.Image, template: Dict) -> float:
        """Calculate layout similarity"""
        yolo_config = template.get("region_config", {}).get("yolo_config", {})
        classes = yolo_config.get("classes", [])
        
        logger.info(f"    Region classes configured: {len(classes)}")
        for cls in classes:
            logger.info(f"      - {cls.get('region_name')}")
        
        if not classes:
            logger.info("    ⚠️  No region classes, using placeholder score")
            return 0.5
        
        # TODO: Run YOLO and compare detected regions
        logger.info("    ℹ️  Layout comparison not yet implemented")
        logger.info("    Using placeholder score: 0.6500")
        return 0.65
    
    def _calculate_image_hash_score(self, image: Image.Image, template: Dict) -> float:
        """Calculate perceptual image hash similarity"""
        ref_images = template.get("identification", {}).get("reference_images", [])
        
        logger.info(f"    Reference images for hashing: {len(ref_images)}")
        
        if not ref_images:
            logger.info("    ⚠️  No reference images, using placeholder score")
            return 0.5
        
        try:
            input_hash = imagehash.average_hash(image)
            logger.info(f"    Input image hash: {input_hash}")
            
            # TODO: Load reference images and compare hashes
            logger.info("    ℹ️  Hash comparison not yet implemented")
            logger.info("    Using placeholder score: 0.6000")
            return 0.6
            
        except Exception as e:
            logger.error(f"    ❌ Error calculating hash: {e}")
            return 0.0
    
    def _get_matched_patterns(self, image: Image.Image, template: Dict) -> List[str]:
        """Get list of text patterns that matched"""
        patterns = template.get("identification", {}).get("text_patterns", [])
        
        try:
            import pytesseract
            text = pytesseract.image_to_string(image).lower()
            return [p for p in patterns if p.lower() in text]
        except:
            return []