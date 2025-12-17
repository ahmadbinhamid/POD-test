"""
Diagnostic script to check if your region detection is working correctly
Run this on your 3 sample files to visualize what's being detected
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

def visualize_bol_detections(image_path, save_path="debug_detection.jpg"):
    """
    Visualize what regions your YOLO model is detecting
    Args:
        image_path: Path to BOL image
        save_path: Where to save visualization
    """
    # Load model
    bol_region_detector = YOLO("Models/bol_regions_best.pt").to("cpu")
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Run detection
    results = bol_region_detector.predict(img, conf=0.51)
    
    # Class names
    class_names = {
        0: "stamp",
        1: "bill_of_lading",
        2: "customer_order_info",
        3: "signatures",
    }
    
    # Draw detections
    draw = ImageDraw.Draw(img)
    
    print(f"\n{'='*60}")
    print(f"DETECTION RESULTS FOR: {image_path}")
    print(f"{'='*60}\n")
    
    detection_count = {name: 0 for name in class_names.values()}
    
    for result in results:
        if result.boxes:
            for class_id, box, conf in zip(
                result.boxes.cls.tolist(),
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist()
            ):
                region_name = class_names.get(int(class_id), "unknown")
                detection_count[region_name] += 1
                
                x1, y1, x2, y2 = box
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw label
                label = f"{region_name} ({conf:.2f})"
                draw.text((x1, y1 - 20), label, fill="red")
                
                print(f"✓ Detected: {region_name}")
                print(f"  Confidence: {conf:.2f}")
                print(f"  Box: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
                print(f"  Size: {int(x2-x1)} x {int(y2-y1)} pixels\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY:")
    print(f"{'='*60}")
    for region_name, count in detection_count.items():
        status = "✓ FOUND" if count > 0 else "✗ MISSING"
        print(f"{status}: {region_name} (detected {count} time(s))")
    
    # Critical checks
    print(f"\n{'='*60}")
    print("CRITICAL FIELD CHECKS:")
    print(f"{'='*60}")
    
    issues = []
    
    if detection_count["stamp"] == 0:
        issues.append("⚠️  CRITICAL: No stamp region detected! OCR will fail.")
        print("❌ Stamp region: NOT DETECTED")
    else:
        print("✓ Stamp region: DETECTED")
    
    if detection_count["bill_of_lading"] == 0:
        issues.append("⚠️  WARNING: No bill_of_lading region detected")
        print("❌ Bill of Lading: NOT DETECTED")
    else:
        print("✓ Bill of Lading: DETECTED")
    
    if detection_count["customer_order_info"] == 0:
        issues.append("⚠️  WARNING: No customer_order_info region detected")
        print("❌ Customer Order Info: NOT DETECTED")
    else:
        print("✓ Customer Order Info: DETECTED")
    
    if detection_count["signatures"] == 0:
        print("⚠️  Signatures: NOT DETECTED (optional)")
    else:
        print("✓ Signatures: DETECTED")
    
    # Save visualization
    img.save(save_path)
    print(f"\n✓ Visualization saved to: {save_path}\n")
    
    if issues:
        print(f"\n{'='*60}")
        print("⚠️  ISSUES FOUND:")
        print(f"{'='*60}")
        for issue in issues:
            print(issue)
    
    return detection_count


def check_stamp_coverage(image_path):
    """
    Check if the detected stamp region covers the actual stamp area
    """
    print(f"\n{'='*60}")
    print("STAMP COVERAGE ANALYSIS")
    print(f"{'='*60}\n")
    
    bol_region_detector = YOLO("Models/bol_regions_best.pt").to("cpu")
    img = Image.open(image_path).convert("RGB")
    
    results = bol_region_detector.predict(img, conf=0.51)
    
    img_width, img_height = img.size
    
    for result in results:
        if result.boxes:
            for class_id, box in zip(
                result.boxes.cls.tolist(),
                result.boxes.xyxy.tolist()
            ):
                if int(class_id) == 0:  # stamp class
                    x1, y1, x2, y2 = box
                    
                    # Calculate coverage
                    stamp_width = x2 - x1
                    stamp_height = y2 - y1
                    stamp_area = stamp_width * stamp_height
                    total_area = img_width * img_height
                    coverage_percent = (stamp_area / total_area) * 100
                    
                    # Position analysis
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    relative_x = center_x / img_width
                    relative_y = center_y / img_height
                    
                    print(f"Stamp Region Analysis:")
                    print(f"  Size: {int(stamp_width)} x {int(stamp_height)} pixels")
                    print(f"  Coverage: {coverage_percent:.1f}% of image")
                    print(f"  Position: ({relative_x:.1%} from left, {relative_y:.1%} from top)")
                    
                    # Typical stamp location checks
                    print(f"\nLocation Check:")
                    if relative_y < 0.3:
                        print("  ⚠️  Stamp detected in TOP third (unusual)")
                    elif relative_y > 0.7:
                        print("  ✓ Stamp detected in BOTTOM third (typical)")
                    else:
                        print("  ⚠️  Stamp detected in MIDDLE (verify)")
                    
                    if relative_x < 0.3:
                        print("  ⚠️  Stamp on LEFT side (unusual)")
                    elif relative_x > 0.7:
                        print("  ✓ Stamp on RIGHT side (typical)")
                    else:
                        print("  ⚠️  Stamp in CENTER (verify)")
                    
                    print(f"\nSize Check:")
                    if coverage_percent < 5:
                        print("  ⚠️  Stamp region is VERY SMALL (may miss data)")
                    elif coverage_percent > 30:
                        print("  ⚠️  Stamp region is VERY LARGE (may include noise)")
                    else:
                        print("  ✓ Stamp region size is reasonable")


# Run diagnostics on your 3 sample files
if __name__ == "__main__":
    sample_files = [
        "sample1.jpg",  # Replace with your actual file paths
        "sample2.jpg",
        "sample3.jpg",
    ]
    
    print("\n" + "="*60)
    print("BOL REGION DETECTION DIAGNOSTIC TOOL")
    print("="*60)
    
    for i, sample_file in enumerate(sample_files, 1):
        print(f"\n\n{'#'*60}")
        print(f"# SAMPLE {i}: {sample_file}")
        print(f"{'#'*60}\n")
        
        try:
            # Run detection visualization
            detections = visualize_bol_detections(
                sample_file, 
                save_path=f"debug_sample_{i}.jpg"
            )
            
            # Check stamp coverage
            if detections["stamp"] > 0:
                check_stamp_coverage(sample_file)
            
        except Exception as e:
            print(f"❌ Error processing {sample_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")
    print("\nCheck the generated debug_sample_*.jpg files to see what was detected")
    print("\nNext steps:")
    print("1. If stamp regions are missing → Retrain YOLO model")
    print("2. If stamp regions are too small → Adjust confidence threshold")
    print("3. If stamp regions don't cover full stamp area → Retrain with better annotations")