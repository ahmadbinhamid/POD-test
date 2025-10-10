from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import cv2
from PIL import Image, ImageOps
import numpy as np
import re
from typing import Union, List
import base64
from pathlib import Path
import datetime
import pickle
from io import BytesIO
import uuid


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Month name to number mapping
MONTHS = {
    "Jan": "1", "Feb": "2", "Mar": "3", "Apr": "4", "May": "5", "Jun": "6",
    "Jul": "7", "Aug": "8", "Sep": "9", "Oct": "10", "Nov": "11", "Dec": "12"
}

def build_transform(input_size):
    """
    Build image transformation pipeline for model input
    Args:
        input_size: Target size for image transformation
    Returns:
        Composed transformation pipeline
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find the closest matching aspect ratio from target ratios
    Args:
        aspect_ratio: Current image aspect ratio
        target_ratios: List of possible target ratios
        width: Image width
        height: Image height
        image_size: Target image size
    Returns:
        Best matching ratio tuple
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess image for model input by splitting into blocks
    Args:
        image: Input image
        min_num: Minimum number of blocks
        max_num: Maximum number of blocks
        image_size: Target size for each block
        use_thumbnail: Whether to include thumbnail
    Returns:
        List of processed image blocks
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate possible target ratios
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best ratio and calculate dimensions
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Process image into blocks
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    # Add thumbnail if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        
    return processed_images

def straighten_images(images: Union[List[str], List[np.ndarray], List[Image.Image]]) -> List[Image.Image]:
    """
    Straighten images based on text orientation detection
    Args:
        images: List of images in various formats
    Returns:
        List of straightened PIL Images
    """
    straightened_images = []

    for img in images:
        # Convert input to numpy array
        if isinstance(img, str):
            image = cv2.imread(img)
        elif isinstance(img, Image.Image):
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            image = img.copy()
        else:
            raise ValueError("Unsupported image type")

        # Scale image for better processing
        scale_percent = 300
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

        try:
            # Detect and correct orientation
            cv2.imwrite(f"resized_image_{datetime.datetime.now()}.jpg", resized_image)
            osd = pytesseract.image_to_osd(resized_image)
            rotation_angle = int(re.search(r"Rotate: (\d+)", osd).group(1))

            if rotation_angle != 0:
                # Calculate rotation parameters
                (h, w) = resized_image.shape[:2]
                center = (w // 2, h // 2)
                angle = 90 if rotation_angle == 90 else -90 if rotation_angle == 270 else rotation_angle
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Adjust dimensions for rotation
                cos = np.abs(matrix[0, 0])
                sin = np.abs(matrix[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                matrix[0, 2] += (new_w / 2) - center[0]
                matrix[1, 2] += (new_h / 2) - center[1]

                # Perform rotation
                rotated_image = cv2.warpAffine(
                    resized_image, matrix, (new_w, new_h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
                straightened_images.append(Image.fromarray(rotated_image))
            else:
                straightened_images.append(Image.fromarray(resized_image))

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            straightened_images.append(resized_image)

    return straightened_images

def save_image_to_tmp_dir(image: Image.Image, file_extension: str = "png") -> str:
    """
    Save PIL image to temporary directory
    Args:
        image: PIL Image to save
        file_extension: File extension for saved image
    Returns:
        Path to saved image
    """
    # image_path = Path("./tmp") / f"image.{file_extension}"
    # image.save(image_path)
    # return str(image_path)

    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    image_path = Path("./tmp") / unique_filename
    image.save(image_path)
    return str(image_path)


def convert_image_to_base64(image: Image.Image, file_extension: str = "png") -> str:
    """
    Convert PIL image to base64 string
    Args:
        image: PIL Image to convert
        file_extension: Image format extension
    Returns:
        Base64 encoded image string with data URI prefix
    """
    buffer = BytesIO()
    image.save(buffer, format=file_extension.upper())
    buffer.seek(0)
    encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
    return "data:image/png;base64," + encoded_string

def make_image_square(img: Image.Image):
    """
    Convert image to square by adding white padding
    Args:
        img: Input PIL Image
    Returns:
        Square PIL Image with white padding
    """
    width, height = img.size
    square_size = max(width, height)
    square_img = Image.new("RGB", (square_size, square_size), (255, 255, 255))
    paste_x = (square_size - width) // 2
    paste_y = (square_size - height) // 2
    square_img.paste(img, (paste_x, paste_y))
    return square_img

def base64topil(img_str):
    """
    Convert base64 image string to PIL Image
    Args:
        img_str: Base64 encoded image string
    Returns:
        PIL Image or None if conversion fails
    """
    try:
        MAX_PIXELS = Image.MAX_IMAGE_PIXELS or 178956970
        img = Image.open(BytesIO(base64.b64decode(img_str.split(",")[1])))
        return resize_large_image(img, MAX_PIXELS)

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def adjust_date(month, day, year=None):
    """
    Adjust invalid date values to nearest valid date
    Args:
        month: Month number
        day: Day number
        year: Optional year number
    Returns:
        Tuple of adjusted (month, day, year)
    """
    current_year = datetime.datetime.now().year
    year_to_use = year if year is not None else current_year
    month = min(max(month, 1), 12)
    day = max(day, 1)
    last_day = (datetime.date(year_to_use, month, 1) + datetime.timedelta(days=31)).replace(day=1) - datetime.timedelta(days=1)
    return month, min(day, last_day.day), year_to_use

def preprocess_date(date_str):
    """
    Normalize date string to MM/DD/YY format
    Args:
        date_str: Input date string in various formats
    Returns:
        Normalized date string or error message
    """
    try:
        # Clean input
        date_str = re.sub(r'\s+', ' ', date_str).strip()
        date_str = re.sub(r'[.,]', '/', date_str)

        # Handle text month format
        match = re.match(r'^([A-Za-z]{3,})[ /]*(\d{1,2})[ /]*(\d{2,4})?$', date_str)
        if match:
            month, day, year = match.groups()
            month = MONTHS.get(month[:3].title(), month)
            day = int(day)
            year = int(year) if year else None
            month, day, year = adjust_date(int(month), day, year)
            return f"{month}/{day}" if year is None else f"{month}/{day}/{str(year)[-2:]}"

        # Handle numeric formats
        date_str = date_str.replace('-', '/')
        parts = date_str.split('/')

        if len(parts) == 2:
            first_num = int(parts[0])
            if first_num > 12:
                return f"{first_num}/{parts[1]}"

        if len(parts) == 3:
            month, day, year = map(int, parts)
            month, day, year = adjust_date(month, day, year)
            return f"{month}/{day}/{str(year)[-2:]}"

        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            num1, num2 = map(int, parts)
            if num1 <= 12:
                month, day, _ = adjust_date(num1, num2, None)
                return f"{month}/{day}"
            return f"{num1}/{num2}"

        return date_str

    except Exception:
        from traceback import print_exc
        print(print_exc())
        return "Error Processing Date"

def validate_customer_number(customer_number, all_numbers=None):
    """
    Validate and extract customer number based on rules
    Args:
        customer_number: Input number to validate
        all_numbers: Optional list of all numbers for length comparison
    Returns:
        Validated customer number or None if invalid
    """
    if isinstance(customer_number, int):
        customer_number = str(customer_number)

    if isinstance(customer_number, str):
        if '/' in customer_number:
            parts = customer_number.split('/')
            if len(parts) == 2 and parts[1].isdigit():
                customer_number = parts[1]
            else:
                return None
        elif '-' in customer_number:
            return None
        elif not customer_number.isdigit():
            return None
    else:
        return None

    if all_numbers:
        str_numbers = [str(num) for num in all_numbers if num is not None]
        max_length = max(len(num) for num in str_numbers)
        return customer_number if len(customer_number) == max_length else None

    return customer_number

def process_customer_numbers(input_data):
    """
    Process single or multiple customer numbers
    Args:
        input_data: Single number or list of numbers
    Returns:
        List of valid customer numbers
    """
    if isinstance(input_data, list):
        valid_numbers = [validate_customer_number(item, input_data) for item in input_data]
        return [num for num in valid_numbers if num is not None]
    else:
        valid_number = validate_customer_number(input_data, [input_data])
        return [valid_number] if valid_number is not None else []

def resize_large_image(img: Image.Image, max_pixels: int = 178956970) -> Image.Image:
    """
    Resize image if it exceeds maximum pixel limit
    Args:
        img: Input PIL Image
        max_pixels: Maximum allowed pixels
    Returns:
        Resized PIL Image if necessary
    """
    width, height = img.size
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img