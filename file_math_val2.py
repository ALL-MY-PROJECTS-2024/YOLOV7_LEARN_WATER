import json
import os

def normalize_coordinates(size, coordinates):
    """
    Normalize segmentation coordinates to YOLO format (relative to image size).
    """
    width, height = size
    normalized = []
    for x, y in coordinates:
        normalized.append(x / width)
        normalized.append(y / height)
    return normalized

def convert_custom_coco_to_yolov7_segmentation(coco_json_path, output_dir):
    """
    Convert custom COCO-style JSON format to YOLOv7 segmentation format.
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    
    # Extract image information
    image_info = coco_data["IMAGE"]
    img_width = int(image_info["WIDTH"])
    img_height = int(image_info["HEIGHT"])
    img_name = os.path.splitext(image_info["FILE_NAME"])[0]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # YOLOv7 annotation file path
    output_file_path = os.path.join(output_dir, f"{img_name}.txt")

    with open(output_file_path, "w") as f:
        for annotation in coco_data["ANNOTATIONS"]:
            category_id = int(coco_data["CATEGORIES"]["CATEGORY"]) - 1  # Convert to 0-based index
            coordinates = annotation["COORDINATE"]

            # Normalize segmentation coordinates
            normalized_coords = normalize_coordinates(
                (img_width, img_height), coordinates
            )
            normalized_coords_str = " ".join(map(str, normalized_coords))

            # Write to file
            f.write(f"{category_id} {normalized_coords_str}\n")

def process_all_json_files(input_dir, output_dir):
    """
    Process all JSON files in a directory and convert them to YOLOv7 segmentation format.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_path = os.path.join(root, file)
                print(f"Processing: {input_path}")
                convert_custom_coco_to_yolov7_segmentation(input_path, output_dir)

if __name__ == "__main__":
    input_dir = "./2_dataset_val_math/labels"  # JSON 파일들이 있는 최상위 폴더
    output_dir = "./data/math/labels/val"  # 변환된 YOLO 어노테이션을 저장할 단일 폴더

    # Process all JSON files
    process_all_json_files(input_dir, output_dir)
