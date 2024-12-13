import json
import os


def convert_coordinates(size, points):
    """
    Normalize segmentation points to YOLO format (relative to image size).
    """
    width, height = size
    normalized_points = []
    for x, y in points:
        normalized_points.append(x / width)
        normalized_points.append(y / height)
    return normalized_points


def convert_coco_to_yolov7_segment(coco_json_path, output_dir):
    """
    Convert COCO JSON format to YOLOv7 segmentation format.
    """
    # Load COCO JSON file
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get image information
    images = {img["id"]: img for img in coco_data["images"]}

    # Process annotations
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        image = images[image_id]
        img_width, img_height = image["width"], image["height"]
        image_name = os.path.splitext(image["file_name"])[0]

        # Create YOLOv7 segmentation file (all in one folder)
        output_file_path = os.path.join(output_dir, f"{image_name}.txt")
        with open(output_file_path, "a") as f:
            category_id = annotation["category_id"]
            segmentation = annotation["segmentation"]

            # Convert segmentation points
            for segment in segmentation:
                normalized_segment = convert_coordinates((img_width, img_height), zip(segment[::2], segment[1::2]))
                normalized_segment_str = " ".join(map(str, normalized_segment))
                f.write(f"{category_id - 1} {normalized_segment_str}\n")


def process_all_jsons(input_dir, output_dir):
    """
    Process all JSON files in the input directory and its subdirectories.
    Convert them to YOLOv7 segmentation format, storing all outputs in one folder.
    """
    # Traverse input directory recursively
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_json_path = os.path.join(root, file)
                print(f"Processing file: {input_json_path}")

                # Convert JSON file to YOLO format
                convert_coco_to_yolov7_segment(input_json_path, output_dir)


def main():
    input_dir = "./dataset_train/labels"  # JSON 파일들이 있는 최상위 폴더
    output_dir = "./data/water/labels/train"  # 변환된 YOLO 어노테이션을 저장할 단일 폴더

    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    # Process all JSON files
    process_all_jsons(input_dir, output_dir)


if __name__ == "__main__":
    main()
