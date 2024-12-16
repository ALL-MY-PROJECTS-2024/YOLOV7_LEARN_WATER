import os
import json

def validate_json_files(folder_path):
    # 폴더 및 하위 폴더 내의 모든 파일 검사
    print(f"Scanning folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        print(f"Current folder: {root}")
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # JSON 파일인지 확인
            if file_name.endswith('.json'):
                try:
                    # 파일 열기 및 JSON 파싱 시도
                    with open(file_path, 'r', encoding='utf-8') as file:
                        json.load(file)
                    print(f"✅ Valid JSON: {file_path}")
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON: {file_path}")
                    print(f"   Error: {e}")
                except Exception as e:
                    print(f"❌ Error reading file: {file_path}")
                    print(f"   Error: {e}")
            else:
                print(f"ℹ️ Skipping non-JSON file: {file_path}")

# 실행 예시
folder_path = "./CHECK"  # JSON 파일이 있는 폴더 경로
validate_json_files(folder_path)
