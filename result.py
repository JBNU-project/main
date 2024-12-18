import subprocess
import re  # 정규표현식 모듈을 사용
import openai
import cv2
import os
import requests
import uuid
import time
import json
import numpy as np

openai.api_key = "OPEN_AI_APIKEY" # api 키 입력
api_url = 'NAVER_CLOVA_OCR_URL'
secret_key = 'NAVER_CLOVA_OCR_KEY'

def merge_boxes_with_text(boxes_with_text, merge_margin=50):
    """
    바운딩 박스와 텍스트를 병합.

    Args:
        boxes_with_text (list of dict): [{'bbox': [x1, y1, x2, y2], 'text': "example"}, ...]
        merge_margin (int): 병합 기준이 되는 여유 공간 (픽셀 단위).

    Returns:
        list of dict: 병합된 바운딩 박스와 텍스트 [{'bbox': [x1, y1, x2, y2], 'text': "merged text"}, ...]
    """
    merged_boxes_with_text = []
    while boxes_with_text:
        # 현재 박스를 가져오고 리스트에서 제거
        curr = boxes_with_text.pop(0)
        x1, y1, x2, y2 = curr['bbox']
        merged_text = curr['text']

        # 병합 기준을 포함한 확장 박스
        expand_x1, expand_y1 = x1 - merge_margin, y1 - merge_margin
        expand_x2, expand_y2 = x2 + merge_margin, y2 + merge_margin

        # 현재 박스와 겹치는 박스를 찾음
        overlaps = []
        for box in boxes_with_text:
            bx1, by1, bx2, by2 = box['bbox']
            if not (bx2 < expand_x1 or bx1 > expand_x2 or by2 < expand_y1 or by1 > expand_y2):
                overlaps.append(box)

        # 겹치는 박스를 모두 병합
        for box in overlaps:
            boxes_with_text.remove(box)
            bx1, by1, bx2, by2 = box['bbox']
            x1 = min(x1, bx1)
            y1 = min(y1, by1)
            x2 = max(x2, bx2)
            y2 = max(y2, by2)
            merged_text += " " + box['text']  # 텍스트 병합

        # 병합된 박스를 결과 리스트에 추가
        merged_boxes_with_text.append({'bbox': [x1, y1, x2, y2], 'text': merged_text.strip()})

    return merged_boxes_with_text


def save_combined_mask(image_shape, mmdetection_bboxes, naver_bboxes, mask_output_path):
    """
    MMDetection과 Naver OCR의 바운딩 박스를 합쳐서 하나의 마스크 생성 및 저장.
    """
    # 빈 검은색 이미지를 생성 (마스크 초기화)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # MMDetection 바운딩 박스 영역을 흰색(255)으로 채움
    for bbox in mmdetection_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255), -1)

    # Naver OCR 바운딩 박스 영역을 흰색(255)으로 채움
    for bbox in naver_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255), -1)

    # 마스크 저장
    cv2.imwrite(mask_output_path, mask)
    print(f"Combined mask saved at {mask_output_path}")
    return mask_output_path

def run_naver_ocr(image_file):
    # 네이버 OCR 요청 설정
    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [('file', open(image_file, 'rb'))]
    headers = {'X-OCR-SECRET': secret_key}

    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)

    # 텍스트와 바운딩 박스 좌표 추출
    ocr_results = []
    try:
        response_data = response.json()
        for field in response_data['images'][0]['fields']:
            text = field['inferText']  # 텍스트
            bounding_box = field['boundingPoly']['vertices']  # 바운딩 박스 좌표

            # 왼쪽 위와 오른쪽 아래 좌표 계산
            min_x = min(point['x'] for point in bounding_box)
            min_y = min(point['y'] for point in bounding_box)
            max_x = max(point['x'] for point in bounding_box)
            max_y = max(point['y'] for point in bounding_box)

            # OCR 결과 저장
            ocr_results.append({
                "text": text,
                "bbox": (min_x, min_y, max_x, max_y)
            })
    except KeyError as e:
        print(f"KeyError: {e}. Check the API response format.")
    except Exception as e:
        print(f"Error: {e}")
    return ocr_results


def extract_all_boxes(stdout):
    # 정규 표현식을 사용하여 "Box x:" 뒤에 오는 숫자 4개를 모두 추출
    boxes = []
    matches = re.findall(r"Box \d+:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", stdout)
    if matches:
        # 모든 매치를 실수(float) 배열로 변환
        boxes.append([[float(value) for value in match] for match in matches])
        return boxes
    else:
        print("No bounding boxes found in the output.")
        return []

def run_inference():
    output_list = []  # 결과를 저장할 리스트 초기화
    image = '16.png'

    # 첫 번째 명령어 (mmdetection 실행)
    command_1 = [
        'python', 'mmdetection/demo/image_demo.py', 
        f'image_folder/{image}', 
        'mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py', 
        '--weights', 'mmdetection/Text_work_dir/best_coco_bbox_mAP_epoch_10.pth', 
        '--texts', 'Text'
    ]

    # 명령어 실행
    result_1 = subprocess.run(command_1, capture_output=True, text=True)

    mmdetection_bboxes = []
    if result_1.returncode == 0:
        print("MMDetection completed successfully!")
        print("Output:\n", result_1.stdout)
        mmdetection_bboxes = extract_all_boxes(result_1.stdout)[0]  # 바운딩 박스 추출
    else:
        print("Error occurred during MMDetection.")
        print("Error:\n", result_1.stderr)
        return
    
        # 두 번째 명령어 (iopaint 실행)
    command_2 = [
        'iopaint', 'run', '--model=lama', '--device=cuda',
        '--image=image_folder', '--mask=mask_folder', '--output=erased_dir'
    ]

    # 두 번째 명령어 실행
    result_2 = subprocess.run(command_2, capture_output=True, text=True)

    # 네이버 OCR 실행
    print(f'mmdetection_bbox:{mmdetection_bboxes}')
    print("Running Naver OCR...")
    naver_results = run_naver_ocr(f'erased_dir/{image}')
    print("Naver OCR Results:", naver_results)

    # 바운딩 박스와 텍스트 병합
    print("Merging Naver OCR bounding boxes and texts...")
    merged_naver_results = merge_boxes_with_text(naver_results)
    print("Merged Naver OCR Results:", merged_naver_results)

    # Naver OCR 바운딩 박스 추출
    merged_naver_bboxes = [list(result['bbox']) for result in merged_naver_results]
    # Step 4: MMDetection과 Naver OCR 바운딩 박스를 하나의 마스크로 저장
    image_path = f'image_folder/{image}'
    image_n = cv2.imread(image_path)
    combined_mask_path = f'mask_folder/{image}'
    save_combined_mask(image_n.shape, mmdetection_bboxes, merged_naver_bboxes, combined_mask_path)
    print(merged_naver_bboxes)
    # ChatGPT 번역 요청
    n_translated_texts = []
    for result in merged_naver_results:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Translate the following Korean text to English. If translation is not possible, return the Korean text in English pronunciation (romanized)."},
                {"role": "user", "content": result['text']}
            ]
        )
        n_translated_text = response['choices'][0]['message']['content']  # 번역된 텍스트
        n_translated_texts.append(n_translated_text)
        result['translated_text'] = n_translated_text  # 번역된 텍스트 저장

    # 이미지에 텍스트와 바운딩 박스 그리기
    #draw_boxes_on_image_naver(f'erased_dir/{image}', naver_results)


    # 두 번째 명령어 (iopaint 실행)
    command_2 = [
        'iopaint', 'run', '--model=lama', '--device=cuda',
        '--image=image_folder', '--mask=mask_folder', '--output=erased_dir'
    ]

    
    # 두 번째 명령어 실행
    result_2 = subprocess.run(command_2, capture_output=True, text=True)

    # 두 번째 명령어 결과 출력 및 리스트에 저장
    if result_2.returncode == 0:
        print("iopaint completed successfully!")
        print("Output:\n", result_2.stdout)
        output_list.append(result_2.stdout)  # 표준 출력을 리스트에 저장
    else:
        print("Error occurred during iopaint execution.")
        print("Error:\n", result_2.stderr)
        output_list.append(result_2.stderr)  # 표준 에러를 리스트에 저장

    # 세 번째 명령어 (OCR 실행)
    command_3 = [
        'python3', 'PaddleOCR/tools/infer_rec.py', 
        '-c', 'PaddleOCR/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml', 
        '-o', 'Global.pretrained_model= PaddleOCR/weights/best_accuracy', 
        'Global.infer_img=cropping_images'
    ]
    
    print("image:", image)
    # 명령어 실행
    result_3 = subprocess.run(command_3, capture_output=True, text=True)

    # 세 번째 명령어 결과 출력 및 리스트에 저장
    if result_3.returncode == 0:
        print("OCR inference completed successfully!")
        print("Output:\n", result_3.stdout)

       # 정규 표현식을 사용하여 result: [결과]에서 모든 결과 추출
        matches = re.findall(r"result:\s*(.+)", result_3.stdout)
        print("matches : ", matches)
        if matches:
            recognized_texts = matches  # 모든 결과를 리스트로 저장
            output_list.extend(recognized_texts)  # 인식된 텍스트를 리스트에 추가
            print("Recognized texts:", recognized_texts)  # 추출된 텍스트 출력

            translated_texts = []
            for recognized_text in recognized_texts:
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # 모델 설정
                    messages=[
                        {"role": "system", "content": "Translate the following Korean onomatopoeia to English. If translation is not possible, return the Korean text in English pronunciation (romanized)."},
                        {"role": "user", "content": recognized_text}  # 예시로 사용할 의성어
                    ]
                )
                translated_text = response['choices'][0]['message']['content']  # 번역된 텍스트
                translated_texts.append(translated_text)

            print("번역된 언어들:", translated_texts)  # 출력된 번역 결과
            print("Position of bbox:", mmdetection_bboxes[0])
        else:
            print("No result found.")
    else:
        print("Error occurred during OCR inference.")
        print("Error:\n", result_3.stderr)
        output_list.append(result_3.stderr)  # 표준 에러를 리스트에 저장
    print("image:", image)
    draw_boxes_on_image(image, mmdetection_bboxes, translated_texts)
    draw_boxes_on_image(image, merged_naver_bboxes, n_translated_texts)
    return result_3  # 모든 명령어 결과를 리스트로 반환

def draw_boxes_on_image(image_name, boxes, translated_texts, erase_dir="erased_dir"):
    # 확장자를 제외한 파일 이름 추출
    base_name = os.path.splitext(image_name)[0]

    # erase_dir에서 이미지 불러오기
    input_image_path = os.path.join(erase_dir, f"{base_name}.png")
    if not os.path.exists(input_image_path):
        print(f"Image {input_image_path} does not exist in {erase_dir}.")
        return

    # OpenCV로 이미지 읽기
    image = cv2.imread(input_image_path)
    flat_boxes = boxes  # 가장 바깥 리스트 제거

    # 박스와 텍스트 매칭
    for i, box in enumerate(flat_boxes):
        if len(box) == 4:  # 박스가 유효한 좌표인지 확인
            x1, y1, x2, y2 = map(int, box)  # 좌표를 정수로 변환
            
            # 번역된 텍스트 선택 (리스트 범위를 초과하지 않도록 확인)
            text = translated_texts[i] if i < len(translated_texts) else "N/A"

            # 박스 크기 계산
            box_width = x2 - x1
            box_height = y2 - y1

            # 폰트 크기와 줄 간격 초기화
            font_scale = 1.0
            font_thickness = 1
            line_spacing = 5  # 줄 간격

            while True:
                # 텍스트 줄바꿈 (글자 단위)
                lines = []
                current_line = ""

                for char in text:
                    test_line = current_line + char
                    text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_TRIPLEX, font_scale, font_thickness)[0]
                    if text_size[0] <= box_width * 0.9:  # 박스 너비 안에 들어가면
                        current_line = test_line
                    else:  # 박스를 초과하면
                        lines.append(current_line)
                        current_line = char

                if current_line:
                    lines.append(current_line)

                # 전체 텍스트 높이 계산
                total_text_height = len(lines) * (cv2.getTextSize("A", cv2.FONT_HERSHEY_TRIPLEX, font_scale, font_thickness)[0][1] + line_spacing)

                # 텍스트가 박스 높이 안에 들어가면 루프 종료
                if total_text_height <= box_height * 0.9:
                    break

                # 폰트 크기를 줄임
                font_scale -= 0.1
                if font_scale < 0.1:  # 폰트 크기가 너무 작아지면 중단
                    font_scale = 0.1
                    break

            # 텍스트를 박스 내에 그리기
            y_offset = y1 + (box_height - total_text_height) // 2  # 텍스트 상단 시작 위치
            for line in lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_TRIPLEX, font_scale, font_thickness)[0]
                text_x = x1 + (box_width - text_size[0]) // 2  # 중앙 정렬
                text_y = y_offset + text_size[1]

                # 검은색 테두리 추가
                cv2.putText(
                    image, line, (text_x, text_y), 
                    cv2.FONT_HERSHEY_TRIPLEX, 
                    font_scale, 
                    (0, 0, 0), 
                    thickness=3, 
                    lineType=cv2.LINE_AA
                )

                # 흰색 텍스트 추가
                cv2.putText(
                    image, line, (text_x, text_y), 
                    cv2.FONT_HERSHEY_TRIPLEX, 
                    font_scale, 
                    (255, 255, 255), 
                    thickness=font_thickness, 
                    lineType=cv2.LINE_AA
                )

                # 다음 줄로 이동
                y_offset += text_size[1] + line_spacing

    # 수정된 이미지 저장
    output_image_path = os.path.join(erase_dir, f"{base_name}.png")
    cv2.imwrite(output_image_path, image)
    print(f"Image with boxes saved at {output_image_path}")

if __name__ == "__main__":
    results = run_inference()
