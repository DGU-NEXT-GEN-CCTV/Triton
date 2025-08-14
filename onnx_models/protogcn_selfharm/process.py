import torch
import numpy as np

def convert_relative_coord(keypoints: list):
    """
    전역 좌표계 키포인트 데이터를 상대 좌표계로 변환하는 함수
    - 입력: keypoints (list): 각 키포인트의 좌표가 포함된 리스트
    - 출력: processed_keypoints (list): 상대 좌표계로 변환된 키포인트 리스트
    - 기준점: 골반 중심점 (COCO 17 keypoints: 11 = right hip, 12 = left hip)
    - 기준 길이: 골반 중심점과 양쪽 엉덩이 사이의 거리
    - 변환: 각 키포인트의 좌표를 기준점으로부터의 상대 좌표로 변환
    - 결과: 각 키포인트의 x, y 좌표를 기준점으로부터 상대적으로 표현한 리스트
    - 예시: [[x1, y1], [x2, y2], ..., [x17, y17]] -> [[x1', y1'], [x2', y2'], ..., [x17', y17']]
    """
    processed_keypoints = []
    
    for k in keypoints:
        base_j = ((k[11][0] + k[12][0]) / 2, (k[11][1] + k[12][1]) / 2) # 기준점 (골반 중심점, COCO 17 keypoints: 11 = right hip, 12 = left hip)
        base_l = np.linalg.norm(np.array(k[11]) - np.array(k[12])) # 기준 길이 (골반 중심점과 양쪽 엉덩이 사이의 거리)
        processed_k = []
        for j in k:
            processed_k.append([(j[0] - base_j[0]) / base_l, (j[1] - base_j[1]) / base_l], j[2]) # 상대 좌표로 변환된 키포인트에 대한 일반화
        processed_keypoints.append(processed_k)
        
    processed_keypoints = np.array(processed_keypoints, dtype=np.float32)
        
    return processed_keypoints

def pre(data: np.ndarray, coord_type: str = 'global') -> np.ndarray:
    
    """
    데이터 전처리 함수
    - 17개의 키포인트로 구성된 2D 포즈 데이터를 입력으로 받아 3개의 가상 키포인트를 추가하여 20개의 키포인트로 구성된 2D 포즈 데이터로 변환
    - 가상의 키포인트 1: 양쪽 어깨 중간점, 가상의 키포인트 2: 양쪽 엉덩이 중간점, 가상의 키포인트 3: 가상의 키포인트 1과 2의 중간점
    
    Args:
        data (np.ndarray): 입력 데이터, shape: (D, N, T, K, C)
            - D: 배치 크기
            - N: 사람 수
            - T: 시퀀스 길이(프레임 단위)
            - K: 키포인트 수 (17개)
            - C: 채널 (x, y, confidence)

    Returns:
        np.ndarray: 변환된 데이터, shape: (D, N, T, 20, C)
            - 20개의 키포인트로 구성된 2D 포즈 데이터
    """
    D, N, T, K, C = data.shape
    
    if K != 17 or C != 3:
        raise ValueError("Input data must have shape (D, N, T, 17, 3)")
    
    if coord_type == 'global':
        # 전역 좌표계에서 상대 좌표계로 변환
        data = convert_relative_coord(data.reshape(-1, K, C)).reshape(D, N, T, K, C)

    # 가상의 키포인트 1: 양쪽 어깨 중간점
    neck = (data[..., 5, :2] + data[..., 2, :2]) / 2
    # 가상의 키포인트 2: 양쪽 엉덩이 중간점
    pelvis = (data[..., 11, :2] + data[..., 8, :2]) / 2
    # 가상의 키포인트 3: 가상의 키포인트 1과 2의 중간점
    body_mid = (neck + pelvis) / 2

    # 20개의 키포인트로 구성된 2D 포즈 데이터로 변환
    new_data = np.zeros((D, N, T, 20, C))
    new_data[..., :17, :] = data
    new_data[..., 17, :2] = neck
    new_data[..., 18, :2] = pelvis
    new_data[..., 19, :2] = body_mid
    
    new_data = new_data.astype(np.float32)  # Ensure the output is in float32 format

    return new_data

def post(data: torch.tensor) -> list:
    """
    데이터 후처리 함수
    - 20개의 키포인트로 구성된 2D 포즈 데이터를 입력으로 받아 17개의 키포인트로 구성된 2D 포즈 데이터로 변환

    Args:
        data (torch.tensor): 입력 데이터, shape: (D, P)
            - D: 배치 크기
            - P: 라벨별 확률값 (기본값 2)

    Returns:
        bool: 변환된 데이터가 selfharm인지 여부
        list: 라벨 리스트
    """
    data = data.cpu().numpy()  # 텐서를 NumPy 배열로 변환
    flag_detail = [d[0] < d[1] for d in data]  # 입력 시퀀스의 포즈별 selfharm 여부 (첫 번째 확률 값: normal, 두 번째 확률 값: selfharm)
    flag = any(flag_detail) # 입력 시퀀스에서 selfharm 여부

    return flag, flag_detail