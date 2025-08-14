import numpy as np
from process import pre, post

def inference():
    # Yolo 사람 감지 코드...
    
    # 포즈 데이터 추론 코드...
    keypoints = np.random.rand(1, 17, 2).astype(np.float32)  # 예시 포즈 데이터
    confidence = np.random.rand(1, 17, 1).astype(np.float32)  # 예시 신뢰도 데이터
    input_data = np.concatenate((keypoints, confidence), axis=-1)  # (1, 17, 3) 형태로 변환
    
    # BotSort 코드 ...
    tracked_sequence = np.random.rand(1, 1, 100, 17, 3).astype(np.float32) # 트래커를 통해 최근 100 프레임의 포즈 데이터 추출

    """
    입력 데이터 예시: 사람 수(1~), 분석 단위(1), 시퀀스 길이(100), 키포인트 수(17), 채널(3)
    - 시퀀스 길이는 100으로 설정, 이 데이터는 BotSort 트래커를 수정해서 얻어온 데이터(TODO: 통합 작업에서 진행)
    - 한 사람에 대한 키포인트 데이터는 17개의 키포인트로 구성되며, 각 키포인트는 (x, y, confidence) 형태로 표현
    """
    input_data = tracked_sequence[:, :, :100, :, :]  # (1, 1, 100, 17, 3) 형태로 변환

    """
    모델 입력에 맞게 키포인트 데이터를 전처리
    - (-1, 1, 100, 17, 3) -> (-1, 1, 100, 20, 3)
    - 17개의 키포인트에 3개의 가상 키포인트를 추가
    - 만약 전역 좌표계를 사용한다면, 상대 좌표계로 변환
      - coord_type: 'global' (전역 좌표계), 'local' (상대 좌표계)
    """
    preprocessed_input = pre(input_data, coord_type='global')

    # Triton 추론코드 ...
    # 예시: triton_client.infer(model_name, inputs=[preprocessed_input])
    output = np.random.rand(1, 2).astype(np.float32)  # 예시 출력 (배치 크기 1, 클래스 수 2)

    """
    추론 결과 후처리
    - (-1, 2) 형태로 제공되는 모델 출력을 이상행동 발생 여부로 변환
    - flag: 입력 시퀀스 내에서 selfharm이 발생했는지 여부 (True/False)
    - flag_detail: 입력 시퀀스 내 사람별 selfharm 여부 리스트
    
    """
    flag, flag_detail = post(output)
    
    if flag:
        print("Self-harm detected in the sequence.")
        for i, detail in enumerate(flag_detail):
            if detail:
                print(f"Person {i+1} is likely to be self-harming.")
    else:
        print("No self-harm detected in the sequence.")

if __name__ == "__main__":
    inference()