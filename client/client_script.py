import numpy as np
import time
import tritonclient.grpc as grpcclient

def inference(triton_server_url: str, model_name: str, model_version:int, num_persons:int, interval: float):
    # --- 2. Triton 클라이언트 생성 ---
    try:
        triton_client = grpcclient.InferenceServerClient(url=triton_server_url)
    except Exception as e:
        print(f"클라이언트 생성 실패: {e}")
        exit(1)

    # --- 3. 모델에 보낼 입력 데이터 준비 ---
    input_data = np.random.rand(num_persons, 1, 100, 20, 3).astype(np.float32)

    # --- 4. 입/출력 텐서 설정 ---
    inputs = []
    inputs.append(grpcclient.InferInput('input', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))

    # --- 5. Triton 서버에 1초마다 추론 요청 보내기 ---
    print(f"'{model_name}' 모델에 {interval}초마다 추론을 요청합니다...")
    print("중지하려면 Ctrl+C를 누르세요.\n")

    # --- 4. 입/출력 텐서 설정 ---
    inputs = []
    inputs.append(grpcclient.InferInput('input', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))

    # --- 5. Triton 서버에 1초마다 추론 요청 보내기 ---
    print(f"'{model_name}' 모델에 {interval}초마다 추론을 요청합니다...")
    print("중지하려면 Ctrl+C를 누르세요.\n")

    try:
        request_count = 0
        while True:
            start_time = time.time()
            request_count += 1
            
            print(f"[요청 #{request_count}] 추론 요청 중...")
            
            # 추론 요청
            results = triton_client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs
            )
            
            # 결과 확인
            output_data = results.as_numpy('output')
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms 단위
            
            print(f"✅ 추론 완료! (소요시간: {inference_time:.2f}ms)")
            print(f"결과 형태: {output_data.shape}")

            for result in output_data:
                print(f"예측 결과: {result}")

            print("-" * 40)
            
            # 대기
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n프로그램이 중지되었습니다. 총 {request_count}번의 요청을 처리했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    TRITON_SERVER_URL = "localhost:8001"  # gRPC 포트는 보통 8001
    MODEL_NAME = "protogcn_selfharm"
    MODEL_VERSION = "1"
    NUM_PERSONS = 1
    INTERVAL = 0.1

    print("Triton 클라이언트 테스트 스크립트가 실행되었습니다.")
    print(f"서버 URL: {TRITON_SERVER_URL}, 모델 이름: {MODEL_NAME}, 버전: {MODEL_VERSION}, 사람 수: {NUM_PERSONS}, 간격: {INTERVAL}초\n")
    inference(TRITON_SERVER_URL, MODEL_NAME, MODEL_VERSION, NUM_PERSONS, INTERVAL)