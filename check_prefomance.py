import os
import time
import json
import psutil
import GPUtil
from tqdm import tqdm
from multiprocessing import Process
import matplotlib.pyplot as plt

from client_script import inference

"""
트리톤 서버에서 추론 시, 컴퓨터 자원 사용량을 체크하는 스크립트입니다.
"""

TRITON_SERVER_URL = "localhost:8001"
MODEL_NAME = "protogcn_selfharm"
MODEL_VERSION = "1"

def get_usage():
    """현재 자원 사용량 체크"""
    cpu_usage = psutil.cpu_percent()
    mem_usage = psutil.virtual_memory().used / (1024**3)
    gpu_usage = []
    gpu_mem_usage = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpu_usage.append(gpu.load*100)
        gpu_mem_usage.append(gpu.memoryUsed)
    return {"cpu": cpu_usage, "memory": mem_usage, "gpu": gpu_usage, "gpu_memory": gpu_mem_usage}

def check_idle_usage():
    """유휴 상태 자원 사용량 체크"""
    checking_time = 10
    
    temp_cpu = []
    temp_mem = []
    temp_gpu = []
    temp_gpu_mem = []
    for i in tqdm(range(checking_time), desc="Checking idle usage"):
        usage = get_usage()
        temp_cpu.append(usage["cpu"])
        temp_mem.append(usage["memory"])
        temp_gpu.append(usage["gpu"])
        temp_gpu_mem.append(usage["gpu_memory"])
        time.sleep(1)
    idle_cpu = sum(temp_cpu) / checking_time
    idle_mem = sum(temp_mem) / checking_time
    idle_gpu = [sum(col) / checking_time for col in zip(*temp_gpu)]
    idle_gpu_mem = [sum(col) / checking_time for col in zip(*temp_gpu_mem)]
    
    return {"cpu": idle_cpu, "memory": idle_mem, "gpu": idle_gpu, "gpu_memory": idle_gpu_mem}

def check_active_usage(checking_time = 1):
    """실행 상태 자원 사용량 체크"""
    active_cpu = []
    active_mem = []
    active_gpu = []
    active_gpu_mem = []
    for i in tqdm(range(checking_time), desc="Checking active usage"):
        usage = get_usage()
        active_cpu.append(usage["cpu"])
        active_mem.append(usage["memory"])
        active_gpu.append(usage["gpu"])
        active_gpu_mem.append(usage["gpu_memory"])
        time.sleep(1)

    return {"cpu": active_cpu, "memory": active_mem, "gpu": active_gpu, "gpu_memory": active_gpu_mem}

def worker(num_persons = 1, interval = 1):
    inference(TRITON_SERVER_URL, MODEL_NAME, MODEL_VERSION, num_persons, interval, show_log=False)


if __name__ == '__main__':
    SETUP_DURATION = 30 # 측정 시간 (초)
    INTERVAL = 1  # 추론 요청 간격 (초)
    
    print("Starting performance check...")
    print(f"Setup duration: {SETUP_DURATION} seconds, Interval: {INTERVAL} seconds")
    
    print("Checking idle usage...")
    idle_usage = check_idle_usage()
    
    # 사람 1명 테스팅
    print("Starting tests with different number of persons...")
    print("Testing with 1 person...")
    process = Process(target=worker, args=(1, INTERVAL))
    process.start()
    test_1_usage = check_active_usage(SETUP_DURATION)
    process.terminate()
    process.join()
    
    time.sleep(5)  # 프로세스 간 간격
    
    # 사람 5명 테스팅
    print("Testing with 5 persons...")
    process = Process(target=worker, args=(5, INTERVAL))
    process.start()
    test_2_usage = check_active_usage(SETUP_DURATION)
    process.terminate()
    process.join()
    
    time.sleep(5)  # 프로세스 간 간격
    
    # 사람 10명 테스팅
    print("Testing with 10 persons...")
    process = Process(target=worker, args=(10, INTERVAL))
    process.start()
    test_3_usage = check_active_usage(SETUP_DURATION)
    process.terminate()
    process.join()
    
    print("All tests completed.")
    
    # CPU 사용량 그래프
    idle_cpu_usage = idle_usage['cpu']
    cpu_usage = {'test_1': test_1_usage['cpu'], 
                 'test_2': test_2_usage['cpu'], 
                 'test_3': test_3_usage['cpu']}
    
    # Memory 사용량 그래프
    idle_mem_usage = idle_usage['memory']
    mem_usage = {'test_1': test_1_usage['memory'], 
                 'test_2': test_2_usage['memory'], 
                 'test_3': test_3_usage['memory']}
    
    # GPU 사용량 그래프(cuda:0)
    idle_gpu_usage = idle_usage['gpu'][0]
    gpu_usage = {'test_1': [x[0] for x in test_1_usage['gpu']], 
                 'test_2': [x[0] for x in test_2_usage['gpu']], 
                 'test_3': [x[0] for x in test_3_usage['gpu']]}

    # GPU Memory 사용량 그래프(cuda:0)
    idle_gpu_mem_usage = idle_usage['gpu_memory'][0]
    gpu_mem_usage = {'test_1': [x[0] for x in test_1_usage['gpu_memory']], 
                     'test_2': [x[0] for x in test_2_usage['gpu_memory']], 
                     'test_3': [x[0] for x in test_3_usage['gpu_memory']]}

    # 2x2 subplot 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 시간 축
    time_axis = list(range(SETUP_DURATION))

    # CPU 사용량 그래프
    axes[0, 0].plot(time_axis, cpu_usage['test_1'], label='1 person', marker='o', linewidth=2)
    axes[0, 0].plot(time_axis, cpu_usage['test_2'], label='2 persons', marker='s', linewidth=2)
    axes[0, 0].plot(time_axis, cpu_usage['test_3'], label='3 persons', marker='^', linewidth=2)
    axes[0, 0].axhline(y=idle_cpu_usage, color='r', linestyle='--', label='Idle CPU Usage')
    axes[0, 0].set_title('CPU Usage Difference from Idle (%)', fontsize=14)
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('CPU Usage Difference (%)')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Memory 사용량 그래프
    axes[0, 1].plot(time_axis, mem_usage['test_1'], label='1 person', marker='o', linewidth=2)
    axes[0, 1].plot(time_axis, mem_usage['test_2'], label='2 persons', marker='s', linewidth=2)
    axes[0, 1].plot(time_axis, mem_usage['test_3'], label='3 persons', marker='^', linewidth=2)
    axes[0, 1].axhline(y=idle_mem_usage, color='r', linestyle='--', label='Idle Memory Usage')
    axes[0, 1].set_title('Memory Usage Difference from Idle (GB)', fontsize=14)
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Memory Usage Difference (GB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # GPU 사용량 그래프
    axes[1, 0].plot(time_axis, gpu_usage['test_1'], label='1 person', marker='o', linewidth=2)
    axes[1, 0].plot(time_axis, gpu_usage['test_2'], label='2 persons', marker='s', linewidth=2)
    axes[1, 0].plot(time_axis, gpu_usage['test_3'], label='3 persons', marker='^', linewidth=2)
    axes[1, 0].axhline(y=idle_gpu_usage, color='r', linestyle='--', label='Idle GPU Usage')
    axes[1, 0].set_title('GPU Usage Difference from Idle (%)', fontsize=14)
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('GPU Usage Difference (%)')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # GPU Memory 사용량 그래프
    axes[1, 1].plot(time_axis, gpu_mem_usage['test_1'], label='1 person', marker='o', linewidth=2)
    axes[1, 1].plot(time_axis, gpu_mem_usage['test_2'], label='2 persons', marker='s', linewidth=2)
    axes[1, 1].plot(time_axis, gpu_mem_usage['test_3'], label='3 persons', marker='^', linewidth=2)
    axes[1, 1].axhline(y=idle_gpu_mem_usage, color='r', linestyle='--', label='Idle GPU Memory Usage')
    axes[1, 1].set_title('GPU Memory Usage Difference from Idle (MB)', fontsize=14)
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('GPU Memory Usage Difference (MB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 레이아웃 조정
    plt.tight_layout()
    plt.savefig('result_perfomenace.png', dpi=300, bbox_inches='tight')
    print("Performance check completed. Results saved to 'result_perfomenace.png'.")