import time
import numpy as np
import platform
import subprocess
import tensorflow as tf
import torch

def get_system_info():
    system = platform.system()
    release = platform.release()
    machine = platform.machine()

    gpu_info = "Not available"
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").strip()
    except:
        pass

    return f"System: {system} {release} {machine}\nGPU: {gpu_info}"

def benchmark_performance(num_iterations=1000, matrix_size=1000):
    # Create random matrices
    matrix_a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    matrix_b = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        np.dot(matrix_a, matrix_b)
    end_time = time.time()

    # Calculate FLOPS
    elapsed_time = end_time - start_time
    operations = 2 * matrix_size**3 * num_iterations  # Approximate number of operations for matrix multiplication
    gflops = operations / (elapsed_time * 1e9)  # Convert to GFLOPS

    return gflops

def get_device_details():
    devices = tf.config.list_physical_devices()
    device_details = []
    for device in devices:
        device_type = device.device_type
        if device_type == 'GPU':
            gpu_details = tf.config.experimental.get_device_details(device)
            device_details.append(f"GPU: {gpu_details['device_name']}")
        else:
            device_details.append(f"{device_type}")
    return ", ".join(device_details)

def benchmark_ai_performance_torch(num_iterations=1000, matrix_size=1000):
    # Create random matrices
    matrix_a = torch.rand(matrix_size, matrix_size, dtype=torch.float32)
    matrix_b = torch.rand(matrix_size, matrix_size, dtype=torch.float32)

    # Warm-up run
    _ = torch.matmul(matrix_a, matrix_b)

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = torch.matmul(matrix_a, matrix_b)
    end_time = time.time()

    # Calculate TOPS
    elapsed_time = end_time - start_time
    operations = 2 * matrix_size**3 * num_iterations  # Approximate number of operations for matrix multiplication
    tops = operations / (elapsed_time * 1e12)  # Convert to TOPS

    return tops


def benchmark_ai_performance_tf(num_iterations=1000, matrix_size=1000):
    # Create random matrices
    matrix_a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    matrix_b = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    # Convert to TensorFlow tensors
    tf_a = tf.constant(matrix_a)
    tf_b = tf.constant(matrix_b)

    # Warm-up run
    _ = tf.matmul(tf_a, tf_b)

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = tf.matmul(tf_a, tf_b)
    end_time = time.time()

    # Calculate TOPS
    elapsed_time = end_time - start_time
    operations = 2 * matrix_size**3 * num_iterations  # Approximate number of operations for matrix multiplication
    tops = operations / (elapsed_time * 1e12)  # Convert to TOPS

    return tops

if __name__ == "__main__":
    # print(get_system_info())
    print(f"Device(s): {get_device_details()}")
    result = benchmark_performance()
    print(f"Performance: {result:.2f} GFLOPS")
    result_torch = benchmark_ai_performance_torch()
    print(f"AI Performance torch: {result_torch:.2f} TOPS")
    result_tf = benchmark_ai_performance_tf()
    print(f"AI Performance tensorflow: {result_tf:.2f} TOPS")
