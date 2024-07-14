import time
import numpy as np
import platform
import subprocess

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

if __name__ == "__main__":
    print(get_system_info())
    result = benchmark_performance()
    print(f"Performance: {result:.2f} GFLOPS")
