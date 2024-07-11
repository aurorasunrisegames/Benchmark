import time
import numpy as np
import tensorflow as tf

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

def benchmark_ai_performance(num_iterations=1000, matrix_size=1000):
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
    result = benchmark_ai_performance()
    print(f"AI Performance: {result:.2f} TOPS")
    print(f"Device(s): {get_device_details()}")
