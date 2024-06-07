import ctypes
from ctypes import c_int
import threading
import numpy as np
import time

# Define a compare and swap function using ctypes
def compare_and_swap(address, old, new):
    if address.contents.value == old:
        address.contents.value = new
        return True
    else:
        return False

class Mutex:
    def __init__(self):
        # Mutex state, 0 means unlocked, 1 means locked
        self.lock = ctypes.pointer(c_int(0))

    def acquire(self):
        while True:
            # Try to set the lock from 0 (unlocked) to 1 (locked)
            if compare_and_swap(self.lock, 0, 1):
                return

    def release(self):
        # Set the lock to 0 (unlocked)
        self.lock.contents.value = 0

def matrix_multiply(A, B, C, row, mutex):
    for j in range(B.shape[1]):
        sum = 0
        for k in range(A.shape[1]):
            sum += A[row, k] * B[k, j]
        mutex.acquire()
        C[row, j] = sum
        mutex.release()

def threaded_matrix_multiply(A, B):
    assert A.shape[1] == B.shape[0], "Incompatible matrices for multiplication"

    C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
    mutex = Mutex()

    threads = []
    for i in range(A.shape[0]):
        t = threading.Thread(target=matrix_multiply, args=(A, B, C, i, mutex))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return C

def single_thread_matrix_multiply(A, B):
    assert A.shape[1] == B.shape[0], "Incompatible matrices for multiplication"
    C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            sum = 0
            for k in range(A.shape[1]):
                sum += A[i, k] * B[k, j]
            C[i, j] = sum
    return C

def load_matrix_from_file(filename):
    return np.loadtxt(filename, dtype=int)

# Load matrices from file
A = load_matrix_from_file('matrices/matrix_A.txt')
B = load_matrix_from_file('matrices/matrix_B.txt')

# Perform matrix multiplication with threading
start_time = time.time()
C_threaded = threaded_matrix_multiply(A, B)
end_time = time.time()
threaded_time = end_time - start_time

# Perform matrix multiplication without threading
start_time = time.time()
C_single_thread = single_thread_matrix_multiply(A, B)
end_time = time.time()
single_thread_time = end_time - start_time

# Compare results
print("Threaded matrix multiplication took: {:.4f} seconds".format(threaded_time))
print("Single-threaded matrix multiplication took: {:.4f} seconds".format(single_thread_time))

# Ensure the results are the same
assert np.array_equal(C_threaded, C_single_thread), "The results are different!"

print("Results are the same for both methods.")

# Load result from result.txt
result_from_file = load_matrix_from_file('matrices/result.txt')

# Compare with result from file
assert np.array_equal(C_threaded, result_from_file), "The result from the file is different!"

print("The result from the file matches the computed result.")

# Append the timing results to result_report.txt
with open('matrices/result_report.txt', 'a') as report_file:
    report_file.write(f"Threaded matrix multiplication took: {threaded_time:.4f} seconds\n")
    report_file.write(f"Single-threaded matrix multiplication took: {single_thread_time:.4f} seconds\n")
