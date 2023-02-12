import tensorflow as tf
import time

# Define the size of the matrices
shape = (10000, 10000)

# Create two random matrices
matrix1 = tf.random.normal(shape=shape)
matrix2 = tf.random.normal(shape=shape)

# Start the timer
start = time.time()

# Perform the matrix multiplication
result = tf.matmul(matrix1, matrix2)

# Force the computation to run on the GPU
with tf.device('GPU:0'):
    result.numpy()

# Stop the timer
end = time.time()

# Calculate the elapsed time
elapsed = end - start

print("Elapsed time: {:.2f} seconds".format(elapsed))
