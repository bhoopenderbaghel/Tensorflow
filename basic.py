import tensorflow as tf
# # print(tf.__version__)

# # const_tensor = tf.constant([1,2,3],dtype=tf.int32)
# # print("Constant Tensor:",const_tensor)

# var_tensor = tf.Variable([1.0,2.0,3.0],dtype=tf.float32)
# print("Variable Tensor:",var_tensor)

# Initializing a variable 

# var_zero = tf.Variable(tf.zeros([2,3]),name="zero_initialized")

# var_random = tf.Variable(tf.random.normal([2,3]),name="random_initialized")

# print("Zero-initialized Variable:", var_zero)
# print("Random-initialized Variable:", var_random)

# weights = tf.Variable(tf.random.normal([784,256]),name="weights")
# print("Weights Variable:", weights)

# bias = tf.Variable(tf.constant([0.1]*256,dtype=tf.float32,name="bias"))
# print("Bias Variable:", bias)

@tf.function
def compute(a,b):
    return a*b + 2

print(compute(12.0,55.0).numpy())