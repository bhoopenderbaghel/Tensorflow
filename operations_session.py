import tensorflow as tf 

# #@@@@@@@@@@@@@@@@@@ operations @@@@@@@@@@@@@@@@@@@@@@@
# # Define two constant tensors 

# a = tf.constant(2.0)
# b = tf.constant(4.0)

# c = tf.add(a,b)
# d =  tf.multiply(a,b)

# # print("operation is :",c)
# # print("operation is :",d)

# print("Addition:",c.numpy())
# print("Multiplication:",d.numpy())


# Model parameters
W = tf.Variable([0.3], dtype=tf.float32, name="weights")
b = tf.Variable([-0.3], dtype=tf.float32, name="bias")

# Training data
train_X = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
train_Y = tf.constant([0.0, -1.0, -2.0, -3.0], dtype=tf.float32)

# Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Training loop
for epoch in range(1000):
    with tf.GradientTape() as tape:
        predictions = W * train_X + b
        loss = tf.reduce_mean(tf.square(predictions - train_Y))

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Loss = {loss.numpy():.6f}")

# Final results
print("\nTraining Complete")
print(f"Trained Weight: {W.numpy()[0]:.6f}")
print(f"Trained Bias: {b.numpy()[0]:.6f}")
print(f"Final Loss: {loss.numpy():.6f}")