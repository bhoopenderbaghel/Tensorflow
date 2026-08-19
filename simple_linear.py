import tensorflow as tf

# Define Contants for Input data

X = tf.constant([[1.0],[2.0],[3.0],[4.0]],dtype=tf.float32)  # Input data
Y = tf.constant([[2.0],[4.0],[6.0],[8.0]],dtype=tf.float32) # target Values

# Define Variables for weights and bias
W = tf.Variable([[0.0]],dtype= tf.float32,name="weights") # weight variable
b = tf.Variable([[0.0]],dtype= tf.float32,name="bias") # bias variable  

def linear_model(X):
    return tf.matmul(X,W)+b

def loss_fn(y_pred,y_true):
    return tf.reduce_mean(tf.square(y_pred-y_true))

optimizer = tf.optimizers.SGD(learning_rate=0.01)

for epoch in range(1000):
    with tf.GradientTape() as tape:
        y_pred = linear_model(X)
        loss = loss_fn(y_pred,Y)
    gradients = tape.gradient(loss,[W,b])
    optimizer.apply_gradients(zip(gradients,[W,b]))
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss: {loss.numpy()}")


print("Trained Weights:", W.numpy())
print("Trained Bias:", b.numpy())