import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

print(f"Training data shape:{train_images.shape}, Training labels shape:{train_labels.shape}")
print(f"Test data shape:{test_images.shape}, Test labels shape:{test_labels.shape}")

# Normalize the images to a range of 0 to 1 by dividing by 255.0

train_images,test_images = train_images/255.0,test_images/255.0

# Split the training data into training and validation sets
train_images,val_images,train_labels,val_labels = train_test_split(train_images,train_labels,test_size = 0.2,
                                                                   random_state = 42)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), # Flatten layer
    tf.keras.layers.Dense(128,activation='relu'), # Hidden dense layer with relu activation
    tf.keras.layers.Dense(10,activation='softmax')]) # Output dense layer with Softmax activation


model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

history = model.fit(train_images,train_labels,epochs = 10,validation_data = (val_images,val_labels))



# Visualize a sample image

plt.imshow(train_images[100],cmap = 'gray')
plt.title(f"Label: {train_labels[100]}")
plt.show()