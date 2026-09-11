import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

# Define the inputs
image_input = Input(shape=(784,))
tabular_input = Input(shape=(10,))

# Image branch
x1 = Dense(64, activation='relu')(image_input)

# Tabular branch
x2 = Dense(32, activation='relu')(tabular_input)

# Concatenate both branches
merged = concatenate([x1, x2])

# Output layer
output = Dense(10, activation='softmax')(merged)

# Create the model
model = Model(
    inputs=[image_input, tabular_input],
    outputs=output
)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()