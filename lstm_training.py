import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate 730 random data points between 0 and 1 with a slight bias towards higher numbers over time
np.random.seed(42)  # For reproducibility
data_points = []
for i in range(730):
    # Add a bias: as time (i) increases, add a small positive bias to the random number
    bias = i / 7300  # This adds up to 0.1 bias at the end, making numbers slightly higher over time
    point = np.random.random() + bias
    point = np.clip(point, 0, 1)  # Ensure the point stays between 0 and 1
    data_points.append(point)

data_points = np.array(data_points)

# Prepare sequences: use previous 10 points to predict the next point
sequence_length = 10
X = []
y = []
for i in range(len(data_points) - sequence_length):
    X.append(data_points[i:i+sequence_length])
    y.append(data_points[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape X for LSTM input: [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing sets (e.g., 80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Example prediction: use the last sequence from the test set for prediction
last_sequence = X_test[-1]
last_sequence = last_sequence.reshape((1, sequence_length, 1))
prediction = model.predict(last_sequence)
print(f"Predicted next value: {prediction[0][0]:.4f}")
print(f"Actual next value: {y_test[-1]:.4f}")