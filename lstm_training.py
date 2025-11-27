import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string

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

# Generate predictions for training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def plot_predictions():
    # Create a plot for training phase
    plt.figure(figsize=(12, 6))
    
    # Training plot
    plt.subplot(1, 2, 1)
    plt.plot(y_train, label='Actual (Training)', alpha=0.7)
    plt.plot(y_train_pred, label='Predicted (Training)', alpha=0.7)
    plt.title('Training Phase: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    
    # Testing plot
    plt.subplot(1, 2, 2)
    plt.plot(y_test, label='Actual (Testing)', alpha=0.7)
    plt.plot(y_test_pred, label='Predicted (Testing)', alpha=0.7)
    plt.title('Testing Phase: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    
    # Save plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # HTML template to display the plot
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Predictions vs Actual</title>
    </head>
    <body>
        <h1>LSTM Model: Predictions vs Actual Values</h1>
        <p>Training Loss: {{ train_loss|round(4) }}, Test Loss: {{ test_loss|round(4) }}</p>
        <img src="data:image/png;base64,{{ plot_url }}" alt="Predictions vs Actual">
    </body>
    </html>
    '''
    return render_template_string(html_template, plot_url=plot_url, train_loss=train_loss, test_loss=test_loss)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)