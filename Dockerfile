FROM python:3.10-slim

# Install system dependencies including libgomp1 (required for XGBoost)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for model artifacts and static files
RUN mkdir -p /app/models
RUN mkdir -p /app/static

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port 8080 for web server
EXPOSE 8080

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "600", "app:app"]
