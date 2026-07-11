FROM python:3.12-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Grant permissions to the application directory
RUN chmod -R 777 /app

# Expose the mandatory Hugging Face port
EXPOSE 7860

CMD ["python", "app.py"]