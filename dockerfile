FROM python:3.12-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 1. Copy all project files into the container first to prevent explicit name-matching crashes
COPY . .

# 2. Install python dependencies from the copied workspace
RUN pip install --no-cache-dir -r requirements.txt

# Grant permissions to the application directory
RUN chmod -R 777 /app

# Expose the mandatory Hugging Face port
EXPOSE 7860

CMD ["python", "app.py"]