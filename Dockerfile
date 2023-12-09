FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y libsndfile1

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

CMD ["streamlit", "run", "app.py"]
