# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application code and dependencies
COPY app.py .
COPY requirements.txt .
copy TFIDF_Twitter_sentiment_model.pkl .
copy Twitter_sentiment_model.pkl .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by Streamlit
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
