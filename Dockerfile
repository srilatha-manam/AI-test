# Use an official Python image
FROM python:3.10-slim
 
# Set the working directory in the container
WORKDIR /app
 
# Copy the requirements file into the container
COPY requirement.txt .
 
# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt
 
# Copy the application code into the container
COPY . .
 
# Expose the port FastAPI will run on
EXPOSE 8000
 
# Run the application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
 