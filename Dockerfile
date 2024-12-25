FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "app.py"]
