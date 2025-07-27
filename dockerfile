FROM python:3.10

 

WORKDIR /app

 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

 

COPY serve_ab.py .

 

CMD ["uvicorn", "serve_ab:app", "--host", "0.0.0.0", "--port", "8000"]