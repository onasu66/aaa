FROM python:3.10
# 既存のコードと併せて以下の行を追加
RUN python -m pip install --upgrade pip
docker build -t my_flask_app_image .

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
WORKDIR /app

CMD ["python", "app.py"]