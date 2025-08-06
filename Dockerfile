FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including libGL for OpenCV
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.0


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN ln -s /mnt/data/data /app/data && \
    ln -s /mnt/data/config.json /app/config.json


EXPOSE 8080

CMD ["python", "app.py"]
