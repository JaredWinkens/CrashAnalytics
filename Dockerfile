FROM python:3.11-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.0


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN ln -s /mnt/data/data /app/data && \
    ln -s /mnt/data/config.json /app/config.json


EXPOSE 8050

CMD ["python", "app.py"]
