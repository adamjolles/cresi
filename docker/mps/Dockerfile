# Dockerfile for running CRESI on Apple hardware.
# Build with: docker build -f docker/mps/Dockerfile -t cresi_mps .

FROM ubuntu:22.04
WORKDIR /opt/cresi

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-pip git wget unzip libgl1 libgdal-dev gdal-bin build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . /opt/cresi

RUN unzip /opt/cresi/results/weights/aws_weights/fold0_best.pth.zip -d /opt/cresi/results/aws_weights/weights

CMD ["/bin/bash"]
