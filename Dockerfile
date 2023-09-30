FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && apt-get clean
RUN pip3 install torch torchvision

WORKDIR /workspace
