#! /bin/bash

IMAGE_NAME=rocm/sgl-dev:rocm720-deepseek-v4-mi35x
CONTAINER_NAME=sgl-deepseek-v4-mi35x-rocm720
SGLANG_WORKDIR=/home/akadhaka/sglang
HF_DIR=/data/workloads-inference/models

docker run -it --rm -d --privileged --name $CONTAINER_NAME \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --network=host --shm-size 64g \
  --entrypoint bash \
  -v "$SGLANG_WORKDIR":/sgl-pr \
  -v "$HF_DIR":/hf \
  -v "$SGLANG_WORKDIR/_alias":/_alias \
  $IMAGE_NAME