#!/bin/sh
 # --mount type=bind,source=$mydataset,target=/root/dataset \
  # --mount type=bind,source=${MY_LOG_DIR},target=/root/logs \

dirname=$(pwd | xargs dirname)
# dataset="/share/private/27th/hirotaka_saito/dataset/"
# dataset="/share/share/RWRC/rwrc21_dl/dataset/"
dataset="/share/private/27th/hirotaka_saito/"
docker run -it \
  --privileged \
  --gpus all \
  -p 15900:5900 \
  --rm \
  --mount type=bind,source=$dirname,target=/root/cnn_base_mae \
  --mount type=bind,source=$dataset,target=/root/dataset \
   --net host \
   --shm-size=100000m \
  CnnBaseMAE
  bash
