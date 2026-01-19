#!/bin/bash
set -e  # se der erro em algum experimento, para tudo

for i in {0..29}; do

  docker run --rm \
    --gpus all \
    --name thermal-img-experiment \
    -v "$(pwd):/experiment" \
    fix_ufpe_images \
    python -m main \
      --raw_root "filtered_raw_dataset" \
      --angle "Frontal" \
      --k 5 \
      --resize_to 224 \
      --n_aug 2 \
      --batch 8 \
      --message "Vgg_AUG_CV_DatasetSeg2classes_17_01_t${i}" \
      --resize_method "BlackPadding" \
      --segmenter_model "yolo" \
      --seg_model_path "runs/segment/train39/weights/best.pt"
done


