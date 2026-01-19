#!/bin/bash
set -e  # se der erro em algum experimento, para tudo


# 1) Primeira leva de 6 modelos (dataset original)

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
      --message "Vgg_AUG_CV_DatasetOriginal_9_01_t${i}" \
      --resize_method "BlackPadding"
done

for i in {0..29}; do
  docker run --rm \
    --gpus all \
    --name thermal-img-experiment \
    -v "$(pwd):/experiment" \
    fix_ufpe_images \
    python -m main \
      --raw_root "recovered_data_rounded" \
      --angle "Frontal" \
      --k 5 \
      --resize_to 224 \
      --n_aug 2 \
      --batch 8 \
      --message "Vgg_AUG_CV_DatasetSemMarcador_09_01_t${i}" \
      --resize_method "BlackPadding"
done
