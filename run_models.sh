#!/bin/bash
set -e  # se der erro em algum experimento, para tudo


# 1) Primeira leva de 6 modelos (dataset original)


# for i in {0..2}; do
#   docker run --rm \
#     --gpus all \
#     --name thermal-img-experiment \
#     -v "$(pwd):/experiment" \
#     fix_ufpe_images \
#     python -m main \
#       --raw_root "Frontal_temp_fixa_txt_rounded" \
#       --angle "Frontal" \
#       --k 5 \
#       --resize_to 224 \
#       --n_aug 2 \
#       --batch 8 \
#       --message "Vgg_AUG_CV_DatasetFixedTagTemp_09_01t${i}_teste" \
#       --resize_method "BlackPadding"
# done

# for i in {0..2}; do
#   docker run --rm \
#     --gpus all \
#     --name thermal-img-experiment \
#     -v "$(pwd):/experiment" \
#     fix_ufpe_images \
#     python -m main \
#       --raw_root "processed_images_padmovido_txt_rounded" \
#       --angle "Frontal" \
#       --k 5 \
#       --resize_to 224 \
#       --n_aug 2 \
#       --batch 8 \
#       --message "Vgg_AUG_CV_DatasetMarcadorMovidoFixo_09_01_t${i}_teste" \
#       --resize_method "BlackPadding"
# done


for i in {0..2}; do

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
      --message "Vgg_AUG_CV_DatasetSegYolo_09_01_t${i}_teste" \
      --resize_method "BlackPadding" \
      --segmenter_model "yolo" \
      --seg_model_path "runs/segment/train39/weights/best.pt"
done


# for i in {0..2}; do
#   docker run --rm \
#     --gpus all \
#     --name thermal-img-experiment \
#     -v "$(pwd):/experiment" \
#     fix_ufpe_images \
#     python -m main \
#       --raw_root "processed_images(pad 28x28px)_teste_txt_rounded" \
#       --angle "Frontal" \
#       --k 5 \
#       --resize_to 224 \
#       --n_aug 2 \
#       --batch 8 \
#       --message "Vgg_AUG_CV_DatasetTagFixedTam_09_01_t${i}_teste" \
#       --resize_method "BlackPadding"
# done
