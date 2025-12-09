#!/bin/bash
set -e  # se der erro em algum experimento, para tudo


echo ">>> Rodando modelo $i do dataset original..."

docker run --rm \
  --gpus all \
  --name thermal-img-experiment \
  -v "$(pwd):/experiment" \
  fix_ufpe_images \
  python -m main \
    --raw_root "filtered_raw_dataset" \
    --message "treinandoYolo" \
    --segment "yolo"


# # 1) Primeira leva de 6 modelos (dataset original)
# for i in {3..5}; do
#   echo ">>> Rodando modelo $i do dataset original..."

#   docker run --rm \
#     --gpus all \
#     --name thermal-img-experiment \
#     -v "$(pwd):/experiment" \
#     fix_ufpe_images \
#     python -m main \
#       --raw_root "recovered_data" \
#       --angle "Frontal" \
#       --k 5 \
#       --resize_to 224 \
#       --n_aug 2 \
#       --batch 8 \
#       --message "Vgg_AUG_CV_DatasetSemMarcador_t${i}" \
#       --resize_method "BlackPadding"
# done

# for i in {0..5}; do
#   echo ">>> Rodando modelo $i do dataset original..."

#   docker run --rm \
#     --gpus all \
#     --name thermal-img-experiment \
#     -v "$(pwd):/experiment" \
#     fix_ufpe_images \
#     python -m main \
#       --raw_root "processed_images_padmovido_txt" \
#       --angle "Frontal" \
#       --k 5 \
#       --resize_to 224 \
#       --n_aug 2 \
#       --batch 8 \
#       --message "Vgg_AUG_CV_DatasetMarcadorMovidoFixo_t${i}" \
#       --resize_method "BlackPadding"
# done

# for i in {0..5}; do
#   echo ">>> Rodando modelo $i do dataset original..."

#   docker run --rm \
#     --gpus all \
#     --name thermal-img-experiment \
#     -v "$(pwd):/experiment" \
#     fix_ufpe_images \
#     python -m main \
#       --raw_root "filtered_raw_dataset" \
#       --angle "Frontal" \
#       --k 5 \
#       --resize_to 224 \
#       --n_aug 2 \
#       --batch 8 \
#       --message "Vgg_AUG_CV_DatasetSegmentadoUnet_t${i}" \
#       --resize_method "BlackPadding" \
#       --segmenter_model "unet" \
#       --seg_model_path "modelos/unet/unet_AUG_6_12.h5"
# done

# #!/bin/bash
# set -e  # se der erro em algum experimento, para tudo

# # 1) Primeira leva de 6 modelos (dataset original)
# for i in {2..5}; do
#   echo ">>> Rodando modelo $i do dataset original..."

#   docker compose run --rm thermal-img-experiment \
#     python -m main \
#       --raw_root "filtered_raw_dataset" \
#       --angle "Frontal" \
#       --k 5 \
#       --resize_to 224 \
#       --n_aug 2 \
#       --batch 8 \
#       --message "Vgg_AUG_CV_DatasetOriginal_t${i}" \
#       --resize_method "BlackPadding"
# done

# # 2) Segunda leva de 6 modelos (dataset processado)
# for i in {0..5}; do
#   echo ">>> Rodando modelo $i do dataset com tags..."

#   docker compose run --rm thermal-img-experiment \
#     python -m main \
#       --raw_root "processed_images(pad 28x28px)_teste_txt" \
#       --angle "Frontal" \
#       --k 5 \
#       --resize_to 224 \
#       --n_aug 2 \
#       --batch 8 \
#       --message "Vgg_AUG_CV_DatasetTagFixedTam_t${i}" \
#       --resize_method "BlackPadding"
# done
