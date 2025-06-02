# eigencam_run.py
import os, numpy as np, tensorflow as tf, cv2
from PIL import Image

def run_eigencam(imgs, masks = None, model_path = "", out_dir="cam_out", layer_name=None):
    """
    imgs  : np.ndarray (N,H,W,1) normalizado 0-1
    masks : np.ndarray (N,H,W) binário OU None
    """
    os.makedirs(out_dir, exist_ok=True)
    N, H, W, _ = imgs.shape
    if masks is not None and masks.shape != (N, H, W):
        raise ValueError("Máscaras e imagens têm shapes incompatíveis")

    # ---- carregando modelo ----
    from tensorflow.keras.utils import custom_object_scope
    from src.models.resNet_34 import ResNet34, ResidualUnit

    with custom_object_scope({'ResidualUnit': ResidualUnit}):
                        model = tf.keras.models.load_model(model_path, compile=False)

    # ---- escolhe última Conv2D se layer_name == None ----
    if layer_name is None:
        for lyr in reversed(model.layers):
            if isinstance(lyr, tf.keras.layers.Conv2D):
                layer_name = lyr.name
                break
    print(f"[INFO] Usando camada '{layer_name}'")

    # # ---- mini-preprocess: só resize para 224×224 (nada de re-escala) ----
    # preprocess = tf.keras.Sequential([
    #     tf.keras.layers.Resizing(224, 224)  # se suas imagens já são 224×224, pode remover
    # ])

    # ---- EigenCAM ----
    feat = model.get_layer(layer_name).output
    cam_model = tf.keras.Model(model.input, {"logits": model.output, "feat": feat})

    def calc_cam(img1):            # img1: (H,W,1) float 0-1
       # x = preprocess(img1)[None]  # (1,224,224,1)
        out = cam_model(img1[None])  # (1,224,224,1)  logits: (1,2) feat: (1,Hc,Wc,C)
        f   = tf.transpose(out["feat"], [0,3,1,2])     # (1,C,H,W)
        f = tf.cast(f, tf.float32)  # converte para float32 se necessário
        s,u,v = tf.linalg.svd(f)
        cam = u[...,:1] @ s[...,:1,None] @ tf.transpose(v[...,:1],[0,1,3,2])
        cam = tf.reduce_sum(cam,1)[0]                  # (Hc,Wc)
        cam -= tf.reduce_min(cam); cam /= tf.reduce_max(cam)+1e-8
        cam = cv2.resize(cam.numpy(), (W, H))          # volta para tamanho original
        return cam, out["logits"][0].numpy()

    def to_rgb(gray):  # (H,W,1) 0-1 → (H,W,3) uint8
        g255 = (gray.squeeze()*255).astype(np.uint8)
        return np.repeat(g255[...,None], 3, axis=2)

    def heatmap_rgb(cam):
        col = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
        return col[:,:,::-1]  # BGR→RGB

    def mix(base, heat, alpha=.4):
        return np.uint8((1-alpha)*base + alpha*heat)

    def overlap(cam, mask, thr=.5):
        hot = cam >= thr
        return (hot & mask).sum() / (hot.sum()+1e-6)

    scores=[]
    for i in range(N):
        cam, logit = calc_cam(imgs[i])

        rgb    = to_rgb(imgs[i])          # imagem em escala de cinza → RGB
        heat   = heatmap_rgb(cam)         # heatmap colorido
        ov_img = mix(rgb, heat)           # overlay

        # ----- salvar -----
        Image.fromarray(heat).save(f"{out_dir}/sample_{i}_heatmap.png")
        Image.fromarray(ov_img).save(f"{out_dir}/sample_{i}_overlay.png")

        # ----- métrica opcional -----
        if masks is not None:
            sc = overlap(cam, masks[i])
            scores.append(sc)
            print(f"sample_{i}: prob={logit.squeeze():.3f}  overlap={sc:.3f}")
        else:
            print(f"sample_{i}: prob={logit.squeeze():.3f}")

    if scores:
        print(f"\nOverlap médio: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# ---------------------------------------------------------------------
# EXEMPLO DE USO
# ---------------------------------------------------------------------
# if __name__ == "__main__":
#     # 1) carregue seus arrays .npy ou crie dummy arrays para testar
#     imgs  = np.load("x_val.npy")            # shape (N,H,W,1), float32 0-1
#     masks = np.load("masks_val.npy")        # shape (N,H,W)   0/1  OU  None

#     run_eigencam(
#         imgs       = imgs,
#         masks      = masks,
#         model_path = "resnet34.h5",
#         out_dir    = "cam_results"
#     )
