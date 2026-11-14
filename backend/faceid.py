"""Shared face pipeline: MTCNN detection + alignment crop, InceptionResnetV1 (VGGFace2) embeddings."""
import numpy as np, torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face

DEV = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=DEV, min_face_size=40, thresholds=[0.6, 0.7, 0.9])
net = InceptionResnetV1(pretrained="vggface2").eval().to(DEV)
MIN_DET_PROB = 0.95

def load(src):
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    img.thumbnail((1600, 1600))
    return img

@torch.no_grad()
def embed(img, largest_only=False):
    """Returns (boxes, det_probs, L2-normalised 512-d embeddings), biggest face first."""
    empty = ([], [], np.zeros((0, 512), dtype="float32"))
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return empty
    order = sorted((i for i, p in enumerate(probs) if p is not None and p >= MIN_DET_PROB),
                   key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    if largest_only:
        order = order[:1]
    if not order:
        return empty
    faces = torch.stack([(extract_face(img, boxes[i], 160, 14) - 127.5) / 128.0 for i in order]).to(DEV)
    e = net(faces).cpu().numpy()
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return [boxes[i].tolist() for i in order], [float(probs[i]) for i in order], e
