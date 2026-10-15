# 시점별 위험 구역(프레스 본체와 조작부). 좌표는 1920x1080 원본 기준, 사람이 화면을 보고 그린 값
import cv2, numpy as np
from huggingface_hub import hf_hub_download
REPO = "Voxel51/Safe_and_Unsafe_Behaviours"
REF = {"A": "data/1_tr54.mp4", "B": "data/1_te1.mp4"}   # A: 프레스 근접 시점, B: 넓은 시점
ZONES = {
    "A": np.array([(0, 150), (600, 150), (660, 560), (780, 600), (780, 760), (640, 1080), (0, 1080)], np.int32),
    "B": np.array([(0, 250), (380, 120), (820, 120), (880, 700), (820, 1080), (0, 1080)], np.int32),
}
def mid_frame(fp):
    cap = cv2.VideoCapture(hf_hub_download(REPO, fp, repo_type="dataset"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, img = cap.read(); cap.release(); return img
def thumb(img): return cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
_REF_T = {}
def view_of(img):
    """기준 프레임 두 장과 상관계수를 비교해 시점을 고른다"""
    if not _REF_T: _REF_T.update({v: thumb(mid_frame(fp)) for v, fp in REF.items()})
    t = thumb(img)
    return max(_REF_T, key=lambda v: np.corrcoef(t, _REF_T[v])[0, 1])
