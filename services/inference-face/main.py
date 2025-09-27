import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
from fastapi.responses import JSONResponse
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from insightface.model_zoo import get_model

inswapper_model = None  # keep this global

def _ensure_inswapper_loaded():
    """Load InSwapper from local cache (preferred) or try named download."""
    global inswapper_model
    if inswapper_model is not None:
        return
    local_inswapper = os.path.join(INSIGHTFACE_HOME, "models", "inswapper_128.onnx")
    try:
        if os.path.isfile(local_inswapper):
            inswapper_model = get_model(local_inswapper, providers=PROVIDERS)
        else:
            inswapper_model = get_model("inswapper_128.onnx", root=INSIGHTFACE_HOME, providers=PROVIDERS)
    except Exception as e:
        inswapper_model = None
        print(f"[WARN] InSwapper load failed at runtime: {e}")

# --------------------------- Config ---------------------------

INSIGHTFACE_HOME = os.getenv("INSIGHTFACE_HOME", "/root/.insightface")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "/data/uploads")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/data/outputs")

app = FastAPI(title="Face Inference (InsightFace buffalo_l + InSwapper)", version="1.0.0")

# Provider auto-detect (GPU if present, else CPU)
_avail = ort.get_available_providers()
if "CUDAExecutionProvider" in _avail:
    PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    CTX_ID = 0
else:
    PROVIDERS = ["CPUExecutionProvider"]
    CTX_ID = -1

face_app: Optional[FaceAnalysis] = None
inswapper_model = None  # loaded at startup

# ------------------------ App lifecycle -----------------------

@app.on_event("startup")
def startup_event():
    global face_app, inswapper_model
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    try:
        print("[INFO] Providers available:", _avail, "using:", PROVIDERS, "CTX_ID:", CTX_ID)
        print("[INFO] INSIGHTFACE_HOME:", INSIGHTFACE_HOME)
        face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS, root=INSIGHTFACE_HOME)
        face_app.prepare(ctx_id=CTX_ID, det_size=(640, 640))  # możesz podnieść do (1024,1024)
        print("[OK] FaceAnalysis ready")
    except Exception as e:
        init_errors.append(f"face_app init failed: {e}")
        print("[WARN] face_app init failed:", e)

    # Try local file first (offline-friendly), then remote name
    local_inswapper = os.path.join(INSIGHTFACE_HOME, "models", "inswapper_128.onnx")
    try:
        if os.path.isfile(local_inswapper):
            inswapper_model = get_model(local_inswapper, providers=PROVIDERS)
            print(f"[OK] Loaded InSwapper locally: {local_inswapper}")
        else:
            inswapper_model = get_model("inswapper_128.onnx", root=INSIGHTFACE_HOME, providers=PROVIDERS)
            print(f"[OK] Downloaded InSwapper into {INSIGHTFACE_HOME}")
    except Exception as e:
        inswapper_model = None
        print(f"[WARN] InSwapper not available: {e}")


# --------------------------- Utils ----------------------------

@app.get("/infer/models")
def models_status():
    local_inswapper = os.path.join(INSIGHTFACE_HOME, "models", "inswapper_128.onnx")
    return {
        "insightface_home": INSIGHTFACE_HOME,
        "inswapper_exists": os.path.isfile(local_inswapper),
        "providers_available": _avail,
        "using_providers": PROVIDERS
    }

def _load_image_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img  # BGR

def _best_face(image_bgr: np.ndarray):
    faces = face_app.get(image_bgr)
    if not faces:
        return None, []
    best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    return best, faces

def _five_points(face_obj) -> Optional[np.ndarray]:
    kps = getattr(face_obj, "kps", None)
    if kps is not None:
        return kps.astype(np.float32)  # (5,2)
    lm106 = getattr(face_obj, "landmark_2d_106", None)
    if lm106 is None or lm106.shape[0] < 5:
        return None
    idxs = [33, 63, 52, 76, 82]  # rough fallback if kps missing
    return lm106[idxs, :].astype(np.float32)

def _get_lm106(face_obj) -> Optional[np.ndarray]:
    lm = getattr(face_obj, "landmark_2d_106", None)
    if lm is None or lm.shape[0] < 20:
        return None
    return lm.astype(np.float32)  # (106,2)

def _estimate_similarity_many(src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    return M  # 2x3

def _warp_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 2)
    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    return (homo @ M.T).astype(np.float32)

def _face_scale_from_landmarks(lm106: np.ndarray) -> float:
    min_xy = lm106.min(axis=0)
    max_xy = lm106.max(axis=0)
    return float(max(max_xy[0]-min_xy[0], max_xy[1]-min_xy[1]))

def _mask_from_points(points: np.ndarray, h: int, w: int, inflate_px: int = 0) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if points is None or len(points) < 3 or not np.isfinite(points).all():
        return mask
    pts_i = np.round(points).astype(np.int32)
    hull = cv2.convexHull(pts_i)
    cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 255)
    if inflate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_px*2+1, inflate_px*2+1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def _color_transfer_lab(src_bgr: np.ndarray, dst_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is not None:
        m = (mask > 0)
        if m.sum() < 10:
            return src_bgr
        src_mu = [src[:, :, c][m].mean() for c in range(3)]
        src_sd = [src[:, :, c][m].std() + 1e-6 for c in range(3)]
        dst_mu = [dst[:, :, c][m].mean() for c in range(3)]
        dst_sd = [dst[:, :, c][m].std() + 1e-6 for c in range(3)]
        out = src.copy()
        for c in range(3):
            out[:, :, c] = ((out[:, :, c] - src_mu[c]) * (dst_sd[c] / src_sd[c])) + dst_mu[c]
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    # global
    src_mu, src_sd = src.mean(axis=(0,1)), src.std(axis=(0,1)) + 1e-6
    dst_mu, dst_sd = dst.mean(axis=(0,1)), dst.std(axis=(0,1)) + 1e-6
    out = ((src - src_mu) * (dst_sd / src_sd)) + dst_mu
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

def _save_or_stream_png(img_bgr: np.ndarray, prefix: str, download: int):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    out_name = f"{prefix}_{ts}.png"
    out_path = os.path.join(OUTPUTS_DIR, out_name)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if not cv2.imwrite(out_path, img_bgr):
        raise HTTPException(500, "Failed to write output PNG")
    if int(download) == 1:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise HTTPException(500, "PNG encode failed")
        return Response(
            content=buf.tobytes(),
            media_type="image/png",
            headers={"Content-Disposition": f'inline; filename="{out_name}"'}
        )
    return JSONResponse({"ok": True, "output_path": out_path})

# -------------------------- Endpoints -------------------------

@app.get("/infer/health")
def health():
    return {
        "ok": True,
        "service": "inference-face",
        "onnx_providers_available": _avail,
        "using_providers": PROVIDERS,
        "ctx_id": CTX_ID,
        "insightface_home": INSIGHTFACE_HOME,
    }

@app.post("/infer/landmarks")
async def infer_landmarks(image: UploadFile = File(...)):
    if face_app is None:
        raise HTTPException(500, "Model not initialized")
    try:
        bgr = _load_image_from_upload(await image.read())
        faces = face_app.get(bgr)
        if not faces:
            return JSONResponse({"faces": [], "best_index": -1})
        results = []
        for f in faces:
            lm = getattr(f, "landmark_2d_106", None)
            lm = (lm.astype(float).tolist() if lm is not None else f.kps.astype(float).tolist())
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            results.append({
                "det_score": float(getattr(f, "det_score", 0.0)),
                "bbox": [x1, y1, x2, y2],
                "landmarks_106": lm
            })
        best_idx = max(range(len(results)), key=lambda i: results[i]["det_score"])
        return JSONResponse({"faces": results, "best_index": best_idx})
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

@app.post("/infer/align")
async def align_and_swap_v1(
    template: UploadFile = File(...),
    portrait: UploadFile = File(...),
    blend: str = Form("seamless"),
    download: int = Form(0),
):
    """Fast 5-point similarity warp + blend."""
    if face_app is None:
        raise HTTPException(500, "Model not initialized")
    try:
        img_t = _load_image_from_upload(await template.read())
        img_p = _load_image_from_upload(await portrait.read())
        f_t, _ = _best_face(img_t)
        f_p, _ = _best_face(img_p)
        if f_t is None: raise HTTPException(400, "No face found in template")
        if f_p is None: raise HTTPException(400, "No face found in portrait")
        pts_t = _five_points(f_t); pts_p = _five_points(f_p)
        if pts_t is None or pts_p is None or pts_t.shape != (5,2) or pts_p.shape != (5,2):
            raise HTTPException(500, "Could not get 5-point landmarks")
        M, _ = cv2.estimateAffinePartial2D(pts_p, pts_t, method=cv2.LMEDS)
        if M is None: raise HTTPException(500, "Failed to estimate transform")
        h, w = img_t.shape[:2]
        warped = cv2.warpAffine(img_p, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # mask from 5 pts
        if not np.isfinite(pts_t).all(): raise HTTPException(500, "Invalid template landmarks")
        pts_i = np.round(pts_t).astype(np.int32)
        hull = cv2.convexHull(pts_i)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 255)
        k = max(3, int(0.03 * max(h, w))); k |= 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

        # blend
        if blend == "seamless":
            m_bin = (mask > 0).astype(np.uint8) * 255
            ys, xs = np.where(m_bin > 0)
            if len(xs) == 0: raise HTTPException(500, "Empty mask")
            cx, cy = int(xs.mean()), int(ys.mean())
            try:
                result = cv2.seamlessClone(warped, img_t, m_bin, (cx, cy), cv2.NORMAL_CLONE)
            except cv2.error:
                alpha = (mask.astype(np.float32)/255.0)[:, :, None]
                result = (alpha * warped + (1.0 - alpha) * img_t).astype(np.uint8)
        else:
            alpha = (mask.astype(np.float32)/255.0)[:, :, None]
            result = (alpha * warped + (1.0 - alpha) * img_t).astype(np.uint8)

        return _save_or_stream_png(result, "swap", download)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Align/merge error: {e}")

@app.post("/infer/align_v2")
async def align_and_swap_v2(
    template: UploadFile = File(...),
    portrait: UploadFile = File(...),
    blend: str = Form("seamless"),
    download: int = Form(0),
):
    """
    Higher quality: 106-pt fit + convex hull mask (inflated) + Lab color transfer + blend.
    """
    if face_app is None:
        raise HTTPException(500, "Model not initialized")
    try:
        img_t = _load_image_from_upload(await template.read())
        img_p = _load_image_from_upload(await portrait.read())
        h, w = img_t.shape[:2]

        f_t, _ = _best_face(img_t)
        f_p, _ = _best_face(img_p)
        if f_t is None: raise HTTPException(400, "No face found in template")
        if f_p is None: raise HTTPException(400, "No face found in portrait")

        lm_t = _get_lm106(f_t)
        lm_p = _get_lm106(f_p)

        if lm_t is None or lm_p is None:
            # fallback to v1
            pts_t = _five_points(f_t); pts_p = _five_points(f_p)
            if pts_t is None or pts_p is None:
                raise HTTPException(500, "No landmarks for alignment")
            M, _ = cv2.estimateAffinePartial2D(pts_p, pts_t, method=cv2.LMEDS)
            if M is None: raise HTTPException(500, "Failed to estimate transform")
            warped = cv2.warpAffine(img_p, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            pts_i = np.round(pts_t).astype(np.int32)
            hull = cv2.convexHull(pts_i)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 255)
        else:
            M = _estimate_similarity_many(lm_p, lm_t)
            if M is None: raise HTTPException(500, "Failed to estimate transform (106)")
            warped = cv2.warpAffine(img_p, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            lm_p_warp = _warp_points(lm_p, M)
            inflate = int(max(8, 0.04 * _face_scale_from_landmarks(lm_p_warp)))
            mask = _mask_from_points(lm_p_warp, h, w, inflate_px=inflate)

        warped_ct = _color_transfer_lab(warped, img_t, mask=mask)

        if blend == "seamless":
            m_bin = (mask > 0).astype(np.uint8) * 255
            ys, xs = np.where(m_bin > 0)
            if len(xs) == 0: raise HTTPException(500, "Empty mask")
            cx, cy = int(xs.mean()), int(ys.mean())
            try:
                result = cv2.seamlessClone(warped_ct, img_t, m_bin, (cx, cy), cv2.NORMAL_CLONE)
            except cv2.error:
                try:
                    result = cv2.seamlessClone(warped_ct, img_t, m_bin, (cx, cy), cv2.MIXED_CLONE)
                except cv2.error:
                    alpha = (mask.astype(np.float32)/255.0)[:, :, None]
                    result = (alpha * warped_ct + (1.0 - alpha) * img_t).astype(np.uint8)
        else:
            alpha = (mask.astype(np.float32)/255.0)[:, :, None]
            result = (alpha * warped_ct + (1.0 - alpha) * img_t).astype(np.uint8)

        return _save_or_stream_png(result, "swap_v2", download)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Align/merge v2 error: {e}")

@app.post("/infer/swap_inswapper")
async def swap_inswapper(
    template: UploadFile = File(...),
    portrait: UploadFile = File(...),
    download: int = Form(0),
):
    if face_app is None:
        raise HTTPException(500, "Detector not initialized")

    # <-- add this
    _ensure_inswapper_loaded()
    if inswapper_model is None:
        raise HTTPException(500, "InSwapper model not available (try restarting or check /infer/models)")

    try:
        img_t = _load_image_from_upload(await template.read())
        img_s = _load_image_from_upload(await portrait.read())
        ft, _ = _best_face(img_t)
        fs, _ = _best_face(img_s)
        if ft is None: raise HTTPException(400, "No face found in template")
        if fs is None: raise HTTPException(400, "No face found in portrait")
        if getattr(fs, "normed_embedding", None) is None:
            raise HTTPException(500, "Source face embedding missing")
        swapped = inswapper_model.get(img_t, ft, fs, paste_back=True)
        return _save_or_stream_png(swapped, "swap_inswapper", download)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"InSwapper failed: {e}")

@app.post("/infer/reload_models")
def reload_models():
    global inswapper_model
    inswapper_model = None
    _ensure_inswapper_loaded()
    return {"ok": inswapper_model is not None}