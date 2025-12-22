from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pathlib import Path
import uvicorn, time, cv2, numpy as np
from ultralytics import YOLO

app = FastAPI(title="LAN Object Detection")
model = YOLO("yolov8n.pt")

OUT_DIR = Path("server/out")
OUT_DIR.mkdir(exist_ok=True)

LAST_ANN = OUT_DIR / "last_annotated.jpg"
LAST_JSON = OUT_DIR / "last.json"

# ---- Runtime state for UI ----
state = {
    "detections": [],
    "latency_ms": 0,
    "img_w": 0,
    "img_h": 0,
    "ts": 0.0,
    "overlay": True,
    "conf_threshold": 0.25,
    "preview_version": 0,
    "topk": [],  # new: top-3 guesses [{label, conf}, ...]
}

# ---- Overlay style ----
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
THICKNESS = 3
TEXT_COLOR = (0, 255, 0)  # green
BG_COLOR = (45, 45, 45)  # dark grey (BGR)
ALPHA = 0.6
PAD_X, PAD_Y = 6, 4


def draw_label_with_bg(frame, text: str, x: int, y: int):
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)

    x1b = max(0, x - PAD_X)
    y1b = max(0, y - text_h - baseline - PAD_Y)
    x2b = min(frame.shape[1] - 1, x + text_w + PAD_X)
    y2b = min(frame.shape[0] - 1, y + PAD_Y)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1b, y1b), (x2b, y2b), BG_COLOR, -1)
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, dst=frame)

    cv2.putText(
        frame, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA
    )


def compute_topk(detections, k=3):
    """Top-k guesses by confidence (label+conf only)."""
    dets_sorted = sorted(detections, key=lambda d: d.get("conf", 0), reverse=True)
    top = dets_sorted[:k]
    return [{"label": d["label"], "conf": d["conf"]} for d in top]


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    t0 = time.time()
    img_bytes = await file.read()

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    h, w = frame.shape[:2]

    results = model(frame, imgsz=640, conf=state["conf_threshold"], verbose=False)[0]

    detections = []
    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        label = results.names[cls]
        conf = float(b.conf[0])

        det = {
            "label": label,
            "conf": round(conf, 4),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        }
        detections.append(det)

        if state["overlay"]:
            cv2.rectangle(
                frame,
                (det["box"][0], det["box"][1]),
                (det["box"][2], det["box"][3]),
                (0, 255, 0),
                2,
            )

            text = f"{label} {conf:.2f}"
            x = det["box"][0]
            y = det["box"][1] - 10

            (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
            if y - (th + baseline + 2 * PAD_Y) < 0:
                y = det["box"][1] + th + baseline + 2 * PAD_Y + 2

            draw_label_with_bg(frame, text, x, y)

    latency_ms = int((time.time() - t0) * 1000)

    # Persist annotated preview for the UI
    cv2.imwrite(str(LAST_ANN), frame)

    # Update state + version counter (used by UI to refresh only on new frames)
    state["preview_version"] += 1
    topk = compute_topk(detections, k=3)
    state.update(
        {
            "detections": detections,
            "latency_ms": latency_ms,
            "img_w": w,
            "img_h": h,
            "ts": time.time(),
            "topk": topk,
        }
    )

    payload = {
        "detections": detections,
        "latency_ms": latency_ms,
        "img_w": w,
        "img_h": h,
        "ts": state["ts"],
        "preview_version": state["preview_version"],
        "topk": topk,
    }

    # Persist JSON (optional)
    LAST_JSON.write_text(JSONResponse(payload).body.decode("utf-8"), encoding="utf-8")

    return JSONResponse(payload)


@app.get("/preview.jpg")
def preview():
    if LAST_ANN.exists():
        return FileResponse(str(LAST_ANN), media_type="image/jpeg")
    return JSONResponse({"error": "no preview yet"}, status_code=404)


@app.get("/status")
def status():
    return JSONResponse(
        {
            "latency_ms": state["latency_ms"],
            "detections_count": len(state["detections"]),
            "img_w": state["img_w"],
            "img_h": state["img_h"],
            "overlay": state["overlay"],
            "conf_threshold": state["conf_threshold"],
            "ts": state["ts"],
            "preview_version": state["preview_version"],
            "topk": state["topk"],  # new
        }
    )


@app.post("/config")
async def config(cfg: dict):
    if "threshold" in cfg:
        t = float(cfg["threshold"])
        state["conf_threshold"] = max(0.0, min(1.0, t))
    if "overlay" in cfg:
        state["overlay"] = bool(cfg["overlay"])
    return JSONResponse(
        {
            "ok": True,
            "conf_threshold": state["conf_threshold"],
            "overlay": state["overlay"],
        }
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("server/static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
