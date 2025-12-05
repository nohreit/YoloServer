from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn, io, time, cv2, numpy as np
from ultralytics import YOLO
from pathlib import Path

app = FastAPI(title="LAN Object Detection")
model = YOLO("yolov8n.pt")  # swap later for a faster/better model
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
LAST_ANN = OUT_DIR / "last_annotated.jpg"


class Det(BaseModel):
    label: str
    conf: float
    box: list  # [x1,y1,x2,y2]


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    t0 = time.time()
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]

    results = model(frame, imgsz=640, conf=0.25, verbose=False)[0]
    detections = []
    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        label = results.names[cls]
        conf = float(b.conf[0])
        detections.append(
            {
                "label": label,
                "conf": round(conf, 4),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            }
        )
        # draw box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (int(x1), max(0, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # write annotated preview for the UI
    cv2.imwrite(str(LAST_ANN), frame)
    return JSONResponse(
        {
            "detections": detections,
            "latency_ms": int((time.time() - t0) * 1000),
            "img_w": w,
            "img_h": h,
        }
    )


@app.get("/preview.jpg")
def preview():
    if LAST_ANN.exists():
        return FileResponse(str(LAST_ANN), media_type="image/jpeg")
    return JSONResponse({"error": "no preview yet"}, status_code=404)


@app.get("/ui", response_class=HTMLResponse)
def ui():
    # ultra-simple page that auto-refreshes the preview + lists detections
    return """<!doctype html><html><head><meta charset="utf-8">
<title>LAN Object Detection</title>
<style>body{font-family:system-ui;margin:20px} img{max-width:90vw;height:auto;border:1px solid #222;border-radius:10px}</style>
</head><body>
<h2>Server Preview</h2>
<img id="prev" src="/preview.jpg" onerror="this.src='';" />
<pre id="det"></pre>
<script>
async function tick(){
  try{
    const r = await fetch('/preview.jpg?ts='+Date.now());
    if(r.ok){
      document.getElementById('prev').src='/preview.jpg?ts='+Date.now();
    }
    const d = await fetch('/last.json?ts='+Date.now());
    if(d.ok){ document.getElementById('det').textContent=JSON.stringify(await d.json(),null,2); }
  }catch(e){}
  setTimeout(tick, 500);
}
tick();
</script>
</body></html>"""


_last = {"detections": [], "latency_ms": 0, "img_w": 0, "img_h": 0}


# small hook to persist the last JSON (optional)
@app.post("/infer", include_in_schema=False)
async def _(): ...  # (placeholder to avoid FastAPI route override warning)


# Monkey-patch: wrap the original infer to also save last.json
orig_infer = app.routes[0].endpoint  # first /infer


async def infer_and_persist(file: UploadFile = File(...)):
    res = await orig_infer(file)
    global _last
    _last = res.body.decode()
    with open(OUT_DIR / "last.json", "w") as f:
        f.write(_last)
    return res


app.router.routes[0].endpoint = infer_and_persist


@app.get("/last.json")
def last_json():
    p = OUT_DIR / "last.json"
    if p.exists():
        return FileResponse(str(p), media_type="application/json")
    return JSONResponse({"detections": []})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
