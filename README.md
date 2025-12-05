Love this direction, Maelis. Let’s build a **LAN-only Android → server object-detection pipeline** with a clean upgrade path (streaming, on-device fallback, tracking) but start super simple so it actually works on day 1.

# Plan at a glance

* **Android app (Kotlin, CameraX)** grabs frames, compresses to JPEG, sends to LAN server.
* **Server (FastAPI + Ultralytics YOLO on your RTX 3060 Ti)** runs inference and returns JSON detections; also serves a minimal web UI that shows the latest annotated frame + detection table.
* **Transport (MVP):** single HTTP `POST /infer` with the frame as multipart JPEG.
  (Later: WebSocket for streaming or WebRTC if we want low-latency overlays back to the phone.)

---

# Architecture (MVP)

```ts
[Android CameraX] --(JPEG over HTTP POST /infer)--> [FastAPI + YOLO + CUDA]
                                                   ↘ writes last_annotated.jpg
                                                    ↘ emits JSON detections
 [Browser on LAN] <-- GET /ui (HTML+JS)  -----------↙ shows boxes & labels
```

---

# Tech choices (pragmatic)

* **Detector:** Ultralytics YOLO (small model for realtime). Start with CPU okay; switch to CUDA/TensorRT on your 3060 Ti for speed.
* **Backend:** Python 3.11, FastAPI, uvicorn, `ultralytics`, OpenCV (`cv2`), `pydantic`.
* **UI:** A single static HTML page served by FastAPI that refreshes the annotated frame every ~250–500 ms and lists detections.
* **LAN-only:** Bind to your server’s LAN IP; no internet required.

---

# API contract

**Request** `POST /infer` (multipart form)

* `file`: JPEG image bytes

**Response** `200 application/json`

```json
{
  "detections": [
    {"label":"person","conf":0.92,"box":[x1,y1,x2,y2]}
  ],
  "latency_ms": 18,
  "img_w": 1280,
  "img_h": 720
}
```

---

# Server — minimal FastAPI app (drop-in)

> Save as `server/main.py`

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn, io, time, cv2, numpy as np
from ultralytics import YOLO
from pathlib import Path

app = FastAPI(title="LAN Object Detection")
model = YOLO("yolov8n.pt")  # swap later for a faster/better model
OUT_DIR = Path("out"); OUT_DIR.mkdir(exist_ok=True)
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
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        label = results.names[cls]
        conf  = float(b.conf[0])
        detections.append({"label": label, "conf": round(conf, 4), "box": [int(x1),int(y1),int(x2),int(y2)]})
        # draw box
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), max(0, int(y1)-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # write annotated preview for the UI
    cv2.imwrite(str(LAST_ANN), frame)
    return JSONResponse({
        "detections": detections,
        "latency_ms": int((time.time()-t0)*1000),
        "img_w": w, "img_h": h
    })

@app.get("/preview.jpg")
def preview():
    if LAST_ANN.exists():
        return FileResponse(str(LAST_ANN), media_type="image/jpeg")
    return JSONResponse({"error":"no preview yet"}, status_code=404)

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

_last = {"detections":[], "latency_ms":0, "img_w":0, "img_h":0}

# small hook to persist the last JSON (optional)
@app.post("/infer", include_in_schema=False)
async def _(): ...  # (placeholder to avoid FastAPI route override warning)

# Monkey-patch: wrap the original infer to also save last.json
orig_infer = app.routes[0].endpoint  # first /infer
async def infer_and_persist(file: UploadFile = File(...)):
    res = await orig_infer(file)
    global _last; _last = res.body.decode()
    with open(OUT_DIR / "last.json", "w") as f: f.write(_last)
    return res
app.router.routes[0].endpoint = infer_and_persist

@app.get("/last.json")
def last_json():
    p = OUT_DIR / "last.json"
    if p.exists(): return FileResponse(str(p), media_type="application/json")
    return JSONResponse({"detections":[]})
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

### (Optional) Docker Compose (CUDA if you want GPU)

> Save as `docker-compose.yml` next to `server/`

```yaml
services:
  detector:
    build:
      context: ./server
      dockerfile: Dockerfile
    ports: ["7860:7860"]
    volumes:
      - ./server/out:/app/out
    deploy: {}
    # For NVIDIA GPU (requires nvidia-container-toolkit installed):
    # runtime: nvidia
    # environment:
    #   - NVIDIA_VISIBLE_DEVICES=all
    #   - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

**Dockerfile (CPU first; switch to CUDA base later if needed):**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git libgl1 && rm -rf /var/lib/apt/lists/*
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python","server/main.py"]
```

**requirements.txt**

```
fastapi
uvicorn[standard]
ultralytics
opencv-python-headless
```

Run locally (no Docker):
`pip install -r requirements.txt && python server/main.py`
Open `http://<server_lan_ip>:7860/ui`

---

# Android (CameraX → HTTP POST)

**Gradle deps (module):**

```gradle
dependencies {
  def camerax = "1.3.4"
  implementation "androidx.camera:camera-core:$camerax"
  implementation "androidx.camera:camera-camera2:$camerax"
  implementation "androidx.camera:camera-lifecycle:$camerax"
  implementation "androidx.camera:camera-view:$camerax"
  implementation "com.squareup.retrofit2:retrofit:2.11.0"
  implementation "com.squareup.retrofit2:converter-moshi:2.11.0"
  implementation "com.squareup.okhttp3:okhttp:4.12.0"
}
```

**Retrofit interface:**

```kotlin
interface DetectApi {
  @Multipart
  @POST("/infer")
  suspend fun infer(@Part file: MultipartBody.Part): Response<ResponseBody>
}
```

**CameraX + upload (Activity/Fragment gist):**

```kotlin
class MainActivity : AppCompatActivity() {
  private lateinit var api: DetectApi
  private var sending = false
  private val serverUrl = "http://192.168.1.50:7860" // change to your LAN IP

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // setContentView(...) add a PreviewView for camera
    api = Retrofit.Builder()
      .baseUrl(serverUrl)
      .addConverterFactory(MoshiConverterFactory.create())
      .build()
      .create(DetectApi::class.java)

    val previewView = findViewById<PreviewView>(R.id.previewView)
    val cameraProvider = ProcessCameraProvider.getInstance(this)
    cameraProvider.addListener({
      val provider = cameraProvider.get()
      val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
      val analysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()

      analysis.setAnalyzer(Executors.newSingleThreadExecutor()) { img ->
        if (!sending) {
          sending = true
          val jpeg = yuvToJpeg(img)    // ByteArray
          img.close()
          uploadFrame(jpeg).invokeOnCompletion { sending = false }
        } else img.close()
      }

      provider.unbindAll()
      provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
    }, ContextCompat.getMainExecutor(this))
  }

  private fun yuvToJpeg(image: ImageProxy): ByteArray {
    val bitmap = imageProxyToBitmap(image)
    val stream = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
    return stream.toByteArray()
  }

  private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val yuv = YuvImage(
      yDataFrom(image), ImageFormat.NV21, image.width, image.height, null
    )
    val out = ByteArrayOutputStream()
    yuv.compressToJpeg(Rect(0,0,image.width,image.height), 90, out)
    val bytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
  }

  private fun yDataFrom(image: ImageProxy): ByteArray {
    // Convert ImageProxy (YUV_420_888) to NV21 quickly
    val y = image.planes[0].buffer
    val u = image.planes[1].buffer
    val v = image.planes[2].buffer
    val ySize = y.remaining()
    val uSize = u.remaining()
    val vSize = v.remaining()
    val nv21 = ByteArray(ySize + uSize + vSize)
    y.get(nv21, 0, ySize)
    v.get(nv21, ySize, vSize)
    u.get(nv21, ySize + vSize, uSize)
    return nv21
  }

  private fun uploadFrame(bytes: ByteArray) = lifecycleScope.launch(Dispatchers.IO) {
    val body = bytes.toRequestBody("image/jpeg".toMediaType())
    val part = MultipartBody.Part.createFormData("file", "frame.jpg", body)
    try {
      val r = api.infer(part)
      if (r.isSuccessful) {
        val json = r.body()!!.string()
        // (Optional) parse and overlay boxes locally if you want
      }
    } catch (_: Exception) { /* ignore for MVP */ }
  }
}
```

> MVP keeps the overlay on the **server UI**. Later we can return boxes and draw overlays on Android with a simple `View` subclass.

---

# Performance tips (when you’re ready)

* Set Android JPEG to 640×360 or 640×480 to keep POSTs small (<100 KB).
* On server, enable CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu...` and use a faster YOLO variant; consider `model.to('cuda')`.
* Batch frames or throttle to ~5–10 FPS for LAN stability.
* Consider **WebSocket** endpoint that returns detections continuously to cut HTTP overhead.

---

# Security / privacy (LAN-friendly)

* Bind FastAPI to `0.0.0.0` but firewall it to your subnet.
* If you expose beyond LAN, add an auth token and HTTPS (Caddy/Traefik), but for home lab keep it private.

---

# Next steps (immediate)

1. Spin up the server (`pip install …`, run `python server/main.py`), visit `http://<LAN-IP>:7860/ui`—you’ll see the placeholder.
2. Build the Android app, point `serverUrl` at your LAN IP, run it, and watch detections populate on the UI.
3. If performance is sluggish, we’ll flip CUDA on your 3060 Ti and/or switch to a faster YOLO build.

If you want, I can also give you a **CUDA-enabled Dockerfile** and a **Canvas overlay snippet** for Android to draw the boxes from the JSON response.
