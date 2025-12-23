# YoLoServer

**Distributed Vision Inference Server (LAN, CPU-Only Baseline)**

---

## Overview

YoLoServer is the **server-side component** of a distributed real-time vision system.
It receives image frames from a LAN-connected Android client, performs **CPU-only deep learning inference**, and exposes live diagnostics through a lightweight web dashboard.

The system is intentionally designed with **clear separation between capture and inference**, mirroring real-world edge–server vision pipelines used in robotics and embedded AI.

This repository represents a **stable, submission-ready baseline**.

---

## System at a Glance

* **Android client**

  * Captures camera frames using CameraX
  * Performs lightweight preprocessing and JPEG compression
  * Sends frames to the server over the local network (LAN)

* **YoLoServer**

  * Receives frames via HTTP
  * Runs deep learning inference (CPU-only)
  * Produces detection metadata and annotated previews
  * Hosts a browser-based visualization and diagnostics dashboard

* **Browser (LAN)**

  * Displays the most recent annotated frame
  * Shows detection outputs and basic performance statistics

---

## Architecture (Baseline)

```text
[Android CameraX]
      |
      |  JPEG frames (HTTP POST)
      v
[FastAPI + YOLO (CPU-only)]
      |
      ├─ Annotated preview image
      ├─ Detection metadata (JSON)
      |
[Browser-based Dashboard]
```

---

## Design Principles

* **Distributed processing**
  Capture and inference are decoupled across devices.

* **CPU-only inference**
  Ensures portability, reproducibility, and minimal system dependencies.

* **LAN-first operation**
  No cloud services, no external APIs, no internet requirement.

* **Diagnostic-first mindset**
  Real-time visualization and performance metrics are core features.

* **Baseline stability**
  The system is intentionally simple, inspectable, and frozen.

---

## Technology Stack

### Server

* **Language:** Python 3.11
* **Framework:** FastAPI
* **Model:** Ultralytics YOLO (small model, CPU execution)
* **Image processing:** OpenCV
* **Execution mode:** CPU-only (no CUDA, no GPU dependency)

### Client

* **Platform:** Android
* **Camera API:** CameraX
* **Transport:** HTTP (JPEG multipart upload)

---

## API Summary

### `POST /infer`

* Accepts a JPEG image
* Returns:

  * Detected object labels
  * Confidence scores
  * Bounding box coordinates
  * Inference latency
  * Image dimensions

### Dashboard Endpoints

* **Live preview:** Most recent annotated frame
* **Diagnostics:** Latest detection metadata and performance stats
* **UI:** Lightweight browser-accessible monitoring page

---

## Performance Characteristics

* Designed for **real-time operation over LAN**
* Typical input resolution: 640×360 or 640×480
* Stable operation at ~5–10 FPS (CPU-dependent)
* Inference latency varies with model and system load

---

## Security & Privacy

* Intended for **local network use only**
* No cloud integration
* No long-term image storage
* If exposed externally, authentication and HTTPS should be added (out of scope)

---

## Scope & Status

This repository represents a **locked, submission-stable baseline**.

The following are **intentionally excluded**:

* GPU / CUDA acceleration
* Streaming transports (WebSocket, WebRTC)
* Model training or fine-tuning
* Reinforcement learning
* Production hardening or scaling

These are potential future extensions and are not required for the core contribution.

---

## Related Repository

* **Android Client:** [https://github.com/nohreit/YoloApp](https://github.com/nohreit/YoloApp)

---

## License

This project is provided for educational and research purposes.

