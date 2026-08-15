# Production Deployment Proposal: Dental OPG Cavity & Lesion Segmentation System
**Target Organization:** Dobbe AI  
**Author:** Candidate (Data Scientist Applicant)  

---

## Executive Summary
This proposal outlines an end-to-end clinical deployment strategy for integrating our deep learning panoramic X-ray (OPG) cavity and lesion segmentation model into dental clinical workflows and cloud platform APIs. The architecture emphasizes low-latency inference (< 200 ms per 1024x512 OPG), DICOM/PACS compatibility, radiologist safety guardrails, and real-time model monitoring.

---

## 1. Model Optimization & Serving Infrastructure

### A. Inference Engine Optimization
To enable fast CPU or edge GPU execution in dental clinics with limited local compute hardware:
- **ONNX / TensorRT Export:** Convert PyTorch `.pth` model checkpoints to FP16 ONNX runtime graph format. On NVIDIA T4 / RTX 3060 GPUs, TensorRT optimization reduces inference latency from ~120 ms to **~28 ms**.
- **Quantization:** Apply Post-Training Dynamic Quantization (INT8) for CPU-only deployments, reducing model file size by 4x (~85 MB to ~21 MB) with < 0.5% degradation in F1-score.

### B. Microservice API Architecture
- **Framework:** **FastAPI** async RESTful microservice wrapped in Docker containers, orchestrated via Kubernetes (EKS/GKE) or serverless AWS ECS.
- **Asynchronous Task Queue:** High-resolution OPG DICOM images are ingested via API endpoints, uploaded to Amazon S3, and processed asynchronously via **Celery + Redis** task workers to handle clinical peak hours gracefully.

```
[Dental Clinic PACS / Client] 
           │ (WADO-RS / HTTP REST)
           ▼
   [FastAPI Gateway]
           │
     ┌─────┴─────────────────────┐
     ▼                           ▼
[S3 DICOM Store]       [Celery Worker Queue]
                                 │
                                 ▼
                     [ONNX Runtime / TensorRT]
                                 │
                                 ▼
                    [Post-Processing & Mask]
                                 │
                                 ▼
                  [DICOM Structured Report (SR)]
```

---

## 2. Clinical Integration & PACS / DICOM Workflow

### A. DICOM Standard Compliance
Dental clinics store X-rays in DICOM format rather than standard PNGs. The inference microservice uses `pydicom` to extract 16-bit monochromatic pixel data and metadata (Window Center/Width, Rescale Slope/Intercept) to ensure accurate visual calibration across image sensor brands (Planmeca, Sirona, Carestream).

### B. PACS / EHR Integration
- Integration via **Orthanc** or **DCM4CHEE** open-source DICOM servers using **WADO-RS** (Web Access to DICOM Objects) and **STOW-RS** (Store Over the Web).
- Output is formatted as a **DICOM Structured Report (SR)** or **Secondary Capture (SC)** containing pixel coordinates, lesion area (mm²), confidence scores, and overlay masks burnable onto clinical viewer screens (e.g., Cliniview, Romexis).

---

## 3. Human-in-the-Loop Safety & Clinical Guardrails

AI in diagnostic dentistry serves as a **second reader** rather than an autonomous decision maker:
1. **Confidence Score Heatmaps:** Probability predictions are displayed as semi-transparent color overlays (Yellow = High confidence > 85%, Cyan = Moderate confidence 50-85%).
2. **Uncertainty & Anomaly Flagging:** OPG radiographs with metallic streak artifacts (dental implants, metallic crowns, braces) trigger an anomaly score check, displaying a notification: *"High metallic artifact present - manual radiologist verification recommended."*
3. **Interactive Dentist Review:** The web viewer allows clinicians to accept, modify, or reject AI-proposed segmentation contours with a single click, feeding feedback into active learning pipelines.

---

## 4. MLOps, Monitoring & Continuous Learning

### A. Data & Concept Drift Monitoring
- **Image Sensor Drift:** Track pixel brightness histograms and signal-to-noise ratio (SNR) across incoming OPGs via **Evidently AI** or **Prometheus** to detect uncalibrated X-ray equipment.
- **Prediction Drift:** Monitor daily average lesion area coverage (%) and positive prediction rate. Sudden shifts alert the MLOps engineering team.

### B. Retraining Pipeline
- Discrepancy logs (where clinicians override predictions) are automatically anonymized, HIPAA/GDPR sanitized, and queued into a monthly retraining pool for continuous fine-tuning.

---

## 5. Deployment Hardware Requirements

| Deployment Tier | Hardware Specs | Throughput / Latency | Estimated Cost |
| :--- | :--- | :--- | :--- |
| **Cloud GPU (Recommended)** | 1x NVIDIA T4 / L4 (16GB VRAM), 4 vCPU, 16GB RAM | ~35 ms / image (50 OPGs/sec) | ~$0.52 / hour (AWS EC2 g4dn.xlarge) |
| **On-Premise Clinic Edge** | 8 vCPU (Intel i7/Xeon), 16GB RAM, INT8 CPU Execution | ~210 ms / image (5 OPGs/sec) | $0 recurring cloud cost |

---

## 6. Implementation Roadmap

- **Week 1-2:** FastAPI microservice packaging, Dockerization, ONNX model serialization.
- **Week 3-4:** Orthanc DICOM WADO-RS interface integration & clinical overlay renderer.
- **Week 5-6:** Shadow deployment alongside 2 pilot dental practices; compute radiologist agreement rate.
- **Week 7-8:** Active learning feedback loop setup & production release.
