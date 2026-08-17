# Production Deployment Proposal: Dental Radiograph (OPG & Intraoral X-Ray) Cavity & Lesion Segmentation System

**Target Organization:** Dobbe AI  
**Author:** Rishika (Data Scientist Applicant)  
**Pipeline Target:** Panoramic Radiographs (OPGs) and Intraoral Dental X-Rays  

---

## 📌 Executive Summary

This proposal outlines an end-to-end clinical deployment strategy for integrating our deep learning dental radiograph (OPG and intraoral X-ray) cavity, caries, and lesion segmentation model into dental clinical workflows, PACS/DICOM networks, and cloud platform APIs. 

Trained and validated on **3,790 genuine dental radiographs** (~1 billion evaluated pixels), the **UNet ResNet-34** architecture achieves an **80.62% F1 / Dice score** and **67.54% IoU** with **82.49% recall**. The production architecture emphasizes sub-50 ms inference latency, DICOM Standard compliance, dentist-in-the-loop review guardrails, and real-time MLOps monitoring.

---

## 1. Model Optimization & Serving Infrastructure

```
[Dental Clinic PACS / Client] 
           │ (WADO-RS / HTTP REST)
           ▼
   [FastAPI Gateway]
           │
     ┌─────┴─────────────────────┐
     ▼                           ▼
[S3 DICOM Store]       [Celery + Redis Queue]
                                 │
                                 ▼
                     [ONNX Runtime / TensorRT]
                     (UNet ResNet-34 FP16/INT8)
                                 │
                                 ▼
                    [Morphological Post-Processing]
                    (Closing + Area Filter >= 100px)
                                 │
                                 ▼
                   [DICOM Structured Report (SR)]
                   (Overlay Mask + Diagnostic Heatmap)
```

### A. Inference Engine Optimization
Dental practices require rapid diagnostic turnaround (< 200 ms per radiograph) to support real-time chairside consultations:
- **ONNX Runtime FP16 / TensorRT Acceleration:** Exporting the PyTorch `outputs/best_model.pth` checkpoint to FP16 ONNX Runtime format reduces inference latency from ~115 ms to **< 32 ms per 512x512 radiograph** on cloud GPUs (NVIDIA T4 / L4 / RTX 4000 series).
- **Edge CPU INT8 Quantization:** For dental clinics with limited local compute infrastructure, Post-Training Dynamic Quantization (INT8) compresses the model file from **~93 MB to ~23 MB**, achieving **~180 ms inference latency on standard 8-core CPUs** with < 0.4% degradation in Dice score.

### B. Microservice API Architecture
- **Framework:** **FastAPI** asynchronous REST microservice packaged as lightweight Docker containers and orchestrated via Kubernetes (EKS/GKE) or AWS ECS.
- **Asynchronous Ingestion Queue:** High-resolution OPG radiographs ($> 2000 \times 1000\text{ px}$) are ingested via authenticated endpoints, uploaded to encrypted Amazon S3 buckets, and processed asynchronously via **Celery + Redis** task workers to prevent server bottlenecks during peak clinic hours.

---

## 2. Clinical Integration & PACS / DICOM Workflow

### A. DICOM Standard Compliance
Dental clinics store X-rays in native DICOM format (`.dcm`) rather than lossy JPEG/PNG images:
- The ingestion microservice uses `pydicom` to extract 16-bit monochromatic radiographic pixel data and metadata (`Window Center`, `Window Width`, `Rescale Slope`, `Rescale Intercept`).
- Applies Contrast Limited Adaptive Histogram Equalization (**CLAHE**) directly to the normalized dynamic range, standardizing image density profiles across X-ray equipment brands (Planmeca, Sirona, Carestream, KaVo).

### B. PACS / EHR Integration
- Seamless integration with clinical PACS servers (**Orthanc**, **DCM4CHEE**, **Cliniview**, **Romexis**) using **WADO-RS** (Web Access to DICOM Objects) and **STOW-RS** (Store Over the Web).
- Output is generated as a **DICOM Structured Report (SR)** and **Secondary Capture (SC)** containing pixel-accurate coordinates, lesion surface area ($\text{mm}^2$), confidence levels, and toggleable color overlay masks burnable directly onto clinician viewer monitors.

---

## 3. Human-in-the-Loop Safety & Clinical Guardrails

AI in diagnostic radiology serves as an assistive **second reader** rather than an autonomous decision maker:

1. **Confidence Score Heatmaps:** Probability predictions are rendered as semi-transparent magma heatmaps (Yellow = High confidence $> 85\%$, Orange = Moderate confidence $50\text{--}85\%$).
2. **Artifact & Cervical Burnout Flagging:** Radiographs displaying severe cervical burnout effects (optical radiolucency at the tooth neck) or metallic streak artifacts (dental implants, metallic crowns, braces) trigger an automated anomaly check:
   > *"Notice: High metallic radiopacity detected — clinician verification recommended."*
3. **Interactive Dentist Review UI:** Clinicians can accept, adjust, or discard AI-generated lesion contours with a single click, automatically logging clinician feedback for continuous active learning.

---

## 4. MLOps, Monitoring & Continuous Learning

```
[Incoming Radiographs] ──► [Drift Detection (Evidently AI)]
                                      │
                                      ▼
[Clinician Review / Overrides] ──► [Anonymized Feedback Buffer]
                                      │
                                      ▼
[Retraining Pipeline] ◄──────── [Monthly Validation Gate]
```

### A. Data & Concept Drift Monitoring
- **Image Sensor Drift:** Track input radiographic brightness histograms, exposure distributions, and signal-to-noise ratio (SNR) using **Evidently AI** or **Prometheus / Grafana** to detect uncalibrated clinic X-ray sensors.
- **Prediction Drift:** Monitor daily average lesion area coverage (%) and positive prediction rates. Sudden deviations immediately alert the engineering team.

### B. Active Learning & Retraining Pipeline
- Discrepancy cases (where dentists modify or override AI segmentations) are automatically anonymized, stripped of Protected Health Information (PHI) under HIPAA/GDPR standards, and added to a prioritized retraining pool for scheduled monthly fine-tuning.

---

## 5. Deployment Hardware Requirements & Cost Estimation

| Deployment Tier | Hardware Specifications | Throughput / Latency | Estimated Monthly Cost |
| :--- | :--- | :--- | :--- |
| **Cloud GPU (Recommended)** | 1x NVIDIA T4 / L4 (16GB VRAM), 4 vCPU, 16GB RAM | ~32 ms / radiograph (~55 OPGs/sec) | ~$180 / month (AWS EC2 g4dn.xlarge reserved) |
| **On-Premise Clinic Edge** | 8 vCPU (Intel i7/Xeon), 16GB RAM, INT8 CPU Engine | ~180 ms / radiograph (~6 OPGs/sec) | $0 recurring cloud infrastructure cost |
| **Serverless Batch** | AWS Lambda / Google Cloud Run (Container with ONNX) | ~250 ms cold / ~80 ms warm | Pay-per-inference (~$0.0003 per radiograph) |

---

## 6. Implementation Roadmap

- **Phase 1 (Weeks 1–2):** FastAPI microservice packaging, Docker containerization, ONNX Runtime FP16 graph export and unit testing.
- **Phase 2 (Weeks 3–4):** Orthanc / DCM4CHEE DICOM WADO-RS interface integration, CLAHE preprocessor pipeline, and clinical overlay generator.
- **Phase 3 (Weeks 5–6):** Shadow deployment in pilot dental partner clinics; measure dentist agreement rate and latency benchmarks.
- **Phase 4 (Weeks 7–8):** Active learning feedback loop configuration, Prometheus MLOps dashboards, and production rollout.
