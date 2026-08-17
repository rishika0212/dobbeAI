Subject: Resubmission: Data Scientist Intern Assignment - Dental X-ray Image Segmentation - Rishika

Dear Rishik and the Dobbe AI Team,

I am really sorry about the previous oversight where I misunderstood the requirement and used optical camera photographs of teeth rather than genuine dental radiographs. Thank you for catching that and giving me the opportunity to resubmit.

I have completely redone the pipeline from scratch exclusively on genuine Dental Panoramic (OPG) and Intraoral X-rays. Below is the updated executive summary of the pipeline, results, and deliverables.

EXECUTIVE SUMMARY OF PIPELINE & RESULTS

1. DATASET & EDA:
- Dataset Pool: Curated 3,790 genuine Dental Radiographs (panoramic OPGs & intraoral X-rays) from verified sources including DENTEX 2023 Challenge Radiographs, Dental-Radiography-Teeth-Cavity, and Sudhakar Dental X-Ray Segmentation.
- Resolution & Geometry: Resized to 512x512 with bilinear interpolation (nearest-neighbor for binary ground-truth lesion masks).
- Image Enhancement: Contrast Limited Adaptive Histogram Equalization (CLAHE) applied to address X-ray sensor exposure and density variations across sensor brands.
- Imbalance Mitigation: Foreground cavity/lesion area accounts for < 4.5% of total pixels; handled via custom BCE + Dice combined loss.

2. MODEL ARCHITECTURE & TRAINING:
- Model: U-Net with ResNet34 encoder initialized with ImageNet weights (via segmentation_models_pytorch).
- Loss Function: Combined 0.5 * BCE + 0.5 * Dice Loss.
- Optimization: AdamW (lr=1e-3, weight_decay=1e-4), Cosine Annealing scheduler, and PyTorch Automatic Mixed Precision (AMP).
- Training Convergence: Validation loss steadily decreased from 0.3607 down to 0.1439 across 15 epochs.

3. POST-PROCESSING GUARDRAILS:
- Confidence thresholding at p >= 0.50.
- Morphological closing (3x3 kernel) to bridge interior cavity void holes + opening to detach thin noise boundaries.
- Connected component area filtering (min_area=100 pixels) to eliminate false positive background speckles in jawbone marrow and cervical burnout regions.

4. FULL DATASET EVALUATION (3,790 Images / ~1 Billion Pixels):
---------------------------------------------------------------
- Precision          : 78.84%
- Recall (Sensitivity): 82.49%
- F1-Score / Dice    : 80.62%
- IoU (Jaccard Index): 67.54%
---------------------------------------------------------------
- True Positive Overlap: 38,052,227 pixels
- True Negative Background: 937,181,345 pixels

5. PRODUCTION DEPLOYMENT PROPOSAL:
- Includes an end-to-end FastAPI microservice architecture, ONNX Runtime FP16 / TensorRT optimization (< 35 ms latency/OPG on NVIDIA GPU), Orthanc/DCM4CHEE PACS DICOM integration via WADO-RS, and radiologist-in-the-loop safety guardrails.

DELIVERABLES INCLUDED
1. Resume (Attached PDF)
2. Source Code & Notebook: README.md, train.py, evaluate.py, download_dataset.py, dental_segmentation_pipeline.ipynb, src/
3. Trained Weights: outputs/best_model.pth
4. Visual Results: outputs/eval_results/evaluation_visual_overlay.png and outputs/qualitative_results.png
5. Production Deployment Proposal: production_deployment_proposal.md

INTERVIEW AVAILABILITY
- Weekday Availability (Mon - Fri): 10:00 AM - 1:00 PM / 2:00 PM - 6:00 PM IST
- Weekend Availability (Sat - Sun): 11:00 AM - 4:00 PM IST

GitHub Repository Link: https://github.com/rishika0212/dobbeAI
Trained weights drive link: [best_model.pth](https://drive.google.com/file/d/1Wpm4eMhrZGcBShUNL5Jn3RZ0CaQIQeJ6/view?usp=drive_web)

Thank you again for your time and consideration. I look forward to discussing my technical approach during the interview.

Best regards,
Rishika
9013375466
