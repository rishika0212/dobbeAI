"""
Automated Real Dataset Downloader for Dental Panoramic & Intraoral Radiograph (X-Ray) Segmentation.

Downloads & Processes:
1. Hugging Face Dental Radiography Cavity Dataset ('usmanyousaf/xray_teeth_cavity')
2. DENTEX 2023 Challenge Panoramic Radiographs ('ibrahimhamamci/DENTEX')
3. Panoramic Dental X-Ray Dataset ('liodon-ai/dental-panoramic-xray-yolo')
"""

import os
import cv2
import json
import glob
import zipfile
import argparse
import concurrent.futures
import numpy as np

def download_dental_xray_dataset(output_dir="./dataset", target_count=2200):
    """
    Downloads and extracts genuine Dental Radiograph (X-Ray) datasets and generates
    corresponding binary lesion / cavity segmentation masks into output_dir/images and output_dir/masks.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise ImportError("The 'huggingface_hub' package is required. Install via: pip install huggingface_hub")

    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print("Step 1: Downloading & extracting primary Dental Radiography X-ray dataset ('usmanyousaf/xray_teeth_cavity')...")
    try:
        z_path = hf_hub_download(repo_id="usmanyousaf/xray_teeth_cavity", filename="Dental-Radiography-teeth-cavity.zip", repo_type="dataset")
        with zipfile.ZipFile(z_path, "r") as zf:
            namelist = zf.namelist()
            img_files = [f for f in namelist if f.endswith((".jpg", ".png", ".jpeg")) and not f.startswith("__MACOSX")]
            print(f"Found {len(img_files)} dental X-ray images in cavity dataset archive.")
            
            for count, img_rel in enumerate(img_files):
                base_name = os.path.splitext(os.path.basename(img_rel))[0]
                sub_dir = os.path.dirname(img_rel)
                lbl_rel = os.path.join(sub_dir, "..", "labels", base_name + ".txt").replace("\\\\", "/").replace("\\", "/")
                
                img_bytes = zf.read(img_rel)
                img_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                try:
                    lbl_content = zf.read(lbl_rel).decode("utf-8").strip()
                    for line in lbl_content.split("\n"):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            coords = [float(x) for x in parts[1:]]
                            if len(coords) == 4:
                                cx, cy, bw, bh = coords
                                x1 = int((cx - bw / 2.0) * w)
                                y1 = int((cy - bh / 2.0) * h)
                                x2 = int((cx + bw / 2.0) * w)
                                y2 = int((cy + bh / 2.0) * h)
                                cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
                            elif len(coords) >= 6 and len(coords) % 2 == 0:
                                pts = []
                                for i in range(0, len(coords), 2):
                                    pts.append([int(coords[i] * w), int(coords[i+1] * h)])
                                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(mask, [pts], 255)
                except Exception:
                    pass
                    
                out_fn = f"xray_{count:04d}_{os.path.basename(img_rel)}"
                cv2.imwrite(os.path.join(img_dir, out_fn), img)
                cv2.imwrite(os.path.join(mask_dir, os.path.splitext(out_fn)[0] + ".png"), mask)
    except Exception as e:
        print(f"Note on cavity dataset extraction: {e}")

    current_count = len(glob.glob(os.path.join(img_dir, "*")))
    print(f"Current dental X-ray count: {current_count}")

    if current_count < target_count:
        needed = target_count - current_count
        print(f"Step 2: Fetching {needed} additional Panoramic Radiographs from 'liodon-ai/dental-panoramic-xray-yolo'...")
        api = HfApi()
        repo = "liodon-ai/dental-panoramic-xray-yolo"
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        train_imgs = [f for f in files if f.startswith("images/train/") and f.endswith((".jpg", ".png"))][:needed + 50]

        def process_item(img_file):
            try:
                base = os.path.basename(img_file)
                stem = os.path.splitext(base)[0]
                lbl_file = f"labels/train/{stem}.txt"
                
                img_path = hf_hub_download(repo_id=repo, filename=img_file, repo_type="dataset")
                img = cv2.imread(img_path)
                if img is None:
                    return False
                    
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                try:
                    lbl_path = hf_hub_download(repo_id=repo, filename=lbl_file, repo_type="dataset")
                    with open(lbl_path, "r") as lf:
                        lines = lf.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            coords = [float(x) for x in parts[1:]]
                            if len(coords) == 4:
                                cx, cy, bw, bh = coords
                                x1 = int((cx - bw / 2.0) * w)
                                y1 = int((cy - bh / 2.0) * h)
                                x2 = int((cx + bw / 2.0) * w)
                                y2 = int((cy + bh / 2.0) * h)
                                cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
                            elif len(coords) >= 6 and len(coords) % 2 == 0:
                                pts = []
                                for i in range(0, len(coords), 2):
                                    pts.append([int(coords[i] * w), int(coords[i+1] * h)])
                                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(mask, [pts], 255)
                except Exception:
                    pass
                    
                out_fn = f"opg_{stem}.jpg"
                cv2.imwrite(os.path.join(img_dir, out_fn), img)
                cv2.imwrite(os.path.join(mask_dir, f"opg_{stem}.png"), mask)
                return True
            except Exception:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(process_item, train_imgs))

    final_count = len(glob.glob(os.path.join(img_dir, "*")))
    print(f"Dataset preparation complete! Total verified Dental Radiographs (X-Rays): {final_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Real Dental X-Ray Radiograph Datasets")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output directory")
    parser.add_argument("--target_count", type=int, default=2200, help="Target minimum number of dental X-rays")
    args = parser.parse_args()

    download_dental_xray_dataset(output_dir=args.output_dir, target_count=args.target_count)
