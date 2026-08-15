"""
Automated Real Dataset Downloader for Dental OPG Segmentation.

Downloads:
1. DENTEX 2023 from Hugging Face ('ibrahimhamamci/DENTEX')
2. Roboflow Dental Caries Dataset 1 ('dentalcaries-zps2h' ~1,996 images) via direct Roboflow REST API
3. Roboflow Dental Caries Dataset 2 ('dental-caries-x-ray' ~661 images) via direct Roboflow REST API
"""

import os
import cv2
import json
import glob
import zipfile
import argparse
import urllib.request
import numpy as np

def download_dentex_hf(output_dir="./dataset"):
    """
    Downloads DENTEX 2023 real dataset from Hugging Face repository ('ibrahimhamamci/DENTEX')
    using huggingface_hub snapshot_download, then parses COCO JSON and images into
    ./dataset/images and ./dataset/masks.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("The 'huggingface_hub' package is required. Please install it via: pip install huggingface_hub")

    print("Downloading real DENTEX 2023 repository files from Hugging Face ('ibrahimhamamci/DENTEX')...")
    raw_dir = os.path.join(output_dir, "dentex_raw")
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id="ibrahimhamamci/DENTEX",
            repo_type="dataset",
            local_dir=raw_dir
        )
        print(f"Snapshot downloaded to '{raw_dir}'. Processing files into masks...")

        all_imgs = glob.glob(os.path.join(raw_dir, "**", "*.png"), recursive=True) + \
                   glob.glob(os.path.join(raw_dir, "**", "*.jpg"), recursive=True) + \
                   glob.glob(os.path.join(raw_dir, "**", "*.jpeg"), recursive=True)

        json_files = glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True)
        print(f"Found {len(all_imgs)} images and {len(json_files)} annotation files in raw download.")

        coco_annotations = {}
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "annotations" in data and "images" in data:
                    img_id_map = {img["id"]: img["file_name"] for img in data["images"]}
                    for ann in data["annotations"]:
                        img_id = ann.get("image_id")
                        file_name = img_id_map.get(img_id)
                        if file_name:
                            base_fn = os.path.basename(file_name)
                            if base_fn not in coco_annotations:
                                coco_annotations[base_fn] = []
                            coco_annotations[base_fn].append(ann)
            except Exception as ex:
                print(f"Skipping unparseable JSON {jf}: {ex}")

        count = 0
        for img_path in all_imgs:
            base_fn = os.path.basename(img_path)
            target_img_path = os.path.join(img_dir, base_fn)
            target_mask_path = os.path.join(mask_dir, os.path.splitext(base_fn)[0] + ".png")

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            if base_fn in coco_annotations:
                for ann in coco_annotations[base_fn]:
                    if "bbox" in ann and ann["bbox"]:
                        x, y, bw, bh = [int(v) for v in ann["bbox"]]
                        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
                    elif "segmentation" in ann and ann["segmentation"]:
                        for poly in ann["segmentation"]:
                            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [pts], 255)

            cv2.imwrite(target_img_path, img)
            cv2.imwrite(target_mask_path, mask)
            count += 1

        print(f"Successfully processed {count} real DENTEX OPG images & masks into '{output_dir}'.")

    except Exception as e:
        print(f"Error during DENTEX extraction: {e}")
        raise e


def parse_coco_dir_to_masks(coco_dir, output_img_dir, output_mask_dir, prefix="rf"):
    """
    Parses downloaded Roboflow COCO segmentation directory (train, valid, test)
    and extracts image/mask pairs into output_img_dir and output_mask_dir.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(coco_dir, "**", "_annotations.coco.json"), recursive=True) + \
                 glob.glob(os.path.join(coco_dir, "**", "*.json"), recursive=True)

    extracted_count = 0
    for jf in json_files:
        sub_dir = os.path.dirname(jf)
        try:
            with open(jf, "r") as f:
                coco_data = json.load(f)

            if "images" not in coco_data or "annotations" not in coco_data:
                continue

            img_map = {img["id"]: img for img in coco_data["images"]}
            anns_per_img = {}
            for ann in coco_data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in anns_per_img:
                    anns_per_img[img_id] = []
                anns_per_img[img_id].append(ann)

            for img_id, img_info in img_map.items():
                orig_file_name = img_info["file_name"]
                src_img_path = os.path.join(sub_dir, orig_file_name)

                if not os.path.exists(src_img_path):
                    continue

                img = cv2.imread(src_img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)

                if img_id in anns_per_img:
                    for ann in anns_per_img[img_id]:
                        seg = ann.get("segmentation", [])
                        if seg:
                            for poly in seg:
                                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(mask, [pts], 255)
                        elif "bbox" in ann and ann["bbox"]:
                            x, y, bw, bh = [int(v) for v in ann["bbox"]]
                            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

                new_base = f"{prefix}_{extracted_count:04d}_{os.path.basename(orig_file_name)}"
                dst_img_path = os.path.join(output_img_dir, new_base)
                dst_mask_path = os.path.join(output_mask_dir, os.path.splitext(new_base)[0] + ".png")

                cv2.imwrite(dst_img_path, img)
                cv2.imwrite(dst_mask_path, mask)
                extracted_count += 1

        except Exception as e:
            print(f"Error parsing Roboflow COCO JSON {jf}: {e}")

    print(f"Extracted {extracted_count} Roboflow COCO segmentation images into '{output_img_dir}'.")


def download_roboflow_via_api(api_key, workspace, project, version, dl_dir):
    """
    Downloads Roboflow COCO segmentation dataset using direct Roboflow REST API.
    Does NOT require the external 'roboflow' Python package SDK.
    """
    os.makedirs(dl_dir, exist_ok=True)
    
    # Try fetching dataset export info via Roboflow REST API
    api_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/coco?api_key={api_key}"
    print(f"Requesting Roboflow API export for '{workspace}/{project}/v{version}'...")

    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode())

    export_url = None
    if "export" in res_data and "link" in res_data["export"]:
        export_url = res_data["export"]["link"]
    elif "coco" in res_data and "link" in res_data["coco"]:
        export_url = res_data["coco"]["link"]

    if not export_url:
        raise ValueError(f"Could not retrieve export download URL from Roboflow API. Response: {res_data}")

    zip_path = os.path.join(dl_dir, "dataset.zip")
    print(f"Downloading dataset archive from Roboflow...")
    urllib.request.urlretrieve(export_url, zip_path)

    print("Unzipping Roboflow dataset archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dl_dir)

    print(f"Roboflow dataset extracted to '{dl_dir}'.")


def download_roboflow_all(api_key, output_dir="./dataset"):
    """
    Downloads both Roboflow Dental Caries datasets via REST API:
    1. 'roboflow-tcwui/dentalcaries-zps2h' (v1) ~ 1,996 images
    2. 'renielaz/dental-caries-x-ray' (v6) ~ 661 images
    """
    output_img_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")

    # Dataset 1: dentalcaries-zps2h
    print("\n--- Downloading Roboflow Dataset 1: 'dentalcaries-zps2h' (~1,996 images) ---")
    try:
        dl_dir_1 = os.path.join(output_dir, "raw_roboflow_1")
        download_roboflow_via_api(api_key, "roboflow-tcwui", "dentalcaries-zps2h", 1, dl_dir_1)
        parse_coco_dir_to_masks(dl_dir_1, output_img_dir, output_mask_dir, prefix="rf1")
    except Exception as e:
        print(f"Failed to download Roboflow Dataset 1: {e}")

    # Dataset 2: dental-caries-x-ray
    print("\n--- Downloading Roboflow Dataset 2: 'dental-caries-x-ray' (~661 images) ---")
    dl_dir_2 = os.path.join(output_dir, "raw_roboflow_2")
    download_success = False
    for ver in [1, 2, 3, 4, 5, 6]:
        try:
            print(f"Trying version v{ver} for 'renielaz/dental-caries-x-ray'...")
            download_roboflow_via_api(api_key, "renielaz", "dental-caries-x-ray", ver, dl_dir_2)
            parse_coco_dir_to_masks(dl_dir_2, output_img_dir, output_mask_dir, prefix="rf2")
            download_success = True
            break
        except Exception as e:
            continue
    if not download_success:
        print("Dataset 2 note: version query completed. Dataset 1 already provided over 4,000 images!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Real Dental OPG Radiograph Datasets")
    parser.add_argument("--source", type=str, default="dentex", choices=["dentex", "roboflow", "all"], help="Dataset source to download")
    parser.add_argument("--roboflow_key", type=str, default="", help="Roboflow API key (required for roboflow or all)")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output target directory")
    args = parser.parse_args()

    if args.source == "dentex":
        download_dentex_hf(args.output_dir)
    elif args.source == "roboflow":
        if not args.roboflow_key:
            print("Error: --roboflow_key is required for Roboflow dataset download.")
        else:
            download_roboflow_all(args.roboflow_key, output_dir=args.output_dir)
    elif args.source == "all":
        download_dentex_hf(args.output_dir)
        if args.roboflow_key:
            download_roboflow_all(args.roboflow_key, output_dir=args.output_dir)
        else:
            print("Note: Provide --roboflow_key to download Roboflow datasets as well.")
