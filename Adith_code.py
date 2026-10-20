import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from ultralytics import FastSAM

# -------------------------------------------------------------------------
# Configuration & Hardware
# -------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.72  # Below this, tools are treated as unmatched
UNCERTAINTY_UPPER = 0.82     # Between 0.72 and 0.82 triggers human verification

# 1. Load Pretrained Models
# FastSAM provides zero-shot instance masks; DINOv2 provides self-supervised embeddings
sam_model = FastSAM("FastSAM-s.pt")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEVICE)

# DINOv2 Preprocessing transform
dinov2_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------------------------------------------------------------
# Step 1: Upright Geometric Alignment (PCA Rotation Normalization)
# -------------------------------------------------------------------------
def align_tool_upright(masked_bgr, binary_mask):
    """
    Uses mask contour moments and PCA to find the primary longitudinal axis 
    of the instrument, rotating it into a vertical (canonical) orientation.
    """
    coords = np.column_stack(np.where(binary_mask > 0))
    if len(coords) < 100:
        return masked_bgr

    # Mean center coordinates
    mean = np.mean(coords, axis=0)
    center = (int(mean[1]), int(mean[0]))

    # PCA to get orientation angle
    covariance_matrix = np.cov(coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    primary_vector = eigenvectors[:, -1]  # Dominant axis
    angle_rad = np.arctan2(primary_vector[0], primary_vector[1])
    angle_deg = np.degrees(angle_rad)

    # Rotate upright (align dominant axis to vertical 90 degrees)
    rotation_angle = angle_deg - 90
    h, w = masked_bgr.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale=1.0)
    
    # Calculate bounding size after rotation to avoid clipping
    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    aligned_crop = cv2.warpAffine(
        masked_bgr, rot_matrix, (new_w, new_h), 
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    # Crop out surrounding black border
    gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
    x, y, bw, bh = cv2.boundingRect(gray)
    if bw > 10 and bh > 10:
        aligned_crop = aligned_crop[y:y+bh, x:x+bw]

    return aligned_crop

# -------------------------------------------------------------------------
# Step 2: Tool Isolation & Segmentation
# -------------------------------------------------------------------------
def extract_segmented_tools(image_path, min_area_ratio=0.001, max_area_ratio=0.75):
    """
    Extracts individual tools isolated from background tray glares.
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_bgr.shape[:2]
    total_area = h_img * w_img

    # Run FastSAM to segment all objects
    results = sam_model(img_rgb, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, verbose=False)
    prompt_process = results[0]

    tools_data = []
    if prompt_process.masks is None:
        return tools_data

    masks = prompt_process.masks.data.cpu().numpy()
    boxes = prompt_process.boxes.xyxy.cpu().numpy()

    for idx, (mask_tensor, box) in enumerate(zip(masks, boxes)):
        mask = (cv2.resize(mask_tensor, (w_img, h_img)) > 0.5).astype(np.uint8)
        mask_area = np.sum(mask)

        # Exclude tray edges, blue drapes, or dust spots
        if mask_area < total_area * min_area_ratio or mask_area > total_area * max_area_ratio:
            continue

        x1, y1, x2, y2 = map(int, box)
        crop_bgr = img_bgr[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        # Black-out background cloth/steel reflections
        clean_tool = cv2.bitwise_and(crop_bgr, crop_bgr, mask=crop_mask)

        # Canonical rotation to cancel out arbitrary angles
        aligned_tool = align_tool_upright(clean_tool, crop_mask)

        tools_data.append({
            "id": idx,
            "raw_crop": crop_bgr,
            "aligned_bgr": aligned_tool,
            "aspect_ratio": max(aligned_tool.shape[0], aligned_tool.shape[1]) / (min(aligned_tool.shape[0], aligned_tool.shape[1]) + 1e-5),
            "bbox": (x1, y1, x2, y2)
        })

    return tools_data

# -------------------------------------------------------------------------
# Step 3: Deep Feature Extraction
# -------------------------------------------------------------------------
@torch.no_grad()
def get_embeddings(tool_crops):
    """
    Generates rotation-invariant deep embeddings by pooling upright and 180-degree views.
    """
    embeddings = []
    for tool in tool_crops:
        pil_img = Image.fromarray(cv2.cvtColor(tool["aligned_bgr"], cv2.COLOR_BGR2RGB))
        
        # Test both 0 deg and 180 deg upright orientations
        view_0 = dinov2_transforms(pil_img)
        view_180 = dinov2_transforms(pil_img.rotate(180))

        batch = torch.stack([view_0, view_180]).to(DEVICE)
        features = dinov2(batch)
        features = F.normalize(features, dim=-1)

        # Max-pool views to create an invariant signature
        f_combined = torch.max(features, dim=0, keepdim=True).values
        f_norm = F.normalize(f_combined, dim=-1).squeeze(0).cpu().numpy()
        embeddings.append(f_norm)

    return np.array(embeddings)

# -------------------------------------------------------------------------
# Step 4: Cost Matrix & Hungarian Assignment
# -------------------------------------------------------------------------
def detect_missing_instruments(before_img_path, after_img_path, output_dir="missing_output"):
    os.makedirs(output_dir, exist_ok=True)

    print("[*] Segmenting and aligning tools from Image A (Before)...")
    tools_a = extract_segmented_tools(before_img_path)
    print(f"    Found {len(tools_a)} instruments in Image A.")

    print("[*] Segmenting and aligning tools from Image B (After)...")
    tools_b = extract_segmented_tools(after_img_path)
    print(f"    Found {len(tools_b)} instruments in Image B.")

    if not tools_a:
        print("[!] No instruments detected in Image A.")
        return []
    if not tools_b:
        print("[!] Zero instruments detected in Image B. All instruments missing!")
        return [t["raw_crop"] for t in tools_a]

    # Compute DINOv2 Deep Embeddings
    emb_a = get_embeddings(tools_a)
    emb_b = get_embeddings(tools_b)

    # Semantic similarity matrix
    sim_matrix = np.dot(emb_a, emb_b.T)

    # Cost Matrix incorporating visual cosine distance + aspect ratio penalty
    cost_matrix = np.zeros_like(sim_matrix)
    for i in range(len(tools_a)):
        for j in range(len(tools_b)):
            ar_diff = abs(tools_a[i]["aspect_ratio"] - tools_b[j]["aspect_ratio"])
            # Penalize match if geometry proportions severely disagree
            ar_penalty = min(ar_diff / 5.0, 0.25)
            cost_matrix[i, j] = (1.0 - sim_matrix[i, j]) + ar_penalty

    # Optimal Bipartite Matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_indices_a = set()
    warnings = []

    for r, c in zip(row_ind, col_ind):
        confidence = sim_matrix[r, c]
        if confidence >= SIMILARITY_THRESHOLD:
            matched_indices_a.add(r)
            if confidence < UNCERTAINTY_UPPER:
                warnings.append((r, c, confidence))

    # Identify unmatched items from Before Surgery
    missing_tools = []
    for idx_a, tool in enumerate(tools_a):
        if idx_a not in matched_indices_a:
            missing_tools.append(tool)
            out_path = os.path.join(output_dir, f"missing_tool_{len(missing_tools)}.png")
            cv2.imwrite(out_path, tool["raw_crop"])
            print(f"[-] Missing Tool Detected -> Saved: {out_path}")

    # Zero-Error Protocol / Ambiguity Flag
    if warnings:
        print("\n[!] AMBIGUOUS MATCH WARNING (Manual Review Required):")
        for r, c, conf in warnings:
            print(f"    Tool A#{r} paired with Tool B#{c} with lower confidence: {conf*100:.1f}%")

    print(f"\n[✓] Audit Complete: {len(missing_tools)} missing surgical instrument(s) found.")
    return [t["raw_crop"] for t in missing_tools]

# -------------------------------------------------------------------------
# Execution Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    before_img = "before_surgery.jpeg"
    after_img = "after_surgery.jpeg"
    
    missing = detect_missing_instruments(before_img, after_img)
