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
SIMILARITY_THRESHOLD = 0.74  # Below this, tools are treated as unmatched
UNCERTAINTY_UPPER = 0.83     # Between 0.74 and 0.83 triggers human verification
SHOW_MATCHED_BOXES = False   # Set True to also draw thin green boxes on present tools

# Load Pretrained Deep Learning Models
sam_model = FastSAM("FastSAM-s.pt")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEVICE)

dinov2_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def letterbox_pil(bgr_img):
    """Pads an image to a 1:1 square so thin scissors are never stretched."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    max_dim = max(h, w)
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left

    squared = cv2.copyMakeBorder(
        rgb, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return Image.fromarray(squared)


# -------------------------------------------------------------------------
# Step 1: Multi-Background Reflection Removal & Chromatic Metallic Masking
# -------------------------------------------------------------------------
def fuse_multi_background_images(image_paths):
    frames_bgr = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            frames_bgr.append(img)

    if not frames_bgr:
        raise FileNotFoundError(f"Could not load any valid images from: {image_paths}")

    h, w = frames_bgr[0].shape[:2]
    frames_bgr = [cv2.resize(f, (w, h)) for f in frames_bgr]

    if len(frames_bgr) == 1:
        base = frames_bgr[0]
        lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_chan)
        view_clahe = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.minimum(hsv[:, :, 2], 210)
        view_dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frames_bgr = [base, view_clahe, view_dark]

    stack = np.stack(frames_bgr, axis=0).astype(np.float32)
    reflection_free_bgr = np.percentile(stack, 35, axis=0).astype(np.uint8)

    s_channels = []
    for f in frames_bgr:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        s_channels.append(hsv[:, :, 1].astype(np.float32) / 255.0)

    mean_saturation = np.mean(s_channels, axis=0)
    chromatic_metal_mask = (mean_saturation < 0.38).astype(np.uint8)

    return frames_bgr, reflection_free_bgr, chromatic_metal_mask


# -------------------------------------------------------------------------
# Step 2: Upright Geometric Alignment (PCA Rotation Normalization)
# -------------------------------------------------------------------------
def align_tool_upright(masked_bgr, binary_mask):
    coords = np.column_stack(np.where(binary_mask > 0))
    if len(coords) < 100:
        return masked_bgr

    mean = np.mean(coords, axis=0)
    center = (float(mean[1]), float(mean[0]))

    covariance_matrix = np.cov(coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    primary_vector = eigenvectors[:, -1]
    angle_rad = np.arctan2(primary_vector[0], primary_vector[1])
    angle_deg = np.degrees(angle_rad)

    rotation_angle = angle_deg - 90.0
    h, w = masked_bgr.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale=1.0)

    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_matrix[0, 2] += (new_w / 2.0) - center[0]
    rot_matrix[1, 2] += (new_h / 2.0) - center[1]

    aligned_crop = cv2.warpAffine(
        masked_bgr, rot_matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
    x, y, bw, bh = cv2.boundingRect(gray)
    if bw > 10 and bh > 10:
        aligned_crop = aligned_crop[y:y+bh, x:x+bw]

    return aligned_crop


# -------------------------------------------------------------------------
# Step 3: Multi-Background Consensus Segmentation (FastSAM Ensemble)
# -------------------------------------------------------------------------
def deduplicate_masks(masks, containment_thresh=0.75):
    if len(masks) == 0:
        return []

    areas = [np.sum(m) for m in masks]
    order = np.argsort(areas)[::-1]

    keep_indices = []
    for idx in order:
        m_cand = masks[idx]
        cand_area = areas[idx]
        is_submask = False

        for kept_idx in keep_indices:
            m_kept = masks[kept_idx]
            intersection = np.sum(m_cand & m_kept)
            if (intersection / (cand_area + 1e-5)) > containment_thresh:
                is_submask = True
                break

        if not is_submask:
            keep_indices.append(idx)

    return [masks[i] for i in keep_indices]


def extract_tools_multi_bg(image_paths, min_area_ratio=0.005, max_area_ratio=0.70):
    frames_bgr, reflection_free_bgr, chromatic_metal_mask = fuse_multi_background_images(image_paths)
    h_img, w_img = reflection_free_bgr.shape[:2]
    total_area = h_img * w_img

    views_to_segment = [reflection_free_bgr] + frames_bgr
    candidate_masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for view_bgr in views_to_segment:
        view_rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
        results = sam_model(
            view_rgb, device=DEVICE, retina_masks=True,
            imgsz=1024, conf=0.25, iou=0.6, verbose=False
        )
        prompt_process = results[0]
        if prompt_process.masks is None:
            continue

        raw_masks = prompt_process.masks.data.cpu().numpy()

        for mask_tensor in raw_masks:
            mask = (cv2.resize(mask_tensor, (w_img, h_img)) > 0.5).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_area = np.sum(mask)

            if not (min_area_ratio * total_area <= mask_area <= max_area_ratio * total_area):
                continue

            metal_overlap = np.sum(mask & chromatic_metal_mask) / (mask_area + 1e-5)
            if metal_overlap < 0.45:
                continue
              
            bx, by, bw, bh = cv2.boundingRect(mask)
            if bx > w_img * 0.85 and by > h_img * 0.80 and mask_area < total_area * 0.005:
                continue  

            candidate_masks.append(mask)

    clean_masks = deduplicate_masks(candidate_masks, containment_thresh=0.75)

    # Sort tools left-to-right by their x-coordinate for clean numbering
    clean_masks.sort(key=lambda m: cv2.boundingRect(m)[0])

    tools_data = []
    for idx, mask in enumerate(clean_masks):
        bx, by, bw, bh = cv2.boundingRect(mask)
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

        crop_mask = mask[y1:y2, x1:x2]
        deglared_crop = cv2.bitwise_and(
            reflection_free_bgr[y1:y2, x1:x2],
            reflection_free_bgr[y1:y2, x1:x2],
            mask=crop_mask
        )
        aligned_deglared = align_tool_upright(deglared_crop, crop_mask)

        multi_view_aligned = [aligned_deglared]
        for f_bgr in frames_bgr:
            f_crop = cv2.bitwise_and(f_bgr[y1:y2, x1:x2], f_bgr[y1:y2, x1:x2], mask=crop_mask)
            multi_view_aligned.append(align_tool_upright(f_crop, crop_mask))

        ah, aw = aligned_deglared.shape[:2]
        tools_data.append({
            "id": idx,
            "mask": mask,
            "raw_crop": frames_bgr[0][y1:y2, x1:x2],
            "aligned_bgr": aligned_deglared,
            "multi_view_aligned": multi_view_aligned,
            "aspect_ratio": max(ah, aw) / (min(ah, aw) + 1e-5),
            "bbox": (x1, y1, x2, y2)
        })

    return tools_data, frames_bgr[0]


# -------------------------------------------------------------------------
# Step 4: Multi-View + Tip-Aware Deep Feature Extraction (DINOv2 Ensemble)
# -------------------------------------------------------------------------
@torch.no_grad()
def get_multi_view_embeddings(tool_crops):
    embeddings = []
    for tool in tool_crops:
        view_embeddings = []

        for aligned in tool["multi_view_aligned"]:
            h, w = aligned.shape[:2]

            pil_full = letterbox_pil(aligned)
            v_0 = dinov2_transforms(pil_full)
            v_180 = dinov2_transforms(pil_full.rotate(180))

            cut = max(int(h * 0.35), 16)
            pil_top = letterbox_pil(aligned[:cut, :])
            pil_bot = letterbox_pil(aligned[h - cut:, :])

            end_1 = dinov2_transforms(pil_top)
            end_2 = dinov2_transforms(pil_bot.rotate(180))

            batch = torch.stack([v_0, v_180, end_1, end_2]).to(DEVICE)
            feats = F.normalize(dinov2(batch), dim=-1)

            f_global = torch.max(feats[0:2], dim=0).values
            f_ends = torch.max(feats[2:4], dim=0).values

            f_view = F.normalize(torch.cat([f_global * 0.6, f_ends * 0.4], dim=-1), dim=-1)
            view_embeddings.append(f_view)

        multi_bg_stack = torch.stack(view_embeddings, dim=0)
        f_consensus = F.normalize(torch.mean(multi_bg_stack, dim=0), dim=-1).cpu().numpy()
        embeddings.append(f_consensus)

    return np.array(embeddings)


# -------------------------------------------------------------------------
# Step 5: Annotate First Image with Bounding Boxes Around Missing Tools
# -------------------------------------------------------------------------
def draw_missing_boxes_on_first_image(first_img_bgr, tools_a, matched_indices_a, warnings_dict):
    """
    Draws bounding boxes directly on the first (Before) image for all missing tools.
    """
    annotated = first_img_bgr.copy()
    overlay = first_img_bgr.copy()
    missing_count = 0

    for idx_a, tool in enumerate(tools_a):
        x1, y1, x2, y2 = tool["bbox"]
        mask_bool = tool["mask"] > 0

        if idx_a not in matched_indices_a:
            missing_count += 1
            color = (0, 0, 255)  # Red in BGR for Missing Tools
            label = f"MISSING #{missing_count}"

            # Soft red tint over the missing instrument's segmentation mask
            overlay[mask_bool] = (
                overlay[mask_bool] * 0.65 + np.array([0, 0, 255]) * 0.35
            ).astype(np.uint8)

            # Draw thick Red Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=3)

            # Draw filled label banner at top of the box
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            label_y1 = max(0, y1 - th - 10)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 12, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
            )

        elif idx_a in warnings_dict:
            # Yellow box if confidence was in the uncertain range
            conf = warnings_dict[idx_a]
            color = (0, 215, 255)
            label = f"VERIFY ({conf*100:.0f}%)"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

        elif SHOW_MATCHED_BOXES:
            color = (0, 200, 0)  # Green in BGR for Matched Tools
            label = "PRESENT"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

    # Blend the mask highlight cleanly with the sharp bounding boxes
    final_img = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)
    # Re-draw crisp boxes/text on top of blended layer
    for idx_a, tool in enumerate(tools_a):
        if idx_a not in matched_indices_a:
            x1, y1, x2, y2 = tool["bbox"]
            cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

    return final_img


# -------------------------------------------------------------------------
# Step 6: Cost Matrix & Hungarian Assignment
# -------------------------------------------------------------------------
def detect_missing_instruments(before_img_paths, after_img_paths, output_dir="missing_output"):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(before_img_paths, str):
        before_img_paths = [before_img_paths]
    if isinstance(after_img_paths, str):
        after_img_paths = [after_img_paths]

    print(f"[*] Segmenting tools from Image A (Before)...")
    tools_a, first_img_bgr = extract_tools_multi_bg(before_img_paths)
    print(f"    Found {len(tools_a)} instruments in Image A.")

    print(f"[*] Segmenting tools from Image B (After)...")
    tools_b, _ = extract_tools_multi_bg(after_img_paths)
    print(f"    Found {len(tools_b)} instruments in Image B.")

    if not tools_a:
        print("[!] No instruments detected in Image A.")
        return None

    matched_indices_a = set()
    warnings_dict = {}

    if tools_b:
        emb_a = get_multi_view_embeddings(tools_a)
        emb_b = get_multi_view_embeddings(tools_b)

        sim_matrix = np.dot(emb_a, emb_b.T)

        cost_matrix = np.zeros_like(sim_matrix)
        for i in range(len(tools_a)):
            for j in range(len(tools_b)):
                ar_diff = abs(tools_a[i]["aspect_ratio"] - tools_b[j]["aspect_ratio"])
                ar_penalty = min(ar_diff / 8.0, 0.15)
                cost_matrix[i, j] = (1.0 - sim_matrix[i, j]) + ar_penalty

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            confidence = sim_matrix[r, c]
            if confidence >= SIMILARITY_THRESHOLD:
                matched_indices_a.add(r)
                if confidence < UNCERTAINTY_UPPER:
                    warnings_dict[r] = confidence

    # Draw bounding boxes around missing tools directly on the first image
    boxed_first_img = draw_missing_boxes_on_first_image(
        first_img_bgr, tools_a, matched_indices_a, warnings_dict
    )

    out_boxed_path = os.path.join(output_dir, "before_missing_boxed.jpeg")
    cv2.imwrite(out_boxed_path, boxed_first_img)

    missing_count = len(tools_a) - len(matched_indices_a)
    print(f"\n[✓] Audit Complete: {missing_count} missing surgical instrument(s) boxed on First Image.")
    print(f"[+] Saved annotated first image to -> {out_boxed_path}")

    return boxed_first_img


if __name__ == "__main__":

    detect_missing_instruments("before_surgery_more.png", "after_surgery_more.png")
