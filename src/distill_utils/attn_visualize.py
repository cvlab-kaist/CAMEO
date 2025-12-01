import torch
import numpy as np

from PIL import Image
import cv2
from typing import List
import torchvision.transforms as T
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def stitch_side_by_side_whole(images: List[Image.Image], padding: int = 0, bg_color: tuple = (0, 0, 0)):
    """
    Stitches a list of PIL Images with varying widths horizontally.

    Args:
        images (List[PIL.Image]): A list of images to stitch.
        padding (int): Pixels of space between images.
        bg_color (tuple): Background color (R, G, B) for the canvas.

    Returns:
        PIL.Image: The stitched panoramic image.
    """
    # Fix: Compute canvas size dynamically from the actual images
    # Sum the widths of all images and add the padding space
    total_width = sum(img.width for img in images) + padding * (len(images) - 1)
    # Find the maximum height to ensure all images fit
    max_height = max(img.height for img in images)

    # Create the empty canvas with the correct dimensions
    canvas = Image.new('RGB', (total_width, max_height), color=bg_color)

    # Paste each image onto the canvas
    x_offset = 0
    for img in images:
        # Vertically center the image on the canvas
        y_offset = (max_height - img.height) // 2
        canvas.paste(img, (x_offset, y_offset))
        # Update the horizontal offset for the next image
        x_offset += img.width + padding

    return canvas

def get_attn_map_whole(attn_layer: torch.Tensor, background: np.ndarray) -> np.ndarray:
    """
    Overlay an attention heatmap on a background image and mark the highest attention point.

    Args:
        attn_layer (torch.Tensor): 2D attention map, shape (H_small, W_small), usually values in [0,1].
        background (np.ndarray): Background image, shape (H_large, W_large, 3).
        normalize (bool): 
            - If True, normalize attn_layer to [0,1] using min/max (per-heatmap contrast).
            - If False, assume attn_layer already in [0,1] (absolute scale preserved).

    Returns:
        np.ndarray: Visualization with heatmap and marked point.
    """
    H_small, W_small = attn_layer.shape
    H_large, W_large = background.shape[:2]

    # 1. Resize attn_layer to background size
    resize_transform = T.Resize(
        (H_large, W_large),
        # interpolation=T.InterpolationMode.BILINEAR,
        interpolation=T.InterpolationMode.NEAREST,
        # antialias=True
    )
    attn_resized = resize_transform(attn_layer.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
    attn_resized = attn_resized.squeeze().cpu().detach().numpy()           # [H, W]

    # 2. Optionally normalize
    min_val, max_val = attn_resized.min(), attn_resized.max()
    if max_val > min_val:  # avoid divide-by-zero
        attn_resized = (attn_resized - min_val) / (max_val - min_val)
    else:
        attn_resized = np.zeros_like(attn_resized)

    # 3. Map values to colormap (viridis by default)
    heatmap = (cm.viridis(attn_resized)[:, :, :3] * 255).astype(np.uint8)

    # 4. Blend with background
    background_uint8 = background.astype(np.uint8).copy()
    blended_map = cv2.addWeighted(background_uint8, 0.3, heatmap, 0.7, 0)

    return blended_map # if no marking point

    # 5. Find the max attention location in the original small map
    attn_small = attn_layer.cpu().detach().numpy()
    max_idx_y, max_idx_x = np.unravel_index(np.argmax(attn_small), attn_small.shape)

    # 6. Scale coordinates to background size
    max_x = int((max_idx_x + 0.5) * (W_large / W_small))
    max_y = int((max_idx_y + 0.5) * (H_large / H_small))

    # 7. Draw a circle at the max point
    final_map = cv2.circle(blended_map, (max_x, max_y), 14, (255, 255, 255), -1)
    final_map = cv2.circle(blended_map, (max_x, max_y), 10, (255, 0, 0), -1)

    return final_map

def mark_point_on_img(tgt_img, x, y, radius=6):
    """
    tgt_img: torch.Tensor [3, 512, 512], values in [0,1] or [0,255]
    x, y: coordinates to mark (0–511)
    radius: circle radius
    returns: np.ndarray [512, 512, 3], uint8
    """
    # Convert to numpy HWC uint8, ensure contiguous
    img = tgt_img.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:   # handle [0,1] case
        img = img * 255.0
    img = img.astype(np.uint8)
    img = np.ascontiguousarray(img)

    # Draw white outer circle + blue inner circle
    img = cv2.circle(img, (int(x), int(y)), radius+4, (255, 255, 255), -1)  # white
    img = cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), -1)        # red

    return img

def save_image_total(images, x_coord, y_coord, score):
    """
    images: torch.Tensor [B, V, 3, 512, 512], values in [0,1] or None
    x, y: int coordinates to mark (0–511)
    score: torch.Tensor [HW, VHW]
        returns image to save, np.ndarray [512, 512, 3], uint8
    """
    # Visualization
    images = images.squeeze()
    vis_list = []

    # Target image with marked point
    if images is not None: 
        tgt_image = images[-1]
        tgt_image = mark_point_on_img(tgt_image, x_coord, y_coord)
        vis_list.append(Image.fromarray(tgt_image))

    V_images, _, _, _ = images.shape
    fHW, VfHW = score.shape
    V_score = int(VfHW/fHW)

    if V_images != V_score: # cross_only
        ref_image = images[:-1]
    else:
        ref_image = images
    background = []
    HW, VHW = score.shape
    V = int(VHW/HW)

    for i in range(V):
        background_elem = ref_image[i].squeeze().permute(1,2,0).cpu().detach()
        background.append(background_elem)
    background = torch.stack(background, dim=1).reshape(512, -1, 3).numpy()
    background = np.clip(background * 255.0, 0, 255).astype(np.uint8)
    
    attn_heatmap_img = get_attn_map_whole(score, background,)
    vis_list.append(Image.fromarray(attn_heatmap_img.astype(np.uint8)))
    combined_img = stitch_side_by_side_whole(vis_list)

    return combined_img

def save_image_jinhk(tgt_gt_image, images, target_idx, x_coord, y_coord, score):
    """
    images: torch.Tensor [B, V, 3, 512, 512], values in [0,1] or None
    x, y: int coordinates to mark (0–511)
    score: torch.Tensor [HW, VHW]
        returns image to save, np.ndarray [512, 512, 3], uint8
    """
    # Visualization
    images = images.squeeze()
    vis_list = []

    # Target image with marked point
    if tgt_gt_image is not None:
        tgt_image = tgt_gt_image
    else:
        tgt_image = images[target_idx]
    tgt_image = mark_point_on_img(tgt_image, x_coord, y_coord, radius=14)
    vis_list.append(Image.fromarray(tgt_image))

    V_images, _, _, _ = images.shape
    fHW, VfHW = score.shape
    V_score = int(VfHW/fHW)

    if V_images != V_score: # cross_only
        cross_only = True
        ref_image = torch.stack([images[i] for i in range(V_images) if i != target_idx], dim=0) # shape: [V-1, 3, 512, 512]
    else:
        cross_only = False
        ref_image = images
    background = []
    HW, VHW = score.shape
    V = int(VHW/HW)

    for i in range(V):
        background_elem = ref_image[i].squeeze().permute(1,2,0).cpu().detach()
        background.append(background_elem)
    background = torch.stack(background, dim=1).reshape(512, -1, 3).numpy()
    background = np.clip(background * 255.0, 0, 255).astype(np.uint8)
    
    attn_heatmap_img = get_attn_map_whole(score, background,)
    # insert gray image on the target image index t
    # in the attn_heatmap_img
    if cross_only:
        dummy_img = np.ones((512, 512, 3), dtype=np.uint8) * 127
        attn_heatmap_img = np.concatenate(
            [attn_heatmap_img[:, :512*(target_idx), :],
            dummy_img,
            attn_heatmap_img[:, 512*(target_idx):, :]], axis=1)
    vis_list.append(Image.fromarray(attn_heatmap_img.astype(np.uint8)))
    combined_img = stitch_side_by_side_whole(vis_list)

    return combined_img

def overlay_grid_and_save(img_tensor: torch.Tensor, spacing=64, out_path="grid_overlay.png"):
    """
    Overlays a grid every `spacing` pixels, but only labels the
    x-axis at the top edge and the y-axis at the left edge,
    with labels in black. The image is shown without flipping.
    
    img_tensor: [H, W] or [C, H, W]
    """
    # 1. Convert to H×W or H×W×C numpy
    if img_tensor.ndim == 3:
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
    elif img_tensor.ndim == 2:
        img = img_tensor.cpu().numpy()
    else:
        raise ValueError(f"Unsupported shape {img_tensor.shape}")

    H, W = img.shape[:2]
    xs = np.arange(0, W, spacing)
    ys = np.arange(0, H, spacing)

    # 2. Plot
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=None if img.ndim == 3 else "gray", origin="upper")
    # — no ax.invert_xaxis(), so image is not flipped

    # 3. Draw grid lines
    for y in ys:
        ax.axhline(y, color="white", linewidth=0.8)
    for x in xs:
        ax.axvline(x, color="white", linewidth=0.8)

    # 4. Set ticks at grid lines
    ax.set_xticks(xs)
    ax.set_yticks(ys)

    # 5. Label ticks in black
    ax.set_xticklabels([str(x) for x in xs], color="black", fontsize=8)
    ax.set_yticklabels([str(y) for y in ys], color="black", fontsize=8)

    # 6. Show x labels on top only, y labels on left only
    ax.tick_params(
        axis='x', which='both',
        labelbottom=False,
        labeltop=True,
        bottom=False, top=False
    )
    ax.tick_params(
        axis='y', which='both',
        labelleft=True,
        labelright=False,
        left=False, right=False
    )

    # 7. Remove frame lines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

