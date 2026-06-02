import os
import json
import numpy as np
import nibabel as nib
from PIL import Image

def process_and_export_nifti(nii_path, output_dir, prefix="scan", is_mask=False):
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"Could not find NIfTI file at: {nii_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {nii_path}...")
    
    img = nib.load(nii_path)
    volume_data = img.get_fdata()
    
    # Transpose to (Z, Y, X)
    volume_data = np.transpose(volume_data, (2, 1, 0))
    num_slices = volume_data.shape[0]
    
    # Pre-calculate anatomy scan normalization if it's not a mask
    if not is_mask:
        p_min, p_max = np.percentile(volume_data, (1.0, 99.0))
        volume_clipped = np.clip(volume_data, p_min, p_max)
        volume_normalized = (volume_clipped - p_min) / (p_max - p_min) * 255.0
        volume_uint8 = volume_normalized.astype(np.uint8)
    
    print(f"Exporting {num_slices} slices to '{output_dir}/'...")
    
    for i in range(num_slices):
        if is_mask:
            # 1. Create a binary mask (0 for background, 255 for anomaly)
            mask_slice = (volume_data[i, :, :] > 0.5).astype(np.uint8) * 255
            
            # 2. Create an RGBA image array (Height, Width, 4 channels: R, G, B, Alpha)
            rgba = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 4), dtype=np.uint8)
            
            # Make the active pixels white (RGB = 255, 255, 255)
            rgba[..., 0:3] = 255  
            
            # Make the background transparent using the mask slice for the Alpha channel
            rgba[..., 3] = mask_slice  
            
            img_out = Image.fromarray(rgba, mode='RGBA')
        else:
            # Standard Grayscale export for anatomy scans
            slice_data = volume_uint8[i, :, :]
            img_out = Image.fromarray(slice_data, mode='L')
        
        # 4-digit zero-padding to support >999 slices safely
        filename = f"{prefix}_{i+1:04d}.png"
        filepath = os.path.join(output_dir, filename)
        img_out.save(filepath)
        
    print(f"✅ Successfully completed export for {prefix}.")
    return num_slices

if __name__ == "__main__":
    
    CBCT_IMAGE_PATH = "/Users/jaden/Work/DentalAI/Showcase/cbct_fractured_001_0000.nii.gz"
    AI_MASK_PATH = "/Users/jaden/Work/DentalAI/Showcase/cbct_fractured_001_TEST1.nii.gz"
    
    CASE_DIR = "/Users/jaden/Work/DentalAI/Showcase/public/data/case_01"
    FRONTEND_IMAGE_DIR = os.path.join(CASE_DIR, "slices")
    FRONTEND_MASK_DIR = os.path.join(CASE_DIR, "masks")

    # Export images and capture total slice count
    total_slices = process_and_export_nifti(
        nii_path=CBCT_IMAGE_PATH,
        output_dir=FRONTEND_IMAGE_DIR,
        prefix="scan",
        is_mask=False
    )
    
    process_and_export_nifti(
        nii_path=AI_MASK_PATH,
        output_dir=FRONTEND_MASK_DIR,
        prefix="mask",
        is_mask=True
    )
    
    # Save the slice count for the frontend to read
    metadata = {"total_slices": total_slices}
    with open(os.path.join(CASE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    print(f"✅ Saved metadata.json with total_slices: {total_slices}")
