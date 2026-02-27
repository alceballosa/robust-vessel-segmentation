"""Filter distant connected components from vessel segmentations.

Removes connected components whose center of mass is farther than a
physical distance threshold (in mm) from the largest connected component.
The threshold is resolution-independent: 120 mm by default (equivalent to
300 voxels at 0.4 mm isotropic spacing).

Usage:
    python filter_vessel_components.py <input_dir> <output_dir> [--distance-mm 120] [--workers 4]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm

# 300 voxels * 0.4 mm/voxel = 120 mm
DEFAULT_DISTANCE_MM = 120.0


def filter_distant_components(binary, max_dist_voxels):
    """Remove connected components far from the largest one.

    Finds the largest connected component, computes its center of mass,
    then removes any connected component whose center of mass is more than max_dist_voxels
    voxels away. Returns the filtered binary mask (modified in-place).
    """
    labels, n_cc = ndimage.label(binary)
    if n_cc <= 1:
        return binary

    cc_ids = range(1, n_cc + 1)
    all_coms = ndimage.center_of_mass(binary, labels, cc_ids)
    cc_sizes = ndimage.sum(binary, labels, cc_ids)

    largest_idx = np.argmax(cc_sizes)
    largest_com = np.array(all_coms[largest_idx])

    for i, cc_id in enumerate(cc_ids):
        if i == largest_idx:
            continue
        dist = np.linalg.norm(np.array(all_coms[i]) - largest_com)
        if dist > max_dist_voxels:
            binary[labels == cc_id] = 0

    return binary


def process_file(input_path, output_path, distance_mm):
    """Load a vessel segmentation, filter distant CCs, and save."""
    img = nib.load(input_path)
    data = np.asarray(img.dataobj, dtype=np.uint8)

    # Get voxel spacing — assume isotropic, use first spatial dimension
    spacing_mm = float(img.header.get_zooms()[0])
    max_dist_voxels = distance_mm / spacing_mm

    # Binarize (any label > 0 is vessel)
    binary = (data > 0).astype(np.uint8)

    # Filter distant components
    binary = filter_distant_components(binary, max_dist_voxels)

    # Apply mask to original labels (preserve artery/vein distinction)
    filtered = data * binary

    # Save with same affine and header
    out_img = nib.Nifti1Image(filtered, img.affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, output_path)

    return os.path.basename(input_path), spacing_mm, max_dist_voxels


def main():
    parser = argparse.ArgumentParser(
        description="Filter distant connected components from vessel segmentations."
    )
    parser.add_argument("input_dir", help="Directory containing vessel segmentation NIfTI files.")
    parser.add_argument("output_dir", help="Directory to save filtered segmentations.")
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=DEFAULT_DISTANCE_MM,
        help=f"Max distance (mm) from largest CC center of mass. Default: {DEFAULT_DISTANCE_MM}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Default: 1",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".nii.gz"))
    if not files:
        print(f"No .nii.gz files found in {args.input_dir}")
        return

    print(f"Processing {len(files)} files")
    print(f"Distance threshold: {args.distance_mm} mm")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for fname in files:
                input_path = os.path.join(args.input_dir, fname)
                output_path = os.path.join(args.output_dir, fname)
                future = executor.submit(
                    process_file, input_path, output_path, args.distance_mm
                )
                futures[future] = fname

            for future in tqdm(as_completed(futures), total=len(futures)):
                fname, spacing, dist_vox = future.result()
                tqdm.write(
                    f"  {fname}: spacing={spacing:.3f}mm, "
                    f"dist_threshold={dist_vox:.1f} voxels"
                )
    else:
        for fname in tqdm(files):
            input_path = os.path.join(args.input_dir, fname)
            output_path = os.path.join(args.output_dir, fname)
            fname, spacing, dist_vox = process_file(
                input_path, output_path, args.distance_mm
            )
            tqdm.write(
                f"  {fname}: spacing={spacing:.3f}mm, "
                f"dist_threshold={dist_vox:.1f} voxels"
            )

    print("Done.")


if __name__ == "__main__":
    main()
