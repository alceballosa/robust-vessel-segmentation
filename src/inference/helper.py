import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from utils import execute_and_log, log_to_queue

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS_DIR = (
    REPO_ROOT
    / "atlases_and_weights"
    / "atlases"
    / "rectangle_neck_scene_RegistrationMask"
)
TEMPLATE_PATH = ATLAS_DIR / "template_with_skull.nii.gz"
ROI_TEMPLATE_PATH = ATLAS_DIR / "skull_neck.nii.gz"


def extract_field_from_sidecar(nifti_path, field_name):
    sidecar_path = Path(nifti_path).with_suffix("").with_suffix(".json")
    if not sidecar_path.is_file():
        return None

    with open(sidecar_path, "r") as f:
        sidecar_data = json.load(f)

    if field_name not in sidecar_data:
        return None

    return sidecar_data[field_name]


def parallelize_dataframe(df, func, n_cores=1, log_queue=None):
    df_split = np.array_split(df, n_cores)
    with mp.Pool(n_cores) as pool:
        df = pd.concat(pool.starmap(func, [(split, log_queue) for split in df_split]))
    return df


def make_isotropic(image, interpolator=sitk.sitkLinear, min_spacing=None):
    """Resample an image to isotropic pixels (using smallest spacing from original).

    By default uses a linear interpolator. For label images use
    sitkNearestNeighbor to avoid introducing non-existent labels.
    """
    original_spacing = image.GetSpacing()
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    original_size = image.GetSize()
    if min_spacing is None:
        min_spacing = min(original_spacing)
    new_spacing = [min_spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / min_spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )


def nifti_isotropic(input_image_path, output_image_path, label=False, min_spacing=None):
    image = sitk.ReadImage(input_image_path)
    if label:
        isotropic_image = make_isotropic(
            image, interpolator=sitk.sitkNearestNeighbor, min_spacing=min_spacing
        )
    else:
        isotropic_image = make_isotropic(image, min_spacing=min_spacing)
    sitk.WriteImage(isotropic_image, output_image_path)


def mask_volumes(mask_vol, data_vol, output_vol):
    mask_img = nib.load(mask_vol)
    data_img = nib.load(data_vol)

    mask_data = np.where(mask_img.get_fdata() > 0, 1, 0)
    masked_data = mask_data * data_img.get_fdata()

    masked_img = nib.Nifti1Image(masked_data, data_img.affine, data_img.header)
    nib.save(masked_img, output_vol)


def gather_nifti_data(root_directory):
    data = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(".nii.gz"):
                file_path = str(Path(dirpath) / filename)
                nifti_img = nib.load(file_path)
                data.append(
                    {
                        "patientID": Path(dirpath).name,
                        "file_name": filename,
                        "dimensions": nifti_img.shape,
                        "file_path": file_path,
                        "XRayExposure": extract_field_from_sidecar(
                            file_path, "XRayExposure"
                        ),
                        "AcquisitionTime": extract_field_from_sidecar(
                            file_path, "AcquisitionTime"
                        ),
                        "SeriesDescription": extract_field_from_sidecar(
                            file_path, "SeriesDescription"
                        ),
                        "AcquisitionNumber": extract_field_from_sidecar(
                            file_path, "AcquisitionNumber"
                        ),
                    }
                )
    return pd.DataFrame(data)


def worker_registration(df, log_queue):
    return df.apply(process_row_registration, axis=1, args=(log_queue,))


def worker_resampling(df, log_queue):
    return df.apply(process_row_resampling, axis=1, args=(log_queue,))


def process_row_resampling(row, log_queue):
    try:
        file_path = row["file_path"]
        inference_out_dir = row["inferenceOutDir"]

        cta_vessel_mask_raw = f"{inference_out_dir}{row['file_name_internal']}"
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"The input CTA image {file_path} not found for resampling to patient space."
            )
        if not Path(cta_vessel_mask_raw).exists():
            raise FileNotFoundError(
                f"The predicted vessel mask {cta_vessel_mask_raw} does not exist."
            )
        print(f"{row['cta_predicted']}")
        resampling_command = (
            f"antsApplyTransforms -d 3"
            f" -i {cta_vessel_mask_raw}"
            f" -r {file_path}"
            f" -o {row['cta_predicted']}"
            f" -n NearestNeighbor"
            f" -t identity"
        )
        return execute_and_log(resampling_command, log_queue)
    except FileNotFoundError as e:
        log_to_queue(log_queue, str(e), level=logging.ERROR)
        raise
    except Exception as e:
        log_to_queue(
            log_queue,
            f"Unexpected error during resampling to patient space: {e}",
            level=logging.ERROR,
        )
        raise


def register_images(
    fixed_image_path,
    moving_image_path,
    output_prefix,
    registration_settings_path,
    log_queue,
):
    try:
        affine_mat = f"{output_prefix}0GenericAffine.mat"
        if Path(affine_mat).is_file():
            log_to_queue(
                log_queue,
                f"Affine transformation matrix for {fixed_image_path} found. Skipping to the transformation step.",
            )
            return affine_mat, 55

        log_to_queue(log_queue, f"Starting affine registration for {fixed_image_path}")
        with open(registration_settings_path, "r") as json_file:
            reg_args = json.load(json_file)

        fixed_image = ants.image_read(fixed_image_path)
        moving_image = ants.image_read(moving_image_path)

        start_time = time.time()
        ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform=reg_args["type_of_transform"],
            outprefix=output_prefix,
            flow_sigma=reg_args["flow_sigma"],
            total_sigma=reg_args["total_sigma"],
            aff_metric=reg_args["aff_metric"],
            aff_iterations=tuple(reg_args["aff_iterations"]),
            aff_shrink_factors=tuple(reg_args["aff_shrink_factors"]),
            aff_smoothing_sigmas=tuple(reg_args["aff_smoothing_sigmas"]),
            verbose=reg_args["verbose"],
        )
        elapsed_time = time.time() - start_time

        log_to_queue(
            log_queue, f"The registration took {elapsed_time} seconds to complete."
        )
        return affine_mat, elapsed_time
    except Exception as e:
        log_to_queue(log_queue, f"Error during registration: {e}", level=logging.ERROR)
        raise


def apply_transform(
    fixed_image_path,
    moving_image_path,
    transform_mat_file,
    output_image_path,
    log_queue,
):
    try:
        fixed_image = ants.image_read(fixed_image_path)
        moving_image = ants.image_read(moving_image_path)
        registered_image = ants.apply_transforms(
            fixed_image, moving_image, transform_mat_file
        )
        registered_image.to_file(output_image_path)
    except Exception as e:
        log_to_queue(
            log_queue, f"Error while applying registration: {e}", level=logging.ERROR
        )
        raise


def mask_and_resample_images(
    output_warp_roi, file_path, cta_roi, cta_roi_iso, log_queue, mask=True
):
    try:
        if mask and output_warp_roi is not None:
            log_to_queue(log_queue, f"Masking {file_path} with ROI.")
            mask_volumes(output_warp_roi, file_path, cta_roi)
        else:
            cta_roi = file_path
        log_to_queue(log_queue, f"Resampling to isotropic space for {file_path}.")
        nifti_isotropic(cta_roi, cta_roi_iso, min_spacing=0.468)
    except Exception as e:
        log_to_queue(
            log_queue,
            f"Error during image masking and resampling: {e}",
            level=logging.ERROR,
        )
        raise


def process_row_registration(row, log_queue):
    try:
        cta_name = row["file_name"]
        cta_path = row["file_path"]
        analysis_dir = row["analysisDir"]
        registration_settings_path = row["registration_settings_path"]
        template = str(TEMPLATE_PATH)
        roi_template = str(ROI_TEMPLATE_PATH)

        output_prefix = f"{analysis_dir}{cta_name}_"
        output_warp_roi = f"{analysis_dir}{cta_name}_roi.nii.gz"
        cta_roi = f"{analysis_dir}{cta_name}_cta_inference.nii.gz"
        cta_roi_iso = row["cta_roi_iso"]

        if row["mode"] == "Prediction":
            log_to_queue(log_queue, f"Resampling CTA {cta_name} into {cta_roi_iso}.")
            mask_and_resample_images(
                None, cta_path, cta_roi, cta_roi_iso, log_queue, mask=False
            )
            return 60, None
        else:
            log_to_queue(log_queue, f"Registering {cta_name}")
            affine_matrix, elapsed_time = register_images(
                cta_path, template, output_prefix, registration_settings_path, log_queue
            )

            log_to_queue(log_queue, "Applying transform on ROI image.")
            apply_transform(
                cta_path, roi_template, affine_matrix, output_warp_roi, log_queue
            )

            log_to_queue(log_queue, "Masking CTA image and resampling.")
            mask_and_resample_images(
                output_warp_roi, cta_path, cta_roi, cta_roi_iso, log_queue
            )

        return elapsed_time, affine_matrix
    except Exception as e:
        log_to_queue(
            log_queue, f"Error processing row {cta_name}: {e}", level=logging.ERROR
        )
        return None, None


def mutual_information(hgram):
    """Compute mutual information for joint histogram."""
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def compute_mutual_information(fixed_image_path, moving_image_path, bins=20):
    """Compute mutual information between two 3D NIfTI images."""
    fixed_img = nib.load(fixed_image_path).get_fdata()
    moving_img = nib.load(moving_image_path).get_fdata()
    hist_2d, _, _ = np.histogram2d(fixed_img.ravel(), moving_img.ravel(), bins=bins)
    return mutual_information(hist_2d)


def update_json_file(file_path, field_name, new_value, log_queue):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        data[field_name] = new_value
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        log_to_queue(
            log_queue,
            f"Error updating registration settings file: {file_path}",
            level=logging.ERROR,
        )


def process_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.rename(columns={0: "file_path"}, inplace=True)
    return df


def update_input_dataframe_fields(
    df,
    output_path,
    analysis_dir,
    inference_dir,
    inference_out_dir,
    registration_settings_path_local,
    mode,
    log_queue,
):
    try:
        df["file_name"] = df["file_path"].apply(
            lambda x: Path(x).stem.replace(".nii", "")
        )
        df["analysisDir"] = analysis_dir
        df["inferenceDir"] = inference_dir
        df["inferenceOutDir"] = inference_out_dir
        df["registration_settings_path"] = registration_settings_path_local
        df["mode"] = mode
        df["file_name_internal_inference"] = [
            f"CA_{x:05d}_0000.nii.gz" for x in df.index
        ]
        df["file_name_internal"] = [f"CA_{x:05d}.nii.gz" for x in df.index]
        df["cta_roi_iso"] = df.apply(
            lambda row: f"{row['inferenceDir']}{row['file_name_internal_inference']}",
            axis=1,
        )
        df["cta_predicted"] = df.apply(
            lambda row: f"{row['inferenceOutDir']}{row['file_name']}.nii.gz", axis=1
        )
    except Exception as e:
        log_to_queue(
            log_queue, "Error updating input dataframe fields.", level=logging.ERROR
        )

    return df
