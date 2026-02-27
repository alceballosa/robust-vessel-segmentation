import argparse
import datetime
import logging
import multiprocessing as mp
import os
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from helper import (
    gather_nifti_data,
    parallelize_dataframe,
    process_csv,
    process_row_registration,
    process_row_resampling,
    update_input_dataframe_fields,
    update_json_file,
    worker_registration,
    worker_resampling,
)
from utils import execute_and_log, listener_process, log_to_queue


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
    A deep learning tool to create a binary mask of arteries from a high
    resolution CTA image. Refer to the manuscript for the specifics.
    Familiarity with the ANTs registration toolbox will be helpful but
    not necessary.

    usage with a directory:
        python extractVessels.py -d <input_dir> <output_dir>
    usage with a csv containing nii.gz file paths:
        python extractVessels.py -c <input_csv> <output_dir>"""
        ),
        epilog=textwrap.dedent(
            """\
    outputs:
     This command will create many directories in the output location.
     The relevant files/directories are as follows:
         output_path/Predictions - The nnU-Net model predictions will be
             saved here. Files containing the vessel mask in patient space
             will be saved with the same name as the input CTA images.
         output_path/IntermediateFiles - The files created during the
             registration steps will be saved here.
         output_path/InferenceData - The resampled CT images that will be
             used by the nnU-net model will be saved here.
         output_path/vesselSegmentation_{PIPELINE_MODE}_{dateTime}.log -
             This is the log file of the pipeline.
         output_path/qc_{PIPELINE_MODE}_{dateTime}.csv - csv file
             containing the registration quality control metrics.
         output_path/pipeline_failures_{PIPELINE_MODE}_{dateTime}.csv -
             csv file containing the file paths of failed registrations
             or segmentations along with the reason of the failure.
         output_path/antspy_registration_settings.json - the json file
             containing the arguments that run the registration steps."""
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c",
        "--csv",
        type=str,
        help="Path to the CSV file containing list of CTA NIfTI image paths.",
    )
    group.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Path to the directory containing CTA NIfTI files.",
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the vessel mask.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for brain ROI extraction.",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=int,
        default=241,
        help="The dataset to run.",
    )
    parser.add_argument(
        "-r",
        "--registration_method",
        type=str,
        default="Affine",
        help="Registration method to use in ANTs. Tested options: 'Affine' and 'AffineFast'.",
    )
    parser.add_argument(
        "-g",
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for inference.",
    )
    parser.add_argument(
        "-m",
        "--pipeline_mode",
        type=str,
        default="Full",
        help=(
            "Pipeline mode: 'ROI', 'Prediction', or 'Full'. "
            "'ROI' only registers the template and creates a masked head+neck image. "
            "'Prediction' skips registration and directly runs nnU-Net inference. "
            "'Full' performs both registration and vessel extraction."
        ),
    )
    parser.add_argument(
        "-s",
        "--sliding_window",
        type=float,
        default=0.25,
        help="Sliding window step size for nnU-Net inference. Use -s 1 for testing.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Use batch processing mode (all scans at once). Default is sequential mode with resumability.",
    )

    return parser.parse_args()


def setup_output_directories(output_path):
    output = Path(output_path)
    analysis_dir = output / "IntermediateFiles"
    inference_dir = output / "InferenceData"
    inference_out_dir = output / "Predictions"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)
    inference_out_dir.mkdir(parents=True, exist_ok=True)

    return str(analysis_dir) + "/", str(inference_dir) + "/", str(inference_out_dir) + "/"


def run_pipeline(
    df,
    output_path,
    num_threads,
    num_gpus,
    version,
    log_queue,
    pipeline_failures_csv,
    qc_csv,
    mode,
    sliding_window,
):
    analysis_dir, inference_dir, inference_out_dir = setup_output_directories(
        output_path
    )

    log_to_queue(log_queue, f"There are {len(df)} CTA samples in all.")
    log_to_queue(log_queue, f"Running the pipeline using {num_threads} threads.")

    df["worker_registration"] = parallelize_dataframe(
        df, worker_registration, n_cores=num_threads, log_queue=log_queue
    )
    df[["elapsed_time", "affine_matrix"]] = df["worker_registration"].apply(pd.Series)
    df = df.dropna(subset=["elapsed_time"])

    filtered_df = df[df["elapsed_time"] < 10]
    if len(filtered_df) > 0:
        filtered_df["failure_message"] = "Suspiciously low elapsed time."
        filtered_df[["file_path", "failure_message"]].to_csv(
            pipeline_failures_csv, index=False
        )
        log_to_queue(
            log_queue,
            f"File paths with elapsed time less than 10 seconds have been written to {pipeline_failures_csv}",
        )
    else:
        log_to_queue(log_queue, "No registration took less than 10 seconds.")

    if mode == "ROI":
        log_to_queue(
            log_queue,
            "Pipeline completed. Pass -m 'Full' or -m 'Prediction' for the inference.",
        )
        return

    log_to_queue(
        log_queue,
        f"Starting the inference on all available files under {inference_dir}.",
    )
    inference_command = (
        f"nnUNetv2_predict --continue_prediction"
        f" -p nnUNetResEncUNetLPlans -c 3d_fullres"
        f" -i '{inference_dir}'"
        f" -o '{inference_out_dir}'"
        f" -d {version} -f all -step_size {sliding_window}"
        f" -chk checkpoint_best.pth -device cuda"
    )
    inference_exit = execute_and_log(inference_command, log_queue)
    log_to_queue(log_queue, f"Inference completed. Exit code: {inference_exit}")

    log_to_queue(log_queue, "Starting the resampling step.")
    df["ResamplingExit"] = parallelize_dataframe(
        df, worker_resampling, n_cores=num_threads, log_queue=log_queue
    )
    for result in df["ResamplingExit"]:
        if isinstance(result, Exception):
            log_to_queue(
                log_queue,
                f"An error occurred in a worker process: {result}",
                level=logging.ERROR,
            )

    num_failures = df["ResamplingExit"].sum()
    num_successes = len(df) - num_failures

    log_to_queue(log_queue, f"Number of Successful Resamplings: {num_successes}")

    if num_failures > 0:
        log_to_queue(
            log_queue,
            f"Warning: Unsuccessful ROI transformations for {num_failures} " +
            "samples.",
            level=logging.WARNING,
        )
        log_to_queue(
            log_queue,
            f"Warning: Look at the intermediate files for the following patientIDs: {num_failures}",
            level=logging.WARNING,
        )

    df.to_csv(
        qc_csv,
        sep=",",
        header=True,
        index=False,
        doublequote=False,
        columns=["file_path", "elapsed_time", "ResamplingExit"],
    )


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_pipeline_sequential(
    df,
    output_path,
    version,
    log_queue,
    registration_settings_path_local,
    mode,
    sliding_window,
    gpu_id=0,
):
    output_path = Path(output_path)
    temp_path = output_path.parent / (output_path.name + f"_temp_gpu{gpu_id}")

    log_to_queue(log_queue, f"GPU {gpu_id}: processing {len(df)} scans.")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU {gpu_id}", position=gpu_id):
        scan_name = Path(row["file_path"]).stem.replace(".nii", "")

        # Check if output already exists
        if mode == "ROI":
            output_file = output_path / "ROI" / f"{scan_name}.nii.gz"
        else:
            output_file = output_path / f"{scan_name}.nii.gz"

        if output_file.exists():
            log_to_queue(log_queue, f"Skipping {scan_name}: output already exists.")
            continue

        # Clean up any leftover temp dir from a previous failed run
        if temp_path.exists():
            shutil.rmtree(temp_path)

        # Set up temp directories
        analysis_dir, inference_dir, inference_out_dir = setup_output_directories(
            temp_path
        )

        # Build a 1-row DataFrame for this scan
        scan_df = pd.DataFrame([{"file_path": row["file_path"]}])
        scan_df = update_input_dataframe_fields(
            scan_df,
            str(temp_path),
            analysis_dir,
            inference_dir,
            inference_out_dir,
            str(registration_settings_path_local),
            mode,
            log_queue,
        )
        scan_row = scan_df.iloc[0]

        try:
            # Step 1: Registration / resampling to isotropic
            log_to_queue(log_queue, f"Processing {scan_name}...")
            elapsed_time, _ = process_row_registration(scan_row, log_queue)
            if elapsed_time is None:
                log_to_queue(
                    log_queue,
                    f"Registration/resampling failed for {scan_name}, skipping.",
                    level=logging.ERROR,
                )
                continue

            if mode != "ROI":
                # Step 2: nnUNet inference
                inference_command = (
                    f"nnUNetv2_predict --continue_prediction"
                    f" -p nnUNetResEncUNetLPlans -c 3d_fullres"
                    f" -i '{inference_dir}'"
                    f" -o '{inference_out_dir}'"
                    f" -d {version} -f all -step_size {sliding_window}"
                    f" -chk checkpoint_best.pth -device cuda"
                )
                inference_exit = execute_and_log(inference_command, log_queue)
                if inference_exit != 0:
                    log_to_queue(
                        log_queue,
                        f"Inference failed for {scan_name} with exit code {inference_exit}.",
                        level=logging.ERROR,
                    )
                    continue

                # Step 3: Resample prediction back to patient space
                process_row_resampling(scan_row, log_queue)

                # Step 4: Move result to output
                predicted_file = Path(scan_row["cta_predicted"])
                if predicted_file.exists():
                    shutil.move(str(predicted_file), str(output_file))
                    log_to_queue(log_queue, f"Saved {output_file.name}")
                else:
                    log_to_queue(
                        log_queue,
                        f"Expected prediction file not found: {predicted_file}",
                        level=logging.ERROR,
                    )
            else:
                # ROI mode: move the preprocessed isotropic CTA to output/ROI/
                roi_dir = output_path / "ROI"
                roi_dir.mkdir(parents=True, exist_ok=True)
                iso_file = Path(scan_row["cta_roi_iso"])
                if iso_file.exists():
                    shutil.move(str(iso_file), str(output_file))
                    log_to_queue(log_queue, f"Saved ROI file {output_file.name}")
                else:
                    log_to_queue(
                        log_queue,
                        f"Expected isotropic file not found: {iso_file}",
                        level=logging.ERROR,
                    )

        except Exception as e:
            log_to_queue(
                log_queue,
                f"Error processing {scan_name}: {e}",
                level=logging.ERROR,
            )
        finally:
            # Clean up temp directory
            if temp_path.exists():
                shutil.rmtree(temp_path)


def _gpu_worker(
    df, output_path, version, log_queue,
    registration_settings_path_local, mode,
    sliding_window, gpu_id,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run_pipeline_sequential(
        df, output_path, version, log_queue,
        registration_settings_path_local, mode,
        sliding_window, gpu_id=gpu_id,
    )


def run_pipeline_multigpu(
    df, output_path, num_gpus, version, log_queue,
    registration_settings_path_local, mode, sliding_window,
):
    log_to_queue(log_queue, f"Multi-GPU mode: splitting {len(df)} scans across {num_gpus} GPUs.")
    df_splits = np.array_split(df, num_gpus)

    processes = []
    for gpu_id, df_split in enumerate(df_splits):
        if len(df_split) == 0:
            continue
        p = mp.Process(
            target=_gpu_worker,
            args=(
                df_split, output_path, version, log_queue,
                registration_settings_path_local, mode,
                sliding_window, gpu_id,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    log_to_queue(log_queue, "All GPU workers finished.")


def main():
    args = parse_arguments()

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file = str(
        Path(args.output_path) / f"vessel_segmentation_{args.pipeline_mode}_{current_time}.log"
    )
    print(f"Setting up logging process. Refer to {log_file}")

    manager = mp.Manager()
    log_queue = manager.Queue()

    listener = mp.Process(target=listener_process, args=(log_queue, log_file))
    listener.start()

    try:
        log_to_queue(log_queue, "Setting environment variables.")
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "10"
        os.environ["nnUNet_results"] = str(
            REPO_ROOT / "atlases_and_weights" / "weights"
        )

        registration_settings_path = (
            REPO_ROOT / "atlases_and_weights" / "atlases"
            / "rectangle_neck_scene_RegistrationMask" / "antspy_registration_settings.json"
        )
        registration_settings_path_local = (
            Path(args.output_path) / "antspy_registration_settings.json"
        )

        if not registration_settings_path_local.is_file():
            log_to_queue(
                log_queue,
                "Did not find ants registration json. Using default settings.",
            )
            shutil.copy2(registration_settings_path, registration_settings_path_local)
            if args.registration_method is not None:
                log_to_queue(
                    log_queue,
                    "Updating registration settings file with the supplied registration method.",
                )
                update_json_file(
                    registration_settings_path_local,
                    "type_of_transform",
                    args.registration_method,
                    log_queue,
                )
        else:
            log_to_queue(
                log_queue,
                f"Found ants registration json. Ensure the arguments are correct. "
                f"Delete {registration_settings_path_local} if necessary and run again.",
            )

        output_path = Path(args.output_path)
        pipeline_failures_csv = str(
            output_path / f"pipeline_failures_{args.pipeline_mode}_{current_time}.csv"
        )
        qc_csv = str(
            output_path / f"qc_{args.pipeline_mode}_{current_time}.csv"
        )

        log_to_queue(log_queue, "Starting the template registration pipeline.")

        if args.csv:
            log_to_queue(log_queue, "Processing data from: " + args.csv)
            df = process_csv(args.csv)
        elif args.directory:
            log_to_queue(log_queue, "Processing data from: " + args.directory)
            df = gather_nifti_data(args.directory)
            log_to_queue(log_queue, f"Found {len(df)} files.")

        if args.batch:
            log_to_queue(log_queue, "Running in batch mode.")
            log_to_queue(log_queue, "Creating output directories in: " + args.output_path)
            analysis_dir, inference_dir, inference_out_dir = setup_output_directories(
                args.output_path
            )

            log_to_queue(log_queue, "Updating input dataframe.")
            df = update_input_dataframe_fields(
                df,
                args.output_path,
                analysis_dir,
                inference_dir,
                inference_out_dir,
                registration_settings_path_local,
                args.pipeline_mode,
                log_queue,
            )

            num_threads = min(args.threads, len(df))
            run_pipeline(
                df,
                args.output_path,
                num_threads,
                args.num_gpus,
                args.version,
                log_queue,
                pipeline_failures_csv,
                qc_csv,
                args.pipeline_mode,
                args.sliding_window,
            )
        elif args.num_gpus > 1:
            run_pipeline_multigpu(
                df,
                args.output_path,
                args.num_gpus,
                args.version,
                log_queue,
                registration_settings_path_local,
                args.pipeline_mode,
                args.sliding_window,
            )
        else:
            log_to_queue(log_queue, "Running in sequential mode.")
            run_pipeline_sequential(
                df,
                args.output_path,
                args.version,
                log_queue,
                registration_settings_path_local,
                args.pipeline_mode,
                args.sliding_window,
            )

    except Exception as e:
        log_to_queue(log_queue, f"An error occurred: {e}", level=logging.ERROR)
    finally:
        log_queue.put(None)
        listener.join()


if __name__ == "__main__":
    main()
