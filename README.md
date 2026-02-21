# Robust vessel segmentation using Dynamic 4D-CTA

This repository contains code for our paper on training robust vessel segmentation models using Dynamic 4D-CTA data. For now, we provide code for inference using models trained on TopCoW, VesselVerse, and our DynaVessel dataset, as well as evaluation scripts and visualization tools. The model trained on DynaVessel is subject to the specified license in the CC-BY-NC-SA 4.0 file; models trained on other datasets are subject to the licenses of those datasets.

We aim to release the training and evaluation partitions of DynaVessel in the short term, but it depends obtaining institutional approval from the hospital network from which we sourced the data.

## Setting up weights and other required files

Please download the pretrained models, registration templates, and the Linux executable of ANTS from Google Drive: [link](https://drive.google.com/open?id=1uFTrSajk2oAx4LHctZB_0cg98Ubo1QJ-&usp=drive_fs)

The folder structure should be:

```bash
robust-vessel-segmentation/
├── atlases_and_weights/
│   ├── ants-2.6.3
│   ├── weights
|   └── atlases
```

Our model should be used with either v = 241 or v = 242, which can be set from the `run_segmentation.sh` script.

The 241 model was trained using Z-score normalization. The 242 model was trained with nnUNet's CT normalization. In our experience, the 241 model is more robust to variations in intensity distribution, but the 242 model is more robust to artifacts and distracting elements outside of the brain. We recommend using 241 when it can be guaranteed that the model's outputs can be post-processed to remove elements outside the brain via masking or if the presence of extraneous segmented objects is not a concern.

## Running inference

To run inference, activate a conda environment with nnUNet v2 and the other dependencies, and run the following command:

```bash
bash run_segmentation.sh <input_folder> <output_folder> <num_gpus>
```

The pipeline will automatically use the specified number of GPUs to process the scans in parallel. The output will be saved in the specified output folder.

(More detailed instructions will be posted after the MICCAI deadline on Feb 26, apologies!)