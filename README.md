# Robust brain vessel segmentation in 3D CTA using Dynamic 4D-CTA data \& Scaling up fine-grained intracranial vessel annotations in computed tomography angiography

!! NEWS: Our work has been accepted to MICCAI 2026 and MLHC 2026, see you in Strasbourg and Baltimore! 

This repository covers our two papers on training robust vessel segmentation models using Dynamic 4D-CTA data. We provide code for inference using models trained on TopCoW, VesselVerse, and DynaVessel. The models trained on DynaVessel are subject to the specified license in the CC-BY-NC-SA 4.0 file; models trained on other datasets are subject to the licenses of those datasets. Code for pre-processing data and training/running models on our SemanticVessel dataset will be released shortly.


## Dataset access

To access DynaVessel (around 100 scans with artery, vein labels) and SemanticVessel (around 360 scans with vein and 20 artery labels), please email a request to us through ceballosarroyo.a@northeastern.edu while CC'ing Dr. Geoffrey Young at gsyoung@bwh.harvard.edu. This is necessary as our IRB requires us to keep track of researchers who access the dataset.

In your request, please include your name, affiliation, and a brief description of your research project.

## Citing us

If you find our DynaVessel dataset and pre-trained models, please cite our preprint:

Alberto Mario Ceballos-Arroyo, Shrikanth M. Yadav, Chu-Hsuan Lin, Jisoo Kim, Geoffrey S. Young, Lei Qin, Huaizu Jiang. 2026.
Robust automatic brain vessel segmentation in 3D CTA scans using dynamic 4D-CTA data. <https://arxiv.org/abs/2602.00391>

If you use our SemanticVessel dataset, please also cite:

Chu-Hsuan Lin, Alberto Mario Ceballos-Arroyo, Jisoo Kim, Shrikanth M. Yadav, Huaizu Jiang, Lei Qin, Geoffrey S. Young. <https://arxiv.org/abs/2606.21756>

## Setting up weights and other required files

To run inference on any 3D CTA scan of the head, please download the pretrained models, registration templates, and the Linux executable of ANTS from Google Drive: [URL](https://drive.google.com/open?id=1uFTrSajk2oAx4LHctZB_0cg98Ubo1QJ-&usp=drive_fs)

The folder structure should be:

```bash
robust-vessel-segmentation/
├── atlases_and_weights/
│   ├── ants-2.6.3
│   ├── weights
|   └── atlases
```

Our model should be used with either v = 241 or v = 242, which can be set from the `run_segmentation.sh` script.

The 241 model was trained using Z-score normalization. The 242 model was trained with nnUNet's CT normalization. In our experience, the 241 model is more robust to variations in intensity distribution, but the 242 model is more robust to artifacts and distracting elements outside of the brain. We recommend using 241 when it can be guaranteed that the model's outputs can be post-processed to remove elements outside the brain via masking or if the presence of extraneous segmented objects is not a concern. We provide a way to use a head-neck ROI to mask out non-brain elements by switching from the 'Prediction' flag to 'Full', however, this can fail if the scan does not contain the full head.

## Running inference

To run inference, activate a conda environment with nnUNet v2 and the other dependencies, and run the following command:

```bash
bash run_segmentation.sh <input_folder> <output_folder> <num_gpus>
```

The pipeline will automatically use the specified number of GPUs to process the scans in parallel. The output will be saved in the specified output folder.