# this script assumes the conda environment with nnUNet v2 and other dependencies is active

export version=241
export PATH=./atlases_and_weights/ants-2.6.3/bin:$PATH
python ./src/inference/extractVessels.py -d ${1} ${2}  -m 'Full' -s 0.5 -g ${3} -v ${version}