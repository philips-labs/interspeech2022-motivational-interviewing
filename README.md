# Towards Automated Counselling Decision-Making: Remarks on Therapist Action Forecasting on the AnnoMI Dataset (INTERSPEECH 2022)

## Introduction
* Code for our [INTERSPEECH 2022 paper](https://www.isca-speech.org/archive/pdfs/interspeech_2022/wu22c_interspeech.pdf) titled "Towards Automated Counselling Decision-Making: Remarks on Therapist Action Forecasting on the AnnoMI Dataset"

## Environment Setup
* Note that $REPO is the folder of the repository (i.e. the folder where you see this README), after `git clone`.

```sh
cd $REPO

# install Conda environment
conda env create -f ./environment.yml
conda activate interspeechmi

# install the module
pushd $REPO/interspeechmi
python3 -m build
pip install -e .
popd
```

## Steps for reproducing our paper's results
1. `bash $REPO/interspeechmi/sh_scripts/run_and_collect_results.sh` (may take days to complete depending on your hardware)
2. Use Jupyter Notebook to run `$REPO/interspeechmi/py_scripts/plot_code_forecast_scores.ipynb`, and you'll be able to see the figures that summarise the performances under different settings.

## Dataset used
* [AnnoMI](https://github.com/uccollab/AnnoMI/archive/refs/heads/main.zip) (Wu et al. 2021)

## Citation
```bash
@inproceedings{wu22c_interspeech,
  author={Zixiu Wu and Rim Helaoui and Diego {Reforgiato Recupero} and Daniele Riboni},
  title={{Towards Automated Counselling Decision-Making: Remarks on Therapist Action Forecasting on the AnnoMI Dataset}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={1906--1910},
  doi={10.21437/Interspeech.2022-506}
}
```
