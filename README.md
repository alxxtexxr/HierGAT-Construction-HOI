# HierGAT

## Environment Setup
Install dependencies by running:
```bash
pip install -r requirements_2025.txt
```

## Download Data (Cite from 2G-GCN: https://github.com/tanqiu98/2G-GCN)
Please download the necessary data [here](https://drive.google.com/drive/folders/1yfwItIoQrAnbnk5GTjbbfN8Ls8Ybl_hr?usp=sharing) and put the 
downloaded data folder in this current directory (i.e. `./data/...`).

## Train the Model
To train the model from scratch:
1. Run `sh tt_[dataset]_2025.sh` to execute Stage 1.
2. After Stage 1 completes, edit the `models` field in `conf/config_[dataset]_[number]_2025.yaml`, changing
`2G-GCN_stage1_xxxxx` to `2G-GCN_stage2_xxxxx` for Stage 2.
3. Run `sh tt_[dataset]_2025.sh` again to execute Stage 2.

## Test the Model
Example on MPHOI-72: Once you have obtained the pre-trained models for all subject groups, you can get the cross-validation results by running: 
```bash
mkdir predict_results # If you haven't created the directory yet

python predict_2025.py \
--pretrained_model_dir "./outputs_hiergat/mphoi/2G-GCN/hs512_e40_bs8_lr0.0001_0.3_Subject14" \
--save_visualisations_dir "./predict_results" \
--cross_validate
```

