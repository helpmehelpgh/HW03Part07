# HW03Part07 - Cruise Control State Classifier

This repository contains the solution for **Section 7: Cruise Control State Classifier** from HW03.

The goal of this part is to build a **binary classifier** that predicts whether **Adaptive Cruise Control (ACC)** is enabled or not from vehicle time-history signals.

## Overview

This part was solved in **three different versions** to improve performance by changing:

- model setup
- epochs
- feature design
- training parameters

The implemented work includes:

- preprocessing of decoded CAN signal files
- matching front-left wheel speed with ACC status
- binary target construction
- zero-order hold label alignment
- lagged time-history feature construction
- train/test split and scaling
- training and evaluation of multiple classifier versions
- loss and accuracy plots
- ONNX model export
- inference script

The classifier predicts:

- **1** if ACC status = **6**
- **0** otherwise

## Dataset

Dataset path used:

```bash
/data/CPE_487-587/ACCDataset
```

For the first implementation, the following files were used from each experiment:

- `*_wheel_speed_fl.csv`
- matching `*_acc_status.csv`

For the improved versions, additional signals were included:

- `*_relative_vel.csv`
- `*_lead_distance.csv`
- `*_accely.csv`

A total of **13 matched experiments** were found and processed.

## Data Preparation

The preprocessing pipeline performs the following steps:

1. Read the `Time` and `Message` columns from each decoded signal file.
2. Convert front-left wheel speed from **km/h** to **m/s**.
3. Convert ACC status into a binary label:
   - `1` if status is `6`
   - `0` otherwise
4. Remove duplicate ACC timestamps.
5. Align labels and additional signals to the wheel-speed timeline using **zero-order hold**.
6. Construct historical lag features.

## Implemented Versions

### Version 1

The first version uses only front-left wheel speed history:

- `v_t`
- `v_t-1`
- ...
- `v_t-10`

This version produced a strong result, and in one of the completed runs the accuracy was about **87%**.

### Version 2

The second version uses multiple signals and more derived features:

- current values of speed, relative velocity, lead distance, and longitudinal acceleration
- lagged histories
- first-difference features
- rolling mean and rolling standard deviation

This version was created to improve the model by using richer information. In another completed run, the obtained accuracy was about **78%**.

### Version 3

The third version keeps the richer multi-signal setup but uses:

- smaller history length
- a smaller and more regularized model
- different epochs and learning-rate settings
- modified training configuration to reduce overfitting

In another completed run, the obtained accuracy was about **83%**.

## Files

Main files created for this part:

- `src/mchnpkg/deepl/acc_module.py`
- `src/mchnpkg/deepl/acc_module_v2.py`
- `src/mchnpkg/deepl/acc_module_v3.py`
- `scripts/acc_impl.py`
- `scripts/acc_impl.sh`
- `scripts/acc_inference.py`
- `scripts/acc_impl_v2.py`
- `scripts/acc_impl_v3.py`

## Example Run Commands

### Version 1

```bash
export PYTHONPATH=src
python scripts/acc_impl.py \
  --data_dir /data/CPE_487-587/ACCDataset \
  --output_dir results/acc_test \
  --k 10 \
  --sample_size 300000 \
  --test_size 0.2 \
  --epochs 5 \
  --batch_size 256 \
  --lr 0.001 \
  --num_workers 2
```

### Version 2

```bash
export PYTHONPATH=src
python scripts/acc_impl_v2.py \
  --data_dir /data/CPE_487-587/ACCDataset \
  --output_dir results/acc_v2_test \
  --k 10 \
  --sample_size 300000 \
  --test_ratio 0.2 \
  --epochs 10 \
  --batch_size 256 \
  --lr 0.001 \
  --num_workers 2
```

### Version 3

```bash
export PYTHONPATH=src
python scripts/acc_impl_v3.py \
  --data_dir /data/CPE_487-587/ACCDataset \
  --output_dir results/acc_v3_test \
  --k 5 \
  --sample_size 300000 \
  --test_ratio 0.2 \
  --epochs 10 \
  --batch_size 256 \
  --lr 0.0005 \
  --num_workers 2
```

## Inference

After training, inference can be run using the ONNX model from Version 1:

```bash
export PYTHONPATH=src
python scripts/acc_inference.py \
  --onnx_model results/acc_test/accnet.onnx \
  --features 0.1 0.2 0.3 0.25 0.24 0.22 0.21 0.20 0.18 0.17 0.15
```

The 11 input values correspond to:

- `v_t`
- `v_t-1`
- ...
- `v_t-10`

## Output Directories

The results from the three solved versions can be found in these directories:

- `results/acc_test`
- `results/acc_v2_test`
- `results/acc_v3_test`

These folders contain output files such as:

- accuracy plots
- loss plots
- ONNX models
- summary files

Examples include:

- `acc_accuracy.png`
- `acc_loss.png`
- `accnet.onnx`
- `summary.txt`
- `acc_v2_accuracy.png`
- `acc_v2_loss.png`
- `accnet_v2.onnx`
- `acc_v3_accuracy.png`
- `acc_v3_loss.png`
- `accnet_v3.onnx`

## Summary of Results

This part was solved in **three different versions** by changing epochs and other parameters in order to improve classification accuracy.

Observed results from the completed runs were approximately:

- **Version 1:** about **87%**
- **Version 2:** about **78%**
- **Version 3:** about **83%**

Therefore, the first version gave the strongest accuracy among the tested configurations, while the later versions were useful for exploring richer features, different train/test strategies, and regularization choices.

## Notes

The different versions were intentionally kept to compare how feature design, history length, model complexity, and training parameters affect the final ACC classification performance.

This makes the work useful not only as a final classifier, but also as an experimental comparison of three solution strategies.