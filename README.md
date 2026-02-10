## HW02Q8 — Multiclass Classification (Android Malware) + Boxplot

### What this does
This part trains a simple neural network classifier on the Android malware dataset and:
- saves training/testing metrics (accuracy, macro precision/recall/F1) into CSV files
- aggregates multiple runs and generates a boxplot PDF comparing metrics across runs

### 1) Download the dataset
From the project root (~/DLHW2), run:

chmod +x malwaredatadownload.sh
./malwaredatadownload.sh

This downloads the dataset to:
- data/Android_Malware.csv
------------------------------------------------------------

### 2) Run one training run (creates one metrics CSV)
From the project root (~/DLHW2):

uv run python scripts/multiclass_impl.py --keyword hw02 --standardize --epoch 50 --eta 0.001

This creates a file like:
- results/metrics_hw02_<TIMESTAMP>.csv

------------------------------------------------------------

### 3) Run 5 training runs + generate the boxplot (recommended)
Use the bash script:

chmod +x multiclass_impl.sh
./multiclass_impl.sh

This will:
1) run scripts/multiclass_impl.py five times (same parameters, same keyword)
2) run scripts/multiclass_eval.py to aggregate all CSV files matching the keyword
3) save the boxplot PDF to results/

------------------------------------------------------------

### 4) Generate the boxplot only (if you already have CSV files)
uv run python scripts/multiclass_eval.py --keyword hw02

------------------------------------------------------------

### Where outputs are saved
- Metrics CSV files:
  - results/metrics_hw02_<TIMESTAMP>.csv
- Boxplot PDF:
  - results/boxplot_hw02_<TIMESTAMP>.pdf

------------------------------------------------------------

### Notes
- The dataset includes four classes:
  - Android_Adware, Android_Scareware, Android_SMS_Malware, Benign
- Some non-useful columns are removed during preprocessing (IDs, IPs, ports, protocol, timestamp).
- --standardize standardizes features using training-set statistics only.
