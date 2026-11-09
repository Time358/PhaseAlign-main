# PhaseAlign: From Frequency Priors to Phase Alignment for Long-Term Time Series Forecasting

Official implementation of **PhaseAlign**, a long-term time series forecasting model that injects frequency
priors and explicit phase structure into the attention pathway and decodes the horizon in the phase domain.

The code follows the conventions of the [Time-Series Library](https://github.com/thuml/Time-Series-Library)
(TSLib), so datasets, training entry points and the `checkpoints/` layout are familiar to TSLib users.

中文文档: [README_zh.md](./README_zh.md)

---

## Installation

```bash
git clone https://github.com/<your-account>/PhaseAlign.git
cd PhaseAlign

conda create -n phasealign python=3.10
conda activate phasealign

# the torch build must match your local CUDA toolkit
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

Use the preprocessed datasets from
the [Autoformer / TSLib release](https://github.com/thuml/Time-Series-Library) and place them under `./dataset`,
keeping the layout referenced by the scripts.

## Usage

Every configuration reported in the paper is hard-coded in the scripts, which sweep the standard horizons and
seeds:

```bash
bash ./scripts/ETTh1.sh      # or ETTh2 / ETTm1 / ETTm2 / exchange_rate / Weather / Electricity
```

`run_main.py` accepts the usual TSLib-style arguments if you want a custom run; see any script for the exact
command line it issues.

Each seed writes `checkpoints/<setting>_sd<seed>-<model_comment>/` with the checkpoint, `run_meta.json` (the
effective settings) and `metrics.json`; after the last seed a seed summary with the mean ± std is written and
printed. Once the scripts finish, the result tables are generated from the checkpoints:

```bash
python scripts/collect_results.py --checkpoints ./checkpoints            # Markdown to stdout
python scripts/collect_results.py --checkpoints ./checkpoints --out results.csv
```

Evaluate a trained checkpoint without retraining by passing `--is_training 0` to the same command.

Multi-GPU: `accelerate launch --config_file accelerate_config_ddp.yaml run_main.py <arguments>`. Precision is
set by that config file only.

## Project Structure

```
run_main.py            training / evaluation entry point
data_provider/         dataset loading and train/val/test splits
models/PhaseAlign.py   model implementation
layers/                embedding and normalisation primitives
utils/                 metrics, EarlyStopping, time features
scripts/               one bash recipe per dataset, plus collect_results.py
```

## Citation

```bibtex
@article{phasealign,
  title   = {PhaseAlign: From Frequency Priors to Phase Alignment for Long-Term Time Series Forecasting},
  author  = {<authors>},
  journal = {<venue>},
  year    = {<year>}
}
```

## Acknowledgement

This repository is constructed based on [Time-Series-Library](https://github.com/thuml/Time-Series-Library),
[Autoformer](https://github.com/thuml/Autoformer), [PatchTST](https://github.com/yuqinie98/PatchTST) and
[Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers). All experiment datasets are
public and obtained from the links above.

For questions or suggestions, please open an Issue.
