# PhaseAlign: From Frequency Priors to Phase Alignment for Long-Term Time Series Forecasting

Official implementation of **PhaseAlign**, a time series forecasting model that injects frequency priors and
explicit phase structure into the attention pathway, and decodes the horizon in the phase domain.

The repository follows the code conventions of
the [Time-Series Library](https://github.com/thuml/Time-Series-Library) (TSLib), so that datasets, training
entry points, scripts and the `checkpoints/` layout stay familiar to researchers who already work with TSLib.

中文文档: [README_zh.md](./README_zh.md)

---

## 🚀 Usage

### Prepare Data

Download the well-preprocessed datasets from
the [Autoformer / TSLib release](https://github.com/thuml/Time-Series-Library) and place them under `./dataset`
so that the layout matches the paths used by the scripts:

```
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── exchange_rate/
│   └── exchange_rate.csv
└── illness/
    └── national_illness.csv
```

`ECL`, `Traffic` and `Weather` are already registered in `data_provider/data_factory.py`; add the corresponding
folder and CSV under `./dataset` to use them.

### Environment Setup

Ensure you have the dependencies installed, then you can run the provided scripts directly.

```bash
git clone https://github.com/<your-account>/PhaseAlign-LLM.git
cd PhaseAlign-LLM

conda create -n phasealign python=3.10
conda activate phasealign
```

⚠️ **CUDA Compatibility Notice**
The `torch` prebuilt package is CUDA-version specific. Install the build matching your local CUDA toolkit.

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Running Examples

Example script is provided in the `/scripts` directory. You can run PhaseAlign on ETTh1 by:

```bash
bash ./scripts/ETTh1.sh
```

Or launch a single configuration directly:

```bash
python run_main.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id PhaseAlignDemo --model_comment PhaseAlignDemo \
  --model PhaseAlign --data ETTh1 \
  --root_path ./dataset/ETT-small --data_path ETTh1.csv \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 384 --n_heads 8 --e_layers 4 --d_ff 768 --dropout 0.2 \
  --batch_size 64 --eval_batch_size 64 --train_epochs 15 --learning_rate 5e-4 \
  --phase_model PhaseAlign --phase_dim 16 \
  --num_workers 0 --lradj constant \
  --loss MIXED --mse_weight 0.05 --mae_weight 0.95 \
  --use_seasonal_residual 1 --seasonal_period 24 --seasonal_alpha 0.3 --early_stop 0 \
  --seasonal_weights 0.5,0.25,0.15,0.10
```

Each dataset script sweeps the four standard horizons `96 / 192 / 336 / 720` in one go:

| Script | Dataset | Horizons | Notes |
|---|---|---|---|
| `scripts/ETTh1.sh` | ETTh1 | 96 / 192 / 336 / 720 | hourly, `--freq h`, `--seasonal_period 24` |
| `scripts/ETTh2.sh` | ETTh2 | 96 / 192 / 336 / 720 | hourly |
| `scripts/ETTm1.sh` | ETTm1 | 96 / 192 / 336 / 720 | 15-min, `--freq t`, `--seasonal_period 96` |
| `scripts/ETTm2.sh` | ETTm2 | 96 / 192 / 336 / 720 | 15-min |
| `scripts/exchange_rate.sh` | exchange | 96 / 192 / 336 / 720 | daily, `--freq d`, `enc_in/dec_in/c_out 8` |
| `scripts/illness.sh` | illness | 24 / 36 / 48 / 60 | weekly short series, `--seq_len 36 --label_len 18` |

⚠️ Some scripts set `CUDA_VISIBLE_DEVICES` by default. Please modify or remove this setting according to your
actual GPU configuration, otherwise it may prevent GPU usage.

### Train and Evaluate

We provide the experiment scripts for the supported benchmarks under the folder `./scripts/`. You can reproduce
or extend the results as follows:

```bash
# long-term forecasting, four horizons in one run
bash ./scripts/ETTh1.sh

# evaluation only, reusing the best checkpoint in ./checkpoints
python run_main.py --task_name long_term_forecast --is_training 0 ... --model_id PhaseAlignDemo
```

To run on multiple GPUs, use the bundled DDP configuration. `--batch_size` is the **per-GPU** batch size:

```bash
accelerate launch --config_file accelerate_config_ddp.yaml run_main.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id PhaseAlignDemo --model_comment PhaseAlignDemo \
  --model PhaseAlign --data ETTh1 \
  --root_path ./dataset/ETT-small --data_path ETTh1.csv \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 384 --n_heads 8 --e_layers 4 --d_ff 768 --dropout 0.2 \
  --batch_size 64 --eval_batch_size 64 --train_epochs 15 --learning_rate 5e-4 \
  --num_workers 0 --lradj constant
```

Note: `mixed_precision: bf16` is already enabled in `accelerate_config_ddp.yaml`, so do **not** additionally
pass `--use_amp`.

### Quick Test

A one-epoch smoke test that only checks the data pipeline and the training loop:

```bash
python run_main.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id SmokeTest --model_comment SmokeTest \
  --model PhaseAlign --data ETTh1 \
  --root_path ./dataset/ETT-small --data_path ETTh1.csv \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --n_heads 4 --e_layers 1 --d_ff 128 \
  --batch_size 16 --eval_batch_size 16 --train_epochs 1 \
  --num_workers 0 --show_progress 1
```

---

## 📁 Project Structure

Inspect the project structure:

```
PhaseAlign-LLM/
├── README.md                      # English documentation
├── README_zh.md                   # Chinese documentation
├── requirements.txt               # pip dependency list
├── LICENSE                        # MIT license
├── accelerate_config_ddp.yaml     # 2-GPU DDP + bf16 Accelerate configuration
├── run_main.py                    # unified training / evaluation entry point
├── checkpoints/                   # per-experiment best checkpoints
├── data_provider/                 # dataset loaders and splits
│   ├── data_factory.py            # maps --data to the proper Dataset and builds the DataLoader
│   └── data_loader.py             # Dataset_ETT_hour / Dataset_ETT_minute / Dataset_Custom
├── models/                        # model implementations
│   ├── PhaseAlign.py              # PhaseAlign architecture and all of its sub-modules
│   └── __init__.py                # exports PhaseAlignModel
├── layers/                        # reusable primitives
│   ├── Embed.py                   # PatchEmbedding, TokenEmbedding and timestamp embeddings
│   └── StandardNorm.py            # reversible instance Normalize
├── utils/                         # utility toolbox
│   ├── metrics.py                 # MAE / MSE / RMSE / MAPE / MSPE / RSE / CORR
│   ├── tools.py                   # EarlyStopping, adjust_learning_rate, vali
│   └── timefeatures.py            # time-feature encoding
└── scripts/                       # reproducible bash recipes, one file per dataset
    ├── ETTh1.sh
    ├── ETTh2.sh
    ├── ETTm1.sh
    ├── ETTm2.sh
    ├── exchange_rate.sh
    └── illness.sh
```

---

## 🧠 Understand the Project Architecture

**E2E flow**: configure an experiment through `scripts/*.sh` → run `python run_main.py ...` → `run_main.py`
parses the arguments and calls `data_provider.data_factory.data_provider` for the train / val / test splits →
instantiates `PhaseAlignModel` from `models/` → drives the training loop with helpers from `utils/` → the best
state dict is written to `./checkpoints/<setting>-<model_comment>/checkpoint`.

**Data layer (`data_provider/`)**: `data_factory` selects the `Dataset` class from `data_dict` and builds the
`DataLoader`; `data_loader` implements the sliding-window logic. The loader is **channel-independent**: a
multivariate series is unrolled into single-channel samples indexed by `feat_id = index // tot_len`, and all
channels share one set of model parameters.

**Model layer (`models/` + `layers/`)**: `models/PhaseAlign.py` holds `Model.forecast()`, which is the full
pipeline (normalize → channel expansion → FFT statistics → patch embedding → frequency injection → time–frequency
cross-attention → optional timestamp injection → difference branch → `PhaseAdapterBlock` stack →
`FuturePhaseDecoder` → dual prediction head → de-normalize). Reusable embedding and normalization primitives live
in `layers/` to avoid duplication.

**Training layer (`run_main.py`)**: a single Accelerate-based script. The distributed backend is selected purely
by the Accelerate configuration file, so the same code runs on 1 GPU, on DDP, or on DeepSpeed without edits. The
loop performs two forward passes per iteration (forecast + `future_alignment_repr`), which is why the DDP
`DistributedDataParallelKwargs` uses `find_unused_parameters=True, static_graph=True`.

**Utility layer (`utils/`)**: `metrics` centralizes evaluation, `tools` bundles `EarlyStopping`,
`adjust_learning_rate` and `vali`, and `timefeatures` provides the calendar encodings.

**Learning path**: the recommended reading order is `scripts → run_main.py → data_provider/data_factory.py →
data_provider/data_loader.py → models/PhaseAlign.py → layers/`.

---

## 📖 Citation

If you find this repo useful, please cite our paper.

```bibtex
@article{phasealign,
  title   = {PhaseAlign: From Frequency Priors to Phase Alignment for Long-Term Time Series Forecasting},
  author  = {<authors>},
  journal = {<venue>},
  year    = {<year>}
}
```

---

## 📧 Contact

If you have any questions or suggestions, feel free to contact us:

- `<name>` (`<email>`)

Or describe it in Issues.

---

## 🙏 Acknowledgement

This repository is constructed based on the following projects:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) — the codebase conventions, the training
  entry point layout and the dataset preprocessing.
- [Autoformer](https://github.com/thuml/Autoformer) — the long-term forecasting evaluation protocol and datasets.
- [PatchTST](https://github.com/yuqinie98/PatchTST) — the patch embedding and channel-independent design.
- [Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers) — the reversible
  instance normalization.

All the experiment datasets are public, and we obtain them from the links above.
