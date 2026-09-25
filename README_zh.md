# PhaseAlign：从频率先验到相位对齐的长序列时序预测

**PhaseAlign** 的官方实现。该模型把频率先验与显式相位结构注入注意力通路，并在相位域中解码预测窗口。

本仓库的代码组织沿用了 [Time-Series Library](https://github.com/thuml/Time-Series-Library)（TSLib）的约定，
因此数据集、训练入口、`scripts/` 脚本和 `checkpoints/` 目录结构对 TSLib 用户是熟悉的。

English documentation: [README.md](./README.md)

---

## 🚀 使用方式

### 准备数据

从 [Autoformer / TSLib 发布页](https://github.com/thuml/Time-Series-Library) 下载预处理好的数据集，放到
`./dataset` 下，使目录结构与脚本中的路径一致：

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

`ECL`、`Traffic`、`Weather` 已在 `data_provider/data_factory.py` 中注册，把对应目录与 CSV 放进 `./dataset`
即可使用。

### 环境配置

确认依赖安装完成后，即可直接运行 `scripts/` 中的脚本。

```bash
git clone https://github.com/<your-account>/PhaseAlign-LLM.git
cd PhaseAlign-LLM

conda create -n phasealign python=3.10
conda activate phasealign
```

⚠️ **CUDA 版本兼容提示**
`torch` 预编译包与 CUDA 版本强相关，请安装与本机 CUDA 匹配的版本。

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 运行示例

`/scripts` 目录提供了示例脚本。在 ETTh1 上运行 PhaseAlign：

```bash
bash ./scripts/ETTh1.sh
```

或直接启动单个配置：

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

每个数据集的脚本会一次性跑完四个标准预测长度 `96 / 192 / 336 / 720`：

| 脚本 | 数据集 | 预测长度 | 说明 |
|---|---|---|---|
| `scripts/ETTh1.sh` | ETTh1 | 96 / 192 / 336 / 720 | 小时级，`--freq h`，`--seasonal_period 24` |
| `scripts/ETTh2.sh` | ETTh2 | 96 / 192 / 336 / 720 | 小时级 |
| `scripts/ETTm1.sh` | ETTm1 | 96 / 192 / 336 / 720 | 15 分钟级，`--freq t`，`--seasonal_period 96` |
| `scripts/ETTm2.sh` | ETTm2 | 96 / 192 / 336 / 720 | 15 分钟级 |
| `scripts/exchange_rate.sh` | exchange | 96 / 192 / 336 / 720 | 日级，`--freq d`，`enc_in/dec_in/c_out 8` |
| `scripts/illness.sh` | illness | 24 / 36 / 48 / 60 | 周级短序列，`--seq_len 36 --label_len 18` |

⚠️ 部分脚本默认设置了 `CUDA_VISIBLE_DEVICES`，请根据实际 GPU 配置修改或删除，否则可能导致 GPU 未被使用。

### 训练与评估

我们在 `./scripts/` 下提供了各基准的实验脚本，可据此复现或扩展结果：

```bash
# 长序列预测，一次跑完四个预测长度
bash ./scripts/ETTh1.sh

# 纯评估，复用 ./checkpoints 中的最优 checkpoint
python run_main.py --task_name long_term_forecast --is_training 0 ... --model_id PhaseAlignDemo
```

多卡训练使用仓库自带的 DDP 配置。注意 `--batch_size` 是**每张卡**的 batch size：

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

注意：`accelerate_config_ddp.yaml` 中已启用 `mixed_precision: bf16`，因此**不要**再额外传 `--use_amp`。

### 快速自检

只跑 1 个 epoch，用于验证数据管线和训练循环是否正常：

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

## 📁 仓库结构

```
PhaseAlign-LLM/
├── README.md                      # 英文文档
├── README_zh.md                   # 中文文档
├── requirements.txt               # pip 依赖
├── LICENSE                        # MIT 协议
├── accelerate_config_ddp.yaml     # 双卡 DDP + bf16 的 Accelerate 配置
├── run_main.py                    # 统一的训练/评估入口
├── checkpoints/                   # 各实验的最优 checkpoint
├── data_provider/                 # 数据加载与划分
│   ├── data_factory.py            # 按 --data 选择 Dataset 并构建 DataLoader
│   └── data_loader.py             # Dataset_ETT_hour / Dataset_ETT_minute / Dataset_Custom
├── models/                        # 模型实现
│   ├── PhaseAlign.py              # PhaseAlign 主体及其全部子模块
│   └── __init__.py                # 导出 PhaseAlignModel
├── layers/                        # 可复用基础组件
│   ├── Embed.py                   # PatchEmbedding、TokenEmbedding 与时间戳嵌入
│   └── StandardNorm.py            # 可逆实例归一化 Normalize
├── utils/                         # 工具
│   ├── metrics.py                 # MAE / MSE / RMSE / MAPE / MSPE / RSE / CORR
│   ├── tools.py                   # EarlyStopping、adjust_learning_rate、vali
│   └── timefeatures.py            # 时间特征编码
└── scripts/                       # 可复现脚本，每个数据集一个文件
    ├── ETTh1.sh
    ├── ETTh2.sh
    ├── ETTm1.sh
    ├── ETTm2.sh
    ├── exchange_rate.sh
    └── illness.sh
```

---

## 🧠 理解项目架构

**端到端流程**：通过 `scripts/*.sh` 配置实验 → 运行 `python run_main.py ...` → `run_main.py` 解析参数并调用
`data_provider.data_factory.data_provider` 取得 train / val / test → 从 `models/` 实例化 `PhaseAlignModel` →
用 `utils/` 中的工具驱动训练循环 → 最优 state dict 写入
`./checkpoints/<setting>-<model_comment>/checkpoint`。

**数据层（`data_provider/`）**：`data_factory` 依据 `data_dict` 选择 `Dataset` 类并构建 `DataLoader`；
`data_loader` 实现滑窗逻辑。加载器是**通道独立**的：多变量序列被展开为单通道样本
（`feat_id = index // tot_len`），所有通道共享同一套模型参数。

**模型层（`models/` + `layers/`）**：`models/PhaseAlign.py` 中的 `Model.forecast()` 就是完整流水线
（归一化 → 通道展开 → FFT 统计 → patch embedding → 频率注入 → 时频交叉注意力 → 可选时间戳注入 →
差分分支 → `PhaseAdapterBlock` 堆叠 → `FuturePhaseDecoder` → 双预测头 → 反归一化）。可复用的嵌入与归一化
原语放在 `layers/` 以避免重复。

**训练层（`run_main.py`）**：单个基于 Accelerate 的脚本。分布式后端完全由 Accelerate 配置文件决定，因此
同一份代码无需修改即可跑单卡、DDP 或 DeepSpeed。每次迭代有两次前向（预测 + `future_alignment_repr`），
所以 DDP 的 `DistributedDataParallelKwargs` 使用了 `find_unused_parameters=True, static_graph=True`。

**工具层（`utils/`）**：`metrics` 集中评估指标，`tools` 提供 `EarlyStopping`、`adjust_learning_rate`、
`vali`，`timefeatures` 提供日历编码。

**阅读路径**：建议按 `scripts → run_main.py → data_provider/data_factory.py → data_provider/data_loader.py
→ models/PhaseAlign.py → layers/` 的顺序阅读。

---

## 📖 引用

如果本仓库对你有帮助，请引用我们的论文。

```bibtex
@article{phasealign,
  title   = {PhaseAlign: From Frequency Priors to Phase Alignment for Long-Term Time Series Forecasting},
  author  = {<authors>},
  journal = {<venue>},
  year    = {<year>}
}
```

---

## 📧 联系方式

如有任何问题或建议，欢迎联系：

- `<name>`（`<email>`）

也可以在 Issues 中提出。

---

## 🙏 致谢

本仓库基于以下项目构建：

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) —— 代码组织约定、训练入口布局与数据集预处理。
- [Autoformer](https://github.com/thuml/Autoformer) —— 长序列预测评测协议与数据集。
- [PatchTST](https://github.com/yuqinie98/PatchTST) —— patch embedding 与通道独立设计。
- [Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers) —— 可逆实例归一化。

所有实验数据集均为公开数据，来源见上述链接。
