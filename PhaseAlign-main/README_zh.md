# PhaseAlign：从频率先验到相位对齐的长序列时序预测

**PhaseAlign** 的官方实现：把频率先验与显式相位结构注入注意力通路，并在相位域中解码预测窗口。

代码组织沿用了 [Time-Series Library](https://github.com/thuml/Time-Series-Library)（TSLib）的约定，
数据集、训练入口与 `checkpoints/` 目录结构对 TSLib 用户是熟悉的。

English documentation: [README.md](./README.md)

---

## 安装

```bash
git clone https://github.com/<your-account>/PhaseAlign.git
cd PhaseAlign

conda create -n phasealign python=3.10
conda activate phasealign

# torch 版本需与本机 CUDA 匹配
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 数据

使用 [Autoformer / TSLib 发布页](https://github.com/thuml/Time-Series-Library) 中预处理好的数据集，
放到 `./dataset` 下，并保持与脚本中引用的目录结构一致。

## 使用

论文中报告的所有配置都已硬编码在脚本里，脚本会一次性跑完标准预测长度与随机种子：

```bash
bash ./scripts/ETTh1.sh      # 或 ETTh2 / ETTm1 / ETTm2 / exchange_rate / Weather / Electricity
```

如需自定义实验，`run_main.py` 接受 TSLib 风格的命令行参数；具体取值可参考任意一个脚本。

每个种子写入 `checkpoints/<setting>_sd<种子>-<model_comment>/`，内含 checkpoint、`run_meta.json`
（本次运行的生效配置）与 `metrics.json`；最后一个种子结束后还会写出并打印 mean ± std 的种子汇总。
全部脚本跑完后，结果表格直接由 checkpoint 生成：

```bash
python scripts/collect_results.py --checkpoints ./checkpoints            # 输出 Markdown 表
python scripts/collect_results.py --checkpoints ./checkpoints --out results.csv
```

在同一条命令上加 `--is_training 0` 即可只评估已训练好的 checkpoint，不再训练。

多卡：`accelerate launch --config_file accelerate_config_ddp.yaml run_main.py <参数>`；精度只由该配置文件决定。

## 目录结构

```
run_main.py            训练 / 评估入口
data_provider/         数据加载与 train/val/test 划分
models/PhaseAlign.py   模型实现
layers/                嵌入与归一化等基础组件
utils/                 指标、EarlyStopping、时间特征
scripts/               每个数据集一个 bash 脚本，以及 collect_results.py
```

## 引用

```bibtex
@article{phasealign,
  title   = {PhaseAlign: From Frequency Priors to Phase Alignment for Long-Term Time Series Forecasting},
  author  = {<authors>},
  journal = {<venue>},
  year    = {<year>}
}
```

## 致谢

本仓库基于 [Time-Series-Library](https://github.com/thuml/Time-Series-Library)、
[Autoformer](https://github.com/thuml/Autoformer)、[PatchTST](https://github.com/yuqinie98/PatchTST) 与
[Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers) 构建。
所有实验数据集均为公开数据，来源见上述链接。

如有问题或建议，欢迎提 Issue。
