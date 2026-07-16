# DOVE 纯算法 1.5x 复现说明

本仓库基于原始 DOVE 项目扩展；上游模型、数据集和基础训练说明保存在 [README.upstream.md](README.upstream.md)。

## 1. 复现范围

本说明复现最终采用的单卡纯算法方案：保留 DOVE 的 one-step latent Transformer，用蒸馏后的轻量 VAE encoder 和 decoder 替换原始 VAE。

- Encoder：四个 down block 均保留 1 层，mid block 保留 1 层，即 `[1,1,1,1] + mid=1`。
- Decoder：四个 up block 使用非均匀深度 `[1,1,1,2]`，高分辨率 block 多保留一层。
- Decoder teacher：完整 DOVE 轨迹产生的 decoder 输入 latent 和原始 decoder RGB 输出。
- Encoder teacher：真实退化 LQ 条件及原始 encoder posterior moments。
- 不使用 `torch.compile`、TensorRT、CUDA Graph、多卡推理或减少输出帧数。

该路线不是去 VAE。此前的 pixel/L2P、structured diffusion 和多卡 tile 路线已被实验否决，不属于最终复现范围。

## 2. 已验证结果

验证集为 UDM10，输入 32 帧 `180x318`，输出 32 帧 `720x1272`。速度在同一张 NVIDIA L40S 上统计，剔除模型加载和设备迁移，但包含预处理、推理、视频拼接与保存。

| 模型 | 稳态端到端 | 加速 | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 DOVE | 435.91 s | 1.000x | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5013 |
| 轻量 encoder + decoder | 289.55 s | 1.505x | 26.0109 | 0.7675 | 0.2818 | 0.1553 | 0.4988 |

结果应表述为“达到 1.5x，伴随小幅画质折损”，不能表述为无损优化。详细实验记录见 [pure_algorithm_results.md](docs/experiments/pure_algorithm_results.md)。

## 3. 代码结构

```text
dove_vae_distill/
  data.py                    # HQ-VSR 退化数据与 teacher pipeline 公共逻辑
  encoder.py                 # 轻量 encoder 构建与 checkpoint profile
  decoder.py                 # 轻量 decoder 构建与 checkpoint profile
  runtime.py                 # 推理时安装并校验 student VAE
  profiling.py               # 同步 CUDA 的阶段耗时统计
  cli/
    build_decoder_cache.py   # 完整 DOVE trajectory cache
    train_decoder.py         # decoder 蒸馏
    build_encoder_cache.py   # 退化 LQ encoder cache
    train_encoder.py         # encoder posterior 蒸馏
scripts/slurm/
  submit_pure_algorithm_pipeline.sh
  01_build_decoder_cache.sbatch
  02_train_decoder.sbatch
  03_build_encoder_cache.sbatch
  04_train_encoder.sbatch
  05_evaluate.sbatch
```

## 4. 环境与硬件

- Linux + CUDA，Python 3.11。
- Decoder cache 和训练建议 1x H100 80 GB；encoder cache/训练可使用 1x L40/L40S 48 GB。
- 最终速度必须在单张 L40S 上测量；训练卡不参与速度结论。
- 建议准备至少 100 GB 可写空间，用于两类 cache、checkpoint、运行时编译缓存和 UDM10 输出。
- Slurm 集群；非 Slurm 环境也可逐个执行 `python -m dove_vae_distill.cli...` 命令。

安装：

```bash
conda create -n dove-pure python=3.11 -y
conda activate dove-pure
pip install -r requirements-vae-distill.txt
```

CUDA 对应的 PyTorch wheel 应按集群驱动单独选择。每个 Slurm job 会把 `python --version` 和 `pip freeze` 保存到 `RUN_ROOT/environment/`。

安装完成后执行：

```bash
make check
```

## 5. 数据和模型

按上游 README 下载：

1. DOVE Stage-2 模型，目录中应包含 `vae/`、`transformer/`、`scheduler/`、`tokenizer/` 和 `text_encoder/`。
2. HQ-VSR 训练集及对应列表文件 `HQ-VSR.txt`，至少 1024 个可读取样本。
3. UDM10 的 `LQ-Video/` 和 `GT-Video/`，各 10 个视频。

先检查输入：

```bash
python scripts/check_pure_algorithm_setup.py \
  --model_path /path/to/pretrained_models/DOVE \
  --hqvsr_root /path/to/HQ-VSR \
  --hqvsr_list /path/to/HQ-VSR.txt \
  --udm10_lq /path/to/UDM10/LQ-Video \
  --udm10_gt /path/to/UDM10/GT-Video
```

## 6. 一键提交完整流程

复制环境变量模板并修改为当前机器路径：

```bash
cp scripts/pure_algorithm.env.example pure_algorithm.env
source pure_algorithm.env
bash scripts/slurm/submit_pure_algorithm_pipeline.sh
```

提交器会建立两条并行依赖链：

```text
decoder cache -> decoder 4000-step 主训练 -> 100-step CLIPIQA 微调 --+
                                                                    +-> UDM10 评估
encoder cache -> encoder 3000-step 蒸馏 ----------------------------+
```

`DECODER_PARTITION` 应提供 H100 级别显存，`ENCODER_PARTITION` 可使用 L40/L40S，`EVAL_PARTITION` 必须稳定分配一张 L40S。分区名称由各集群自行填写。显存和时限可在 `sbatch` 模板中按集群规则调整，但不要修改模型结构、loss 权重、样本数和评估口径后仍直接对比上述数字。

## 7. 输出目录

```text
RUN_ROOT/
  cache/decoder/
  cache/encoder/
  models/decoder_main/checkpoint-4000/decoder.pt
  models/decoder_final/checkpoint-4100/decoder.pt
  models/encoder/checkpoint-3000/encoder.pt
  evaluation/baseline/
  evaluation/pred/
  evaluation/baseline_stage_profile.json
  evaluation/stage_profile.json
  evaluation/baseline_metrics/
  evaluation/candidate_metrics/
  evaluation/summary.json
  environment/
  logs/
  source-state.txt
  source.patch
```

缓存 manifest 中必须满足：decoder 和 encoder 均为 1024 条，`failures` 为空。训练日志位于各模型目录的 `train_log.jsonl`。

## 8. 验收检查

1. `stage_profile.json` 中确认没有 compile warmup，命令中没有 `--compile_vae`。
2. `05_evaluate.sbatch` 会在同一张卡上依次运行原始 DOVE 和候选模型；速度不能跨 GPU 型号比较。
3. 指标由同一版 `eval_metrics.py` 和 `eval_temporal_metrics.py` 计算。
4. PSNR/SSIM 相对原始 DOVE 降幅应小于 8%，CLIPIQA 应高于约 0.45。
5. 1.505x 只有约 1 秒余量，部署机器必须复测，不能直接作为保底 SLA。
6. 当前结果不是原生 4K 测试；真实 4K 仍需补显存、tile 边界、吞吐和长视频稳定性验证。

任务完成后也可以手工重建汇总：

```bash
python scripts/summarize_pure_algorithm_run.py "$RUN_ROOT"
```

## 9. 仅验证已有 checkpoint

已有 encoder/decoder checkpoint 时，可跳过四个训练任务，直接参考 `scripts/slurm/05_evaluate.sbatch` 的命令运行。checkpoint 内保存了 block profile，推理脚本会校验命令行 profile 与 checkpoint 是否一致，配置不匹配会直接报错。
