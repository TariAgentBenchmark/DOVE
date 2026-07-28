# DOVE 纯算法 1.5x 复现说明

本仓库基于原始 DOVE 项目扩展；上游模型、数据集和基础训练说明保存在 [README.upstream.md](README.upstream.md)。

## 1. 复现范围

本说明复现最终采用的单卡纯算法方案：保留 DOVE 的 one-step latent Transformer，用蒸馏后的轻量 VAE encoder 和 decoder 替换原始 VAE。

- Encoder：四个 down block 均保留 1 层，mid block 保留 1 层，即 `[1,1,1,1] + mid=1`。
- Decoder：四个 up block 使用非均匀深度 `[1,1,1,2]`，高分辨率 block 多保留一层。
- Decoder teacher：完整 DOVE 轨迹产生的 decoder 输入 latent 和原始 decoder RGB 输出。
- Encoder teacher：真实退化 LQ 条件及原始 encoder posterior moments。
- 感知补偿：用最终轻量 encoder 重建 decoder trajectory cache，再以 GT 为感知目标完成三段确定性 continuation。
- 不使用 `torch.compile`、TensorRT、CUDA Graph、多卡推理或减少输出帧数。

该路线不是去 VAE。此前的 pixel/L2P、structured diffusion 和多卡 tile 路线已被实验否决，不属于最终复现范围。

## 2. 已验证结果

验证集为 UDM10，输入 32 帧 `180x318`，输出 32 帧 `720x1272`。最终候选已完成全量质量、时序和视觉核验：

| 模型 | PSNR | SSIM | LPIPS ↓ | DISTS ↓ | CLIPIQA ↑ | RAFT warp ↓ | 帧差误差 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5012 | 0.010368 | 0.024123 |
| 最终轻量 VAE | 26.0497 | 0.7638 | **0.2700** | **0.1485** | **0.5509** | 0.010530 | 0.024242 |

正式任务 `2975014` 在同一张 NVIDIA L40S 上依次运行原始 DOVE 与候选，双方关闭 `torch.compile`。稳态端到端耗时为 `416.59 s -> 273.00 s`，即 **`1.526x`**；核心推理为 `403.49 s -> 248.56 s`，即 `1.623x`。速度、感知质量和时序门槛全部通过。详细实验记录见 [pure_algorithm_results.md](docs/experiments/pure_algorithm_results.md)。

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
    train_decoder.py         # decoder 蒸馏与 GT/感知补偿
    build_encoder_cache.py   # 退化 LQ encoder cache
    train_encoder.py         # encoder posterior 蒸馏
scripts/slurm/
  submit_pure_algorithm_pipeline.sh
  01-04_*.sbatch             # 基础 decoder/encoder cache 与蒸馏
  06_build_perceptual_cache.sbatch
  07_train_perceptual_sweep.sbatch
  09_train_perceptual_refine.sbatch
  11_train_perceptual_finish.sbatch
  13_eval_perceptual_final.sbatch
scripts/summarize_perceptual_sweep.py
tools/generate_dove_experiment_report.py
```

## 4. 环境与硬件

- Linux + CUDA，Python 3.11。
- Decoder cache 和训练建议 1x H100 80 GB；encoder cache/训练可使用 1x L40/L40S 48 GB。
- 最终速度必须在单张 L40S 上测量；训练卡不参与速度结论。
- 建议准备至少 100 GB 可写空间，用于三类 cache、checkpoint、运行时缓存和 UDM10 输出。
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

主提交器只执行最终选定的确定性路径，实验 sweep 中被否决的候选不会重复训练：

```text
原始 decoder cache -> decoder 4000 + 100 step ----------------------+
                                                                    |
encoder cache -> encoder 3000 step -> encoder-matched decoder cache -+
                                                                    v
perceptual 200 step -> lpips_focus 100 step -> lpips_continue 100 step
                                                                    |
                                                                    v
                  原始 DOVE 与最终候选同卡 UDM10 速度/质量/时序评估
```

选定的数组索引固定为 `perceptual=1`、`lpips_focus=2`、`lpips_continue=0`。`DECODER_PARTITION` 应提供 H100 级别显存，`ENCODER_PARTITION` 可使用 L40/L40S，`EVAL_PARTITION` 必须稳定分配一张 L40S。分区名称由各集群自行填写。显存和时限可按集群规则调整，但不要修改模型结构、loss 权重、样本数或评估口径后仍直接对比上述数字。

## 7. 输出目录

```text
RUN_ROOT/
  cache/decoder/
  cache/encoder/
  cache/decoder_encoder_matched/
  models/decoder_main/checkpoint-4000/decoder.pt
  models/decoder_final/checkpoint-4100/decoder.pt
  models/perceptual/checkpoint-4300/decoder.pt
  models/encoder/checkpoint-3000/encoder.pt
  refine/models/lpips_focus/checkpoint-4400/decoder.pt
  finish/models/lpips_continue/checkpoint-4500/decoder.pt
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

三个缓存 manifest 均应记录 1024 条样本且 `failures` 为空。训练日志位于各模型目录的 `train_log.jsonl`，checkpoint 内同时保存结构 profile 和训练 objective。

## 8. 验收检查

1. `stage_profile.json` 中确认 `compile_vae=false`，命令中没有 `--compile_vae`。
2. `13_eval_perceptual_final.sbatch` 会在同一张卡上依次运行原始 DOVE 和最终候选；速度不能跨 GPU 型号比较。
3. 指标由同一版 `eval_metrics.py` 和 `eval_temporal_metrics.py` 计算。
4. 必须同时满足：速度 `>=1.5x`、PSNR `>=25.5`、LPIPS/DISTS 不高于 DOVE、CLIPIQA 不低于 DOVE、两项时序误差恶化不超过 `5%`。
5. 即使实验通过，部署机器仍需复测，不能直接把单次集群测量作为固定 SLA。
6. 当前结果不是原生 4K 测试；真实 4K 仍需补显存、tile 边界、吞吐和长视频稳定性验证。

任务完成后也可以手工重建汇总：

```bash
python scripts/summarize_pure_algorithm_run.py "$RUN_ROOT"
```

## 9. 仅验证已有 checkpoint

已有最终 encoder/decoder checkpoint 时，可设置 `ENCODER_CHECKPOINT`、`FINAL_DECODER_CHECKPOINT`，再运行：

```bash
bash scripts/slurm/submit_perceptual_final.sh
```

checkpoint 内保存了 block profile，推理脚本会校验命令行 profile 与 checkpoint 是否一致，配置不匹配会直接报错。正式验收完成后，可用 `tools/generate_dove_experiment_report.py` 读取 `evaluation/summary.json`、筛选指标 JSON 和视觉对比图生成 Word 报告；生成器会在任一客户门槛不通过时直接拒绝输出。
