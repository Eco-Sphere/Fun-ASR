# FunASR 昇腾 NPU 适配报告

## 1. 适配概述

将 FunASR 1.3.1 语音识别框架从 CUDA 适配到华为昇腾 NPU，覆盖全部 8 个代表性模型的推理路径。

- **适配日期**: 2026-03-31
- **适配难度**: 低（仅修改 3 个文件，共新增 ~8 行代码）
- **验证结果**: 全部 8 个模型 CPU/NPU 推理通过，文本输出一致

## 2. 环境配置

| 项目 | 版本/配置 |
|------|-----------|
| OS | Linux aarch64 (openEuler 22.03 SP4) |
| Python | 3.10 |
| PyTorch | 2.8.0+cpu |
| torch_npu | 2.8.0 |
| torchaudio | 2.8.0 |
| FunASR | 1.3.1 |
| Conda 环境 | torch280_py310_ali (克隆自 torch280_py310_diffusion) |
| NPU 设备 | ASCEND_RT_VISIBLE_DEVICES=4,5 |

## 3. 代码修改清单

### 3.1 `funasr/auto/auto_model.py` — inference() empty_cache

在 `inference()` 方法末尾（原行 430-433）增加 NPU 缓存释放分支：

```diff
         device = next(model.parameters()).device
         if device.type == "cuda":
             with torch.cuda.device(device):
                 torch.cuda.empty_cache()
+        elif device.type == "npu":
+            import torch_npu
+            torch.npu.empty_cache()
         return asr_result_list
```

### 3.2 `funasr/frontends/fused.py` — FusedFrontends 设备选择

在设备检测链中 cuda 之后插入 npu（原行 79-86）：

```diff
         if torch.cuda.is_available():
             dev = "cuda"
+        elif hasattr(torch, 'npu') and torch.npu.is_available():
+            dev = "npu"
         elif torch.xpu.is_available():
             dev = "xpu"
```

### 3.3 `funasr/models/sense_voice/whisper_lib/__init__.py` — load_model 默认设备

扩展默认设备选择逻辑（原行 126-127）：

```diff
     if device is None:
-        device = "cuda" if torch.cuda.is_available() else "cpu"
+        if torch.cuda.is_available():
+            device = "cuda"
+        elif hasattr(torch, 'npu') and torch.npu.is_available():
+            device = "npu"
+        else:
+            device = "cpu"
```

## 4. 验证结果

### 4.1 模型清单

| # | 模型 | ModelScope ID | 类型 |
|---|------|---------------|------|
| 1 | SenseVoiceSmall | iic/SenseVoiceSmall | 多任务 ASR |
| 2 | Paraformer-zh | paraformer-zh (seaco) | 中文 ASR |
| 3 | Paraformer-en | paraformer-en | 英文 ASR |
| 4 | Paraformer-zh-streaming | paraformer-zh-streaming | 流式中文 ASR |
| 5 | FSMN-VAD | fsmn-vad | 语音活动检测 |
| 6 | CT-Punc | ct-punc | 标点恢复 |
| 7 | Fun-ASR-Nano | FunAudioLLM/Fun-ASR-Nano-2512 | LLM-based ASR |
| 8 | CAM++ | cam++ | 声纹识别 |

### 4.2 CPU vs NPU 推理对比

| 模型 | 语言 | CPU 耗时 | NPU 耗时 | 加速比 | 输出文本 |
|------|------|---------|---------|--------|---------|
| SenseVoice | zh | 32.49s | 6.34s | 5.1x | 开放时间早上9点至下午5点。 |
| SenseVoice | en | 32.68s | 0.45s | 72.6x | The tribal chieftain called for the boy... |
| SenseVoice | ja | 32.94s | 0.46s | 71.6x | うち の 中学 は 弁当 制 で... |
| SenseVoice | ko | 33.57s | 0.37s | 90.7x | 조 금만 생각 을 하 면서... |
| SenseVoice | yue | 30.81s | 0.37s | 83.3x | 呢几个字都表达唔到... |
| Paraformer-zh | zh | 37.03s | 5.17s | 7.2x | 欢迎大家来到么哒社区进行体验。 |
| Paraformer-en | en | 28.52s | 0.31s | 92.0x | procedure telegram impatience... |
| Paraformer-streaming | zh | 244.87s | 2.17s | 112.8x | 菜饭时间早上九点至下午五点 |
| Fun-ASR-Nano | zh | 562.70s | 5.15s | 109.3x | 开饭时间早上九点至下午五点。 |
| CAM++ | spk | ~20s | 2.90s | ~6.9x | same=0.6936, diff=-0.0842 |

**文本输出一致性**: CPU 和 NPU 输出文本完全一致。

**加速效果**: NPU 相比 CPU 平均加速约 **65 倍**，其中首次推理含算子编译时间较长（SenseVoice zh 6.34s），后续推理极快（0.31-0.46s）。

## 5. 已知限制

1. **SenseVoice timing/DTW**: NPU 上 `x.is_cuda=False`，时间对齐自动 fallback 到 CPU Numba，功能正确
2. **Triton 算子**: `triton_ops.py` 的 CUDA Triton kernel 在 NPU 上不可用，自动 fallback 到 PyTorch sort
3. **训练路径**: `trainer.py` / `trainer_ds.py` 中大量 `torch.cuda.amp` 未适配 NPU，本次仅适配推理路径
4. **FusedFrontends**: 虽然已适配，但 SenseVoice/Paraformer 标准推理路径不经过此模块
5. **Fun-ASR-Nano**: 需要指定 `remote_code` 的绝对路径

## 6. 文件清单

| 文件 | 说明 |
|------|------|
| `run_cpu.py` | CPU 推理验证脚本（8 模型全量测试） |
| `run_npu.py` | NPU 推理验证脚本（8 模型全量测试） |
| `cpu_results.json` | CPU 推理结果 |
| `npu_results.json` | NPU 推理结果 |
| `FunASR_NPU_REPORT.md` | 本报告 |
| `README_NPU.md` | 快速开始指南 |
