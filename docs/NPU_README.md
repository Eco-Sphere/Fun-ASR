# FunASR 昇腾 NPU 推理指南

## 快速开始

### 1. 环境准备

```bash
conda activate torch280_py310_ali
export ASCEND_RT_VISIBLE_DEVICES=4,5
```

### 2. 安装 FunASR

```bash
cd /data1/z00879328/01_ALI/00_FunASR/FunASR
pip install -e ".[all]" --no-deps
pip install scipy librosa kaldiio torch_complex sentencepiece jieba pytorch_wpe \
    editdistance oss2 umap-learn jaconv hydra-core tensorboardX requests jamo \
    soundfile torch_optimizer fairscale transformers openai-whisper
```

### 3. 运行推理

**NPU 推理**:
```bash
cd /data1/z00879328/01_ALI/00_FunASR
python run_npu.py
```

**CPU 推理**:
```bash
python run_cpu.py
```

## 使用示例

### SenseVoiceSmall（多任务 ASR，支持 zh/en/ja/ko/yue）

```python
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4,5"
import torch_npu
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="npu:0",
    hub="ms",
)
res = model.generate(
    input="your_audio.wav",
    cache={},
    language="auto",
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
```

### Paraformer-zh（中文 ASR + VAD + 标点）

```python
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device="npu:0",
    hub="ms",
)
res = model.generate(input="your_audio.wav", cache={})
print(res[0]["text"])
```

### Fun-ASR-Nano（LLM-based ASR）

```python
nano_code = "/data1/z00879328/01_ALI/00_FunASR/FunASR/examples/industrial_data_pretraining/fun_asr_nano/model.py"
model = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    trust_remote_code=True,
    remote_code=nano_code,
    device="npu:0",
    hub="ms",
)
res = model.generate(input=["your_audio.wav"], cache={}, batch_size=1, language="中文", itn=True)
print(res[0]["text"])
```

### CAM++（声纹识别）

```python
import torch
model = AutoModel(model="cam++", device="npu:0", hub="ms")
res1 = model.generate(input="speaker1.wav", cache={})
res2 = model.generate(input="speaker2.wav", cache={})
emb1 = res1[0]["spk_embedding"].reshape(1, -1).float()
emb2 = res2[0]["spk_embedding"].reshape(1, -1).float()
similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
print(f"Speaker similarity: {similarity:.4f}")
```

## 支持的模型

| 模型 | 设备 | 状态 |
|------|------|------|
| SenseVoiceSmall | NPU | 通过 |
| Paraformer-zh | NPU | 通过 |
| Paraformer-en | NPU | 通过 |
| Paraformer-zh-streaming | NPU | 通过 |
| FSMN-VAD | NPU | 通过（作为子模型） |
| CT-Punc | NPU | 通过（作为子模型） |
| Fun-ASR-Nano | NPU | 通过 |
| CAM++ | NPU | 通过 |

## 注意事项

1. 首次推理包含 NPU 算子编译时间（约 5-10s），后续推理极快
2. 下载模型权重前请 unset proxy：`unset https_proxy http_proxy ALL_PROXY`
3. Fun-ASR-Nano 需要指定 `remote_code` 的绝对路径
4. 模型权重通过 modelscope 自动下载，缓存在 `~/.cache/modelscope/hub/`
