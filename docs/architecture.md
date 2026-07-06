# 项目讲解：理解 GPT-SoVITS-Pro 并准备二次开发

本文帮你建立 GPT-SoVITS-Pro 的完整心智模型——它由哪两个模型组成、数据如何在其中流动、训练与推理如何对应——读完应能自己定位改动点，进入二次开发。

## 文档职责

- 负责：核心架构心智模型、端到端数据流、训练/推理的代码对应关系、版本差异，以及"改某类功能去哪里"。
- 不负责：安装与首次跑通（见 [快速上手](quickstart.md)）；纯粹的入口/目录速查（见 [项目地图](project-map.md)）；逐行源码解读。
- 适用读者：已能跑通推理、准备改代码或接入自有系统的开发者。
- 阅读方式：先读"核心心智模型"建立骨架，再按需跳到推理、训练或某个模型的细节。文中 `file:line` 可直接跳转源码。

## 一、核心心智模型：两个解耦的模型 + 一座桥

整个系统只需记住一句话：**把"文本→语音"拆成两个独立训练、独立替换的模型，中间用一套共享码本对齐。**

```text
S1 · GPT / AR / Text2Semantic     只管「文本 → 语义」，不碰声音
      输入：音素 + BERT语义 + 参考语义前缀
      输出：semantic token 序列（离散整数，词表 1024）

              ↓  桥：codebook（1024 个 768 维向量）

S2 · SoVITS / VITS                只管「语义 + 音色 → 声学」，不认字义
      输入：semantic token + 参考音频音色
      输出：波形(v1/v2/v2Pro) 或 mel→声码器→波形(v3/v4)
```

理解这套结构，三个关键认知要先建立：

1. **两阶段是物理解耦的。** GPT 和 SoVITS 是两个独立权重文件（`.ckpt` 和 `.pth`），分别训练、可分别替换。推理时你同时选一个 GPT 权重和一个 SoVITS 权重。
2. **它们之间唯一的"共同语言"是 codebook 索引。** 一套 1024 个向量的码本被编码端和解码端共享：SoVITS 的量化器把连续语音特征离散成索引，GPT 就在这 1024 个"词"上做自回归续写，SoVITS 再把索引查表还原成连续向量继续解码。GPT 吐出的整数 = 码本索引 = SoVITS 吃进的整数。这正是两模型能解耦的原因。
3. **"音色"和"内容"走不同通道。** 内容（说什么）由文本经 GPT 变成 semantic token；音色（谁在说）由参考音频经音色编码器（v2Pro 还叠加 speaker embedding）单独提供。零样本克隆之所以成立，就是因为换参考音频只换音色通道，不动内容通道。

这套设计对应学术上的 AR-VITS / VALL-E 思路：GPT 是 VALL-E 式的自回归 decoder，SoVITS 是 VITS 声学模型（`GPT_SoVITS/AR/models/t2s_model.py:1` 注释注明改自 SoundStorm）。

## 二、端到端推理数据流

下面是"参考音频 + 目标文本 → 波形"的完整流转。每一步标注了模型/文件，方便按图索骥。

```text
参考音频(16k) ──cnhubert──> SSL 特征 [B,768,T]
                                │
              SoVITS.extract_latent = ssl_proj + 量化器编码
                                ▼
                        参考 semantic token（作为 GPT 前缀）
                                │
目标文本 ──文本前端──> 音素id + BERT特征 ──┐
                                          ▼
        ┌──────────── S1 · GPT 自回归 ────────────┐
        │  infer_panel：以参考语义为前缀，逐 token   │
        │  续写目标语义，遇 EOS(=1024) 停止          │
        └────────────────────────────────────────┘
                                │  pred_semantic（离散码序列）
                                ▼  ← 这里是 S1/S2 分界点
        ┌──────────── S2 · SoVITS 解码 ───────────┐
        │  量化器.decode(codes) → 连续向量          │
        │  enc_p(TextEncoder+MRTE 融合音色 ge)      │
        │  v1/v2/v2Pro: flow → Generator → 波形     │
        │  v3/v4:      DiT+CFM流匹配 → mel → 声码器  │
        └────────────────────────────────────────┘
                                │
                                ▼
                        后处理（防爆音/句间静音/可选超分）→ 输出波形
```

分阶段说明：

**文本前端**（`GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`）
把目标文本切句、检测语种、做 g2p（字→音素）、抽 BERT 特征。核心是 `get_phones_and_bert`（`TextPreprocessor.py:122`）：用 `LangSegmenter` 切分中英日韩粤混合文本，逐段调 g2p，**仅中文**取 roberta 特征并按 `word2ph` 重复到音素级，其余语言填零向量。切句策略见 `text_segmentation_method.py` 的 `cut0~cut5`。

**S1 · GPT 阶段**（`GPT_SoVITS/AR/models/t2s_model.py`）
参考音频先被转成 semantic token 作为"前缀"，GPT 在其后自回归续写目标语义。推理主路径 `infer_panel`（`t2s_model.py:966`）→ `infer_panel_naive`（`:816`），用 KV-cache 加速（首步 `process_prompt`、增量步 `decode_next_token`）。停止条件是采样到 EOS（值 1024，复用码本末位，`:913`），并有"至少生成约 10 个 token"的最短保护（`:900`）和硬上限。**S1 与 S2 的分界点**就是 `infer_panel` 返回的 `pred_semantic`（推理主控里的计时点 `TTS.py:1281`）。

**S2 · SoVITS 阶段**（`GPT_SoVITS/module/models.py`）
`decode`（`models.py:994`）先用量化器把 codes 还原成连续向量，经 `TextEncoder`（内含 MRTE 音色融合，`:156`）得到先验分布，再走归一化流 + Generator 出波形。v3/v4 改为产 mel 再交外接声码器（下详版本差异）。

**推理主控**：`GPT_SoVITS/TTS_infer_pack/TTS.py` 的 `TTS.run`（`:997`）串起上述全流程，`api_v2.py` 和新式推理服务都走它。而 `GPT_SoVITS/inference_webui.py` 是一份**等价的旧式独立实现**（自带 `get_tts_wav`），不经过 `TTS` 类，适合作为对照阅读理解。

## 三、两个模型的内部结构

### S1 · GPT（Text2SemanticDecoder，`t2s_model.py:260`）

一个 decoder-only 的自回归 Transformer：

```text
音素id ─ ar_text_embedding ┐
BERT   ─ bert_proj ─────────┤(相加)
参考+已生成语义 ─ ar_audio_embedding ┘
        │ + 位置编码
        ▼
   TransformerEncoder（causal mask，文本段全可见 / 语义段下三角）
        ▼
   ar_predict_layer → logits（词表 1024+1，含 EOS）
        ▼
   采样(top-k/top-p/温度/重复惩罚) → 下一个 semantic token
```

- 词表：`vocab_size = 1024 + 1`，`EOS = 1024`（`:273`），正好对齐 SoVITS 码本的 1024 个码。
- 训练封装在 `t2s_lightning_module.py:18`，`training_step` 按 `if_dpo` 选带 DPO 或纯交叉熵。
- 推理为省算力把训练用的标准 Transformer 权重搬进带 KV-cache 的 `T2SBlock`（`:88`）。

### S2 · SoVITS（三套 SynthesizerTrn，`models.py`）

| 类 | 行号 | 版本 | 直接输出 |
| --- | --- | --- | --- |
| `SynthesizerTrn` | `:829` | v1 / v2 / v2Pro / v2ProPlus | 波形（内置 Generator 声码器） |
| `SynthesizerTrnV3` | `:1215` | v3 | mel（外接 BigVGAN） |
| `SynthesizerTrnV3b` | `:1363` | v3 变体 | mel |

v1/v2/v2Pro 的解码路径（`decode`，`:994`）：

```text
codes ─量化器.decode─> 连续向量 ─enc_p(TextEncoder+MRTE, 注入音色ge)─> m_p,logs_p
      ─采样先验z_p─> flow(reverse) ─> z ─> Generator(g=ge) ─> 波形
```

核心枢纽是 `TextEncoder`（`:156`）里的 **MRTE**（Multi-Reference Timbre Encoder，`:201`），它用音色向量 `ge` 把内容流与音色对齐融合。音色 `ge` 来自参考谱经 `MelStyleEncoder`；**v2Pro/v2ProPlus 额外**把 eres2net 的 20480 维 speaker embedding 经 `sv_emb` 融进 `ge`（`:928-944`），这是它音色更像的原因。

### 桥：量化器（`module/quantize.py` + `core_vq.py`）

`ResidualVectorQuantizer(dimension=768, n_q=1, bins=1024)`（`models.py:925`）——单层残差、1024 码。一套码本同时服务编码与解码：

- 编码：`SynthesizerTrn.extract_latent`（`models.py:1094`）= `ssl_proj` + 量化器编码 → 出 codes，供 GPT 训练/前缀使用。
- 解码：`量化器.decode(codes)` 用 `F.embedding` 查表还原向量（`core_vq.py:231`）。

### 特征提取

- **内容/发音**：`feature_extractor/cnhubert.py` 的 `CNHubert`（`:22`），从 16k 波形出 SSL 特征 `[B,768,T]`，是内容表征的源头。
- **音色（v2Pro 专用）**：`sv.py` 的 `SV.compute_embedding3`（`:24`），eres2net 出 20480 维 speaker embedding。

## 四、微调训练流水线

微调是"少样本"路径，把原始录音变成一个专属音色模型。整条流水线由总控 WebUI（`webui.py`）用 subprocess 串起来，产物统一落在 `logs/<exp_name>/`。

```text
[原始录音]
  │ UVR5 人声分离(可选)  webui.py:305 → tools/uvr5/webui.py
  ▼
[干声] ─ 切分 webui.py:714 → tools/slice_audio.py ─ 降噪(可选) ─┐
                                                                 ▼
[切片wav] ─ ASR webui.py:377 → tools/asr/* ─ 打标校对 webui.py:275 → tools/subfix_webui.py
  ▼
[标注 .list]  每行：音频路径|说话人|语言|文本
  │
  ├─ 1Aa 文本    webui.py:808 → prepare_datasets/1-get-text.py
  │      → 2-name2text.txt（音素/word2ph/norm_text）, 3-bert/*.pt
  ├─ 1Ab 特征    webui.py:898 → 2-get-hubert-wav32k.py
  │      → 4-cnhubert/*.pt（SSL）, 5-wav32k/*.wav
  │      (Pro) webui.py:920 → 2-get-sv.py → 7-sv_cn/*.pt（speaker embedding）
  └─ 1Ac 语义    webui.py:992 → 3-get-semantic.py（内部调 extract_latent 量化）
         → 6-name2semantic.tsv（semantic token）
  ▼
[中间产物 logs/<exp>/]
  │
  ├─ 1Ba SoVITS 训练  webui.py:542 → s2_train.py（GAN：SynthesizerTrn + 判别器）
  │      吃 2-name2text / 4-cnhubert / 5-wav32k / 3-bert (+7-sv_cn)
  │      → SoVITS_weights_<ver>/<exp>_e<epoch>_s<step>.pth
  └─ 1Bb GPT 训练     webui.py:636 → s1_train.py（PyTorch Lightning）
         吃 6-name2semantic.tsv + 2-name2text.txt
         → GPT_weights_<ver>/<exp>-e<epoch>.ckpt
  ▼
[微调模型] → 回到推理(1C)，同时选新的 GPT.ckpt + SoVITS.pth
```

理解这条流水线的两个要点：

- **数据准备产物精确对应两个模型的输入。** GPT（S1）只吃 `6-name2semantic.tsv`（语义）+ `2-name2text.txt`（音素）；SoVITS（S2）吃声学相关的 `4-cnhubert`/`5-wav32k`/`3-bert`（v2Pro 还有 `7-sv_cn`）。这正是两阶段解耦在训练侧的体现。
- **步骤有依赖顺序**：`1c` 依赖 `1b` 产出的 `4-cnhubert`，故必须 `1a → 1b → 1c`（一键三连 `open1abc`，`webui.py:1046`，已内置顺序）。

参数传递有两种机制，二次开发时要注意：

```text
数据准备脚本 → 参数写进 os.environ 再 Popen（脚本内 os.environ.get 读回）
训练脚本     → 参数注入模板配置后写 TEMP/tmp_s{1,2}.* ，再 --config 传入
```

## 五、版本差异（v1 / v2 / v2Pro / v2ProPlus / v3 / v4）

版本是二次开发绕不开的分叉。默认版本硬编码在 `webui.py:4`（`v2Pro`），推理侧从权重路径自动探测版本（`change_sovits_weights`，`inference_webui.py:261`）。

```text
v1 / v2 ───── SoVITS 直出波形（VITS + 内置 Generator 声码器）
              v2 优化了文本前端、扩到 5k 小时底模，支持韩语/粤语

v2Pro/ProPlus ─ 在 v2 上叠加 eres2net speaker embedding，音色更像
              硬件成本与速度接近 v2，需要 7-sv_cn 数据与 sv 底模

v3 / v4 ────── 架构换代：SoVITS 只产 mel，再经 DiT+CFM 流匹配 + 外接声码器
              v3 → BigVGAN，原生 24k；v4 → HiFiGAN 式 Generator，原生 48k
              音色更偏向参考音频而非整体训练集；对低质训练集不如 v1/v2/v2Pro
```

选择经验（来自 README 发布说明）：训练集质量一般时 v1/v2/v2Pro 更稳；追求高相似度且有干净参考音频时考虑 v3/v4。版本相关的模型分支集中在 `TTS.py:493-687`（加载）与 `models.py`（三套 SynthesizerTrn）。

## 六、代码地图：改某类功能去哪里

先看 [项目地图](project-map.md) 的目录职责总表；下面按"任务 → 起点"给出二次开发常见改动点。

| 你想改的东西 | 先看这里 |
| --- | --- |
| 推理管线 / 批处理 / 流式 | `TTS_infer_pack/TTS.py`（`run` `:997`），旧式对照 `inference_webui.py` |
| 文本切分 / 多语言发音 / g2p | `TextPreprocessor.py:122`、`text_segmentation_method.py`、`GPT_SoVITS/text/` |
| GPT 采样策略 / 停止条件 / 结构 | `AR/models/t2s_model.py`（`infer_panel_naive` `:816`），训练 `s1_train.py` |
| SoVITS 声学 / 音色融合 / 声码器 | `module/models.py`（`SynthesizerTrn` `:829` / `decode` `:994`），训练 `s2_train.py` |
| semantic token / 码本 | `module/quantize.py`、`module/core_vq.py`、`extract_latent`（`models.py:1094`） |
| 数据准备产物 | `GPT_SoVITS/prepare_datasets/`，及 `webui.py` 里的启动点 |
| WebUI 流程 / 进程编排 | `webui.py`（`open1a/1b/1c` `:780/870/960`，`open1Ba/1Bb` `:489/590`） |
| API 行为 | v1 `api.py`；v2 `api_v2.py` + `TTS.py` |
| 端口 / 精度 / 路径 / 设备探测 | `config.py`（端口 `:140`，设备/精度 `:149-195`，版本→路径 `:12-75`） |

## 七、二次开发坑点

动手前先知道这几处"反直觉"的地方，能省不少调试时间：

- **`is_half` 会被覆盖。** 环境变量传入的 `is_half`（`config.py:127`）在 `config.py:195` 被 GPU 自动探测结果覆盖为"是否存在 fp16 设备"。想强制精度得改探测逻辑，而非只设环境变量。
- **默认版本硬编码。** `webui.py:4` 写死 `version="v2Pro"`，改默认版本要动这里。
- **两套推理实现并存。** `TTS_infer_pack/TTS.py`（新式，被 API 用）和 `inference_webui.py`（旧式，自包含）算法等价但代码独立，改推理逻辑要确认改的是哪一套，必要时两边同步。
- **数据准备靠环境变量传参。** 不是命令行 argv，而是 `os.environ.update(config)`（`webui.py:807`）后 Popen。脱离 WebUI 单独调用这些脚本时，要自己注入环境变量。
- **不要擅自加兜底。** 参考音频时长校验、NaN 过滤、精度回退等是有意的快速失败点，改动时保持错误可见，便于定位。

## 相关文档

- [快速上手](quickstart.md)：安装、跑通第一条零样本合成。
- [项目地图](project-map.md)：入口、目录职责、优先阅读路径速查。
- [README](../README.md)：功能总览、各版本发布说明、模型下载地址。
