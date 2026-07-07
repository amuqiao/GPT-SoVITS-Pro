# 快速上手：从 0 跑通第一条零样本语音

本文带你从一台空环境开始，启动 GPT-SoVITS-Pro 的推理网页，并用一段参考音频合成第一条语音。

## 这篇文档带你完成什么

完成后你应该得到三个结果：

```text
环境可用 -> 浏览器能打开推理页 -> 听到第一条合成语音
```

本文只走**零样本推理**：不训练模型，只用底模和一段 3~10 秒参考音频直接合成。它适合第一次验证项目能不能跑通，也适合快速试一个音色。少样本微调、批量制作数据集、API 接入和二次开发不放在本文里。

## 先建立心智模型：第一条语音由四件事组成

不要一开始就找训练按钮。第一次跑通时，你只需要让推理页同时拿到四类输入：

```text
1. GPT 底模        -> 负责把文字变成语义
2. SoVITS 底模     -> 负责把语义和音色变成声音
3. 参考音频 + 文本  -> 告诉模型“像谁说话”
4. 目标文本 + 语种  -> 告诉模型“要说什么”
```

网页里的合成链路可以这样理解：

```text
参考音频(3~10 秒) + 参考音频对应文字
        │
        ├── 锚定音色、语气和发音特征
        │
目标文本 + 目标语种
        │
        ├── 决定这次要生成的内容
        ▼
推理 WebUI(9872) -> 合成语音 -> 试听 / 下载
```

零样本不需要训练集、不需要 `logs/` 实验目录，也不会生成新的 GPT / SoVITS 权重。

## 第 0 步：确认你在哪里运行

先进入项目根目录：

```bash
cd /Users/admin/Downloads/GPT-SoVITS-Pro
```

如果项目运行在远程开发服务器，本地浏览器不能直接访问服务器的 `localhost`。你需要在本地开 SSH 端口转发：

```bash
ssh -N -L 9872:localhost:9872 <user>@<server-ip>
```

然后在本地浏览器访问：

```text
http://127.0.0.1:9872/
```

`0.0.0.0` 是服务监听地址，不建议作为浏览器访问地址。浏览器侧优先用 `127.0.0.1` 或 `localhost`。

如果你想从总控 WebUI 打开推理页，还需要同时转发总控端口：

```bash
ssh -N -L 9874:localhost:9874 -L 9872:localhost:9872 <user>@<server-ip>
```

## 第 1 步：创建 Python 环境

推荐使用 Python 3.10：

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
```

确认 Python 指向当前环境：

```bash
python --version
```

## 第 2 步：安装依赖和底模

按你的系统选择一个命令。安装脚本会安装依赖，并下载主底模和中文 G2PW 模型。

Linux：

```bash
bash install.sh --device <CU126|CU128|ROCM|CPU> --source <HF|HF-Mirror|ModelScope>
```

macOS：

```bash
bash install.sh --device <MPS|CPU> --source <HF|HF-Mirror|ModelScope>
```

Windows PowerShell：

```pwsh
pwsh -F install.ps1 --Device <CU126|CU128|CPU> --Source <HF|HF-Mirror|ModelScope>
```

参数怎么选：

| 参数 | 什么时候选 |
| --- | --- |
| `CU126` / `CU128` | NVIDIA GPU，CUDA 版本匹配对应 PyTorch wheel |
| `ROCM` | AMD GPU 的 ROCm 环境 |
| `MPS` | Apple silicon Mac |
| `CPU` | 没有可用 GPU，或只想先跑通 |
| `HF` | 能稳定访问 Hugging Face |
| `HF-Mirror` | 国内网络优先尝试 |
| `ModelScope` | 国内网络优先尝试 |

本文不需要 UVR5，所以先不要加 `--download-uvr5` / `--DownloadUVR5`。UVR5 是做歌曲人声分离、微调素材清洗时才用的。

再安装 FFmpeg：

```bash
conda install ffmpeg
```

Ubuntu 也可以使用系统包：

```bash
sudo apt install ffmpeg libsox-dev
```

macOS 也可以使用 Homebrew：

```bash
brew install ffmpeg
```

## 第 3 步：确认底模文件存在

零样本推理必须有底模。安装成功后，重点确认这些目录存在：

```text
GPT_SoVITS/pretrained_models/
GPT_SoVITS/text/G2PWModel/
```

常用 v2Pro 底模路径是：

```text
GPT_SoVITS/pretrained_models/s1v3.ckpt
GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
```

如果安装脚本没有下载成功，需要手动补齐：

| 模型 | 放置位置 |
| --- | --- |
| GPT / SoVITS 主底模 | `GPT_SoVITS/pretrained_models/` |
| 中文 G2PW 模型 | 解压并命名为 `G2PWModel`，放到 `GPT_SoVITS/text/` |

中文合成缺少 `G2PWModel` 时，通常会在文本前端或注音阶段报错。

## 第 4 步：启动推理 WebUI

最短路径是直接启动推理页：

```bash
conda activate GPTSoVits
python GPT_SoVITS/inference_webui.py zh_CN
```

启动成功后，推理页默认监听 `9872`。

如果你更想从总控页进入：

```bash
conda activate GPTSoVits
python webui.py zh_CN
```

浏览器打开：

```text
http://127.0.0.1:9874/
```

然后进入：

```text
1-GPT-SoVITS-TTS -> 1C-推理 -> 开启 TTS 推理 WebUI
```

再打开：

```text
http://127.0.0.1:9872/
```

常用端口：

| 端口 | 用途 | 本文是否必需 |
| --- | --- | --- |
| `9872` | 推理 WebUI | 必需 |
| `9874` | 总控 WebUI | 可选 |
| `9873` | UVR5 人声分离 | 不需要 |
| `9871` | 标注校对 WebUI | 不需要 |

## 第 5 步：准备一段参考音频

准备一段清晰人声：

```text
时长：3~10 秒
内容：最好是一整句自然说话
质量：单人声、无背景音乐、无明显噪声、不要多人混说
格式：wav / mp3 等常见音频格式均可，wav 更稳
```

同时写下这段参考音频里说的原文。第一次跑通不要开启“无参考文本模式”，直接填准确参考文本，排错成本最低。

示例：

```text
参考音频：ref.wav
参考文本：今天的天气不错，我们出去走走吧。
参考语种：中文
```

## 第 6 步：在网页里合成第一条语音

打开：

```text
http://127.0.0.1:9872/
```

页面语言可能显示中文，也可能显示英文。按下面顺序填写：

| 顺序 | 中文字段 | 英文字段 | 第一次建议 |
| --- | --- | --- | --- |
| 1 | `GPT模型列表` | `GPT weight list` | 选择底模，例如 `不训练直接推v3底模` / `Use v3 base model directly without training` |
| 2 | `SoVITS模型列表` | `SoVITS weight list` | 选择底模，例如 `不训练直接推v2Pro底模` / `Use v2Pro base model directly without training` |
| 3 | `请上传3~10秒内参考音频` | `Please upload a reference audio within the 3-10 second range` | 上传第 5 步准备的音频 |
| 4 | `参考音频的文本` | `Text for reference audio` | 填参考音频逐字对应文本 |
| 5 | `参考音频的语种` | `Language for reference audio` | 中文音频选 `中文` / `Chinese` |
| 6 | `需要合成的文本` | `Inference text` | 填你想合成的新文本 |
| 7 | `需要合成的语种` | `Inference text language` | 中文目标文本选 `中文` / `Chinese` |
| 8 | `怎么切` | `How to slice the sentence` | 第一次保持默认 |
| 9 | `top_k` / `top_p` / `temperature` | 同名 | 第一次保持默认 |
| 10 | `合成语音` | `Start inference` | 点击生成 |

不要在第一次测试时同时调整采样参数、语速、句间停顿和多参考音频。先用默认参数合成一条，确认链路可用。

## 第 7 步：判断是否跑通

满足下面三点就算完成快速上手：

```text
1. 页面没有报“模型缺失”或“参考音频时长超出范围”
2. 输出区域出现可播放音频
3. 音频内容是你的目标文本，音色接近参考音频
```

零样本第一次的目标是“听到可用声音”，不是追求最像。想更像，通常需要更干净的参考音频，或者进入少样本微调流程。

## 常见问题

### 浏览器打不开 `9872`

先确认服务是否启动。远程服务器场景下，确认本地 SSH 端口转发包含 `9872`：

```bash
ssh -N -L 9872:localhost:9872 <user>@<server-ip>
```

浏览器访问 `http://127.0.0.1:9872/`，不要依赖 `http://0.0.0.0:9872/`。

### 提示参考音频超出范围

推理页要求参考音频在 3~10 秒内。重新裁剪参考音频，保留一句完整、清晰、自然的话。

### 提示没有上传参考音频

必须上传左侧单个参考音频。多参考音频上传是可选项，第一次不用。

### 中文文本报错

检查 `GPT_SoVITS/text/G2PWModel/` 是否存在。中文 TTS 需要这个模型做注音相关处理。

### 权重列表里没有你想选的模型

点击页面上的 `刷新模型路径` / `refreshing model paths`。底模应位于 `GPT_SoVITS/pretrained_models/`，微调权重应位于对应的 `GPT_weights*` 和 `SoVITS_weights*` 目录。

### 显存不足

快速上手可以先用 CPU 或更小显存压力的底模跑通。推理速度会慢，但能验证安装和页面流程。

## 完成后你可以做什么

你已经跑通了零样本推理链路。接下来通常有三种方向：

```text
想更像、更稳定 -> 做少样本微调
想接入程序     -> 启动 API 服务
想批量生成     -> 使用 CLI 或自行封装推理调用
```
