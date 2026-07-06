# 快速上手：从安装到跑通第一条语音

本文带你在本地把 GPT-SoVITS-Pro 跑起来，并合成出第一条克隆语音。目标是**最短路径跑通**，不追求覆盖全部功能。

## 文档职责

- 负责：环境安装、模型下载、启动 WebUI、跑通一次**零样本（zero-shot）**推理，并指出下一步（微调、API、CLI）该去哪。
- 不负责：模型原理、目录职责、二次开发。想理解"它为什么这样工作、我该改哪里"，读 [项目讲解](architecture.md)；想快速定位入口文件，读 [项目地图](project-map.md)。
- 适用读者：想用它做语音合成 / 声音克隆的新用户，具备基本命令行和 conda 使用能力。

## 先理解一件事：你不一定需要训练

这是最容易走弯路的地方。先看清两条路的边界，再决定跑哪条：

```text
零样本 zero-shot ── 只给 5 秒参考音频 ──> 直接合成，无需训练
   适用：随手克隆一个音色、快速验证效果
   需要：只装环境 + 下载底模

少样本 few-shot ── 用 ~1 分钟数据微调 ──> 音色更像、更稳
   适用：认真做一个专属音色
   需要：环境 + 底模 + 一整套数据准备与训练流程
```

**本文只跑通零样本这条路**，它足以让你听到效果。微调路径在文末给出入口。

整条零样本链路可以先建立这样一个心智画面：

```text
参考音频(5~10s) + 要合成的文本
        │
        ▼
   推理 WebUI (端口 9872)
        │  GPT 底模：文本 → 语义
        │  SoVITS 底模：语义 + 参考音色 → 波形
        ▼
   合成音频（下载 / 试听）
```

## 第 1 步：确认环境

推荐组合（其余组合见 [README 的 Tested Environments](../README.md)）：

```text
Python 3.10  +  PyTorch 2.5.1  +  CUDA 12.4（N 卡）
Python 3.10  +  PyTorch 2.5.1  +  Apple silicon（Mac 用 MPS 或 CPU）
```

> Windows 用户可选更省事的路径：下载官方**整合包**，双击 `go-webui.bat` 即可，跳过下面的安装步骤。整合包地址见 [README](../README.md#windows)。

## 第 2 步：安装

先建 conda 环境，再按平台运行安装脚本。安装脚本会**自动下载底模**，装成功即可跳过第 3 步。

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
```

| 平台 | 命令 |
| --- | --- |
| Linux | `bash install.sh --device <CU126\|CU128\|ROCM\|CPU> --source <HF\|HF-Mirror\|ModelScope> [--download-uvr5]` |
| macOS | `bash install.sh --device <MPS\|CPU> --source <HF\|HF-Mirror\|ModelScope> [--download-uvr5]` |
| Windows | `pwsh -F install.ps1 --Device <CU126\|CU128\|CPU> --Source <HF\|HF-Mirror\|ModelScope> [--DownloadUVR5]` |

参数说明：

- `--source`：模型下载源。国内网络优先 `HF-Mirror` 或 `ModelScope`。
- `--download-uvr5`（可选）：额外下载人声分离模型，**只有做微调、需要从歌曲里剥人声时才需要**，跑零样本可以不加。

还需要 FFmpeg（音频读写依赖）：

```bash
conda activate GPTSoVits
conda install ffmpeg          # 最省事，跨平台
# 或 Ubuntu: sudo apt install ffmpeg libsox-dev
# 或 macOS:  brew install ffmpeg
```

## 第 3 步（可选）：手动补底模

**只有 `install.sh` 没成功下载模型时才需要这步。** 底模是零样本推理的必需品，缺了无法合成。

- 主底模 → 放到 `GPT_SoVITS/pretrained_models/`（GPT `s1*.ckpt` + SoVITS `s2G*.pth`）。
- G2PW 中文注音模型 → 解压重命名为 `G2PWModel`，放到 `GPT_SoVITS/text/`（**仅中文 TTS 需要**）。

下载地址与各版本文件清单见 [README 的 Pretrained Models 章节](../README.md#pretrained-models)。默认版本是 `v2Pro`，对应底模 `v2Pro/s2Gv2Pro.pth` 与 `s1v3.ckpt`。

## 第 4 步：启动 WebUI

```bash
python webui.py           # 可选追加语言，如 python webui.py zh_CN
```

启动后浏览器访问总控 WebUI（默认端口 **9874**）。各端口一览：

```text
9874  总控 WebUI（webui.py）——数据处理 / 训练 / 推理的入口
9872  推理 WebUI            ——本文要用的
9873  UVR5 人声分离（可选）
9871  打标校对（可选）
9880  API 服务（见文末）
```

在总控 WebUI 里进入 `1-GPT-SoVITS-TTS` → `1C-inference`，点击开启推理 WebUI，页面会打到 9872 端口。

> 也可以直接起推理页：`python GPT_SoVITS/inference_webui.py`

## 第 5 步：跑通第一条零样本合成

在推理 WebUI 里，按顺序做四件事：

```text
1. 选模型      —— GPT 权重选底模，SoVITS 权重选底模（如 v2Pro 底模）
2. 传参考音频   —— 一段 5~10 秒的清晰人声（这段决定音色）
3. 填参考文本   —— 参考音频对应的文字 + 它的语言
4. 填目标文本   —— 你想合成的文字 + 它的语言，点击合成
```

几秒后即可试听 / 下载。听到声音，说明整条链路已经跑通。

> **参考音频时长必须在 3~10 秒之间**，这是代码里的硬约束（超出会报错）。想让音色更像，选一段干净、无背景音、语气自然的样本。

## 常见问题排查

按"卡在哪一步"对号入座：

- 启动即报模型缺失 → 回第 3 步补底模，确认路径在 `GPT_SoVITS/pretrained_models/`。
- 中文合成报注音相关错误 → 缺 `G2PWModel`，见第 3 步。
- 参考音频报时长错误 → 裁到 3~10 秒。
- 显存不足 / N 卡老旧 → 用 CPU 或半精度；`config.py` 会自动探测设备与精度。
- Mac 上音质偏低 → 已知现象，Mac 训练/推理建议用 CPU（见 [README 的 macOS 说明](../README.md#macos)）。

## 下一步

跑通零样本后，按需求选方向：

- **想要更像的专属音色（微调）**：跟着 [声音克隆端到端教程](voice-cloning-tutorial.md) 走一遍——它以"克隆一个具体角色的声音"为贯穿案例，覆盖素材准备、切分/ASR/打标、训练集格式化、训练 GPT/SoVITS、用微调模型推理的每一步与产物。想先看背后的流水线原理，见 [项目讲解 · 训练流水线](architecture.md#四微调训练流水线)。
- **接入程序（API）**：`python api_v2.py`（v2，配置 `GPT_SoVITS/configs/tts_infer.yaml`）或 `python api.py`（v1），默认端口 9880。
- **命令行合成（CLI）**：`GPT_SoVITS/inference_cli.py`，用参数指定 GPT / SoVITS 权重、参考音频、目标文本。
- **看懂内部原理、准备二次开发**：读 [项目讲解](architecture.md)。
