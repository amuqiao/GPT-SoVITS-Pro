# 开发服务器环境搭建

在国内 GPU 开发服务器（Linux + N 卡，HuggingFace 不通、系统盘小、数据盘大、多人共用）上把 GPT-SoVITS-Pro 从零装到能跑推理/训练。本文是这类机器的 **runbook**：只写清「因为机器有这些约束，所以命令要这样写」，不重复通用原理。

## 文档职责

- 负责：这类服务器上的前置检查、磁盘与缓存布局、`install.sh` 的正确参数、选卡运行、验证与排查。
- 不负责：模型原理与二次开发（见 [项目讲解](architecture.md)）；跨平台通用安装与零样本跑通（见 [快速上手](quickstart.md)）；微调全流程（见 [声音克隆端到端教程](voice-cloning-tutorial.md)）。
- 适用读者：在共享 GPU 服务器上部署本项目的工程师，具备 Linux、conda、CUDA 基本常识。
- 校准基线：本文命令基于一台 `Ubuntu 22.04 / 2×A10(23G) / 驱动 CUDA 12.4 / Python 3.10 / 系统盘紧张、数据盘在 /data` 的机器写就。同类机器直接照抄；参数不同处按下文「先理解这台机器」逐条替换。

## 先理解这台机器（这些约束决定了命令怎么写）

通用安装文档默认「网络能连 HuggingFace、装哪都行、独占一张卡」。开发服务器几乎每条都不成立，逐条对上再动手：

| 机器约束 | 为什么影响安装 | 本文的应对 |
| --- | --- | --- |
| HuggingFace 443 超时 | 底模、NLTK、JTalk 词典默认从 HF 下载，会直接卡死 | `install.sh --source HF-Mirror`（或 `ModelScope`），**不要**用 `HF`；单独下载也走镜像 |
| 系统盘（`/`）仅剩几十 G，数据盘在 `/data`（TB 级） | conda env、pip/HF 缓存默认落在 `~`（系统盘），几个大依赖就撑爆 | 把 conda env、pip 缓存、HF 缓存**统一重定向到 `/data`** |
| 驱动 CUDA 12.4，`install.sh` 只提供 CU126/CU128 | 没有正好匹配 12.4 的选项 | 选 **CU126**：CUDA 12.x 小版本向前兼容，驱动 ≥525 即可跑 12.6 runtime（本机驱动 550，满足） |
| `nvcc` 缺失（无 CUDA toolkit） | 会让人以为装不了 | 不影响：PyTorch wheel 自带 CUDA runtime，训练/推理不需要系统 `nvcc` |
| 2 张卡，其中一张常被别人占用 | 不指定卡会撞到别人的显存 | 运行时用 `CUDA_VISIBLE_DEVICES` 显式选空闲卡 |
| 多人共用、`conda` 不一定装好 | `install.sh` 硬依赖 conda，缺了直接报错退出 | 安装前先确认 conda；缺失则装 Miniconda 到 `/data` |

一句话心智：**镜像源走 HF-Mirror，一切缓存落 /data，设备选 CU126，运行显式选卡。**

## 第 1 步：前置检查

先跑仓库自带的体检脚本，确认基础依赖齐、并看清网络与磁盘现状：

```bash
cd /data/<你的用户名>/GPT-SoVITS-Pro
bash scripts/check_server_env.sh
```

重点看三行：HuggingFace 连通性（预期超时）、`/data` 剩余空间（安装约需 15~20G 富余）、`nvidia-smi` 里两张卡各自的占用。

体检脚本**不检查 conda**，单独确认一次：

```bash
command -v conda && conda --version
```

有输出即可跳到第 2 步。若为空，安装 Miniconda 到数据盘（避免占系统盘）：

```bash
cd /data/<你的用户名>
curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p /data/<你的用户名>/miniconda3
/data/<你的用户名>/miniconda3/bin/conda init bash
exec bash   # 重载 shell，使 conda 生效
```

## 第 2 步：把缓存重定向到数据盘

在建环境之前先改缓存落点，否则依赖和模型缓存会写进系统盘。让 conda 环境、pip 缓存、HuggingFace 缓存全部指向 `/data`：

```bash
# conda 环境与包缓存放数据盘
conda config --add envs_dirs /data/<你的用户名>/conda/envs
conda config --add pkgs_dirs /data/<你的用户名>/conda/pkgs

# pip 与 HuggingFace 缓存放数据盘（写进 shell 配置，长期生效）
echo 'export PIP_CACHE_DIR=/data/<你的用户名>/.cache/pip'  >> ~/.bashrc
echo 'export HF_HOME=/data/<你的用户名>/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc
```

> 底模不受影响：`install.sh` 会把权重下到仓库内的 `GPT_SoVITS/pretrained_models/`，仓库本身在 `/data`，天然落在数据盘。

## 第 3 步：建环境并安装

`install.sh` 依赖 conda，内部用 conda 装系统级依赖（gcc、ffmpeg、cmake），再用 pip 装 PyTorch 和 `requirements.txt`，最后从镜像拉底模与词典。

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits

cd /data/<你的用户名>/GPT-SoVITS-Pro
bash install.sh --device CU126 --source HF-Mirror
```

参数按本机约束固定为：

- `--device CU126`：匹配驱动 CUDA 12.4（理由见上表）。若安装或首次 `torch.cuda` 调用报驱动版本不兼容，再降级到与驱动更贴合的 PyTorch 轮子（见排查）。
- `--source HF-Mirror`：绕开被墙的 HuggingFace。镜像偶发不稳时可改 `--source ModelScope`。
- **不加** `--download-uvr5`：人声分离模型只在「从歌曲里剥人声做微调素材」时才需要，纯推理/常规微调不用，省几个 G。需要时再补跑一次带该参数的安装。

装完出现 `Installation Completed` 即成功。

## 第 4 步：验证

分两层验证，先证 GPU 可用，再证服务能起。

```bash
# 1) PyTorch 能看到卡（验证 CU126 与驱动 12.4 兼容）
conda activate GPTSoVits
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 预期：2.x True 2
```

```bash
# 2) 起推理服务，选一张空闲卡（用 nvidia-smi 确认哪张空）
CUDA_VISIBLE_DEVICES=0 python GPT_SoVITS/inference_webui.py
```

服务正常监听、加载底模无报错，即整条链路可用。跑通第一条零样本合成的页面操作见 [快速上手 · 第 5 步](quickstart.md)。

> 共享机器铁律：**任何训练/推理命令都带 `CUDA_VISIBLE_DEVICES` 显式选卡**。先 `nvidia-smi` 看哪张卡显存空，再把序号填进去，避免撞掉别人的任务。

## 排查

按现象对号入座（通用现象见 [快速上手 · 常见问题排查](quickstart.md)，这里只列服务器特有的）：

- **下载卡住 / 超时** → 确认用了 `--source HF-Mirror` 而非 `HF`；镜像本身抖动时切 `ModelScope` 重跑 `install.sh`。
- **`/data` 或 `/` 空间不足** → 检查缓存重定向是否生效：`conda config --show envs_dirs`、`echo $HF_HOME`；清理系统盘上旧的 `~/.cache`。
- **`torch.cuda.is_available()` 为 False，或报驱动/CUDA 版本不兼容** → 多半是 CU126 runtime 与驱动错配。先 `nvidia-smi` 确认驱动仍是 12.4；如确需贴合，卸载 torch 后按 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 装匹配 CUDA 12.4 的轮子（`pip install torch --index-url https://download.pytorch.org/whl/cu124`）再重试。
- **`conda: command not found`** → 回第 1 步装 Miniconda 并 `conda init`。
- **显存不足（OOM）** → 先 `nvidia-smi` 看是否选到了被占用的卡；换空闲卡，或调小 batch / 用半精度。

## 维护

- 本文只覆盖「这类国内 GPU 服务器」的部署。跨平台通用安装仍以 [快速上手](quickstart.md) 为准；两者出现冲突时，以本文的服务器约束为准。
- 体检脚本 `scripts/check_server_env.sh` 若新增检查项（如 conda、磁盘阈值），同步更新本文第 1 步。
- 若默认设备版本、镜像源选项或 `install.sh` 参数变化，同步更新第 3 步。
