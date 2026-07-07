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
| 系统盘（`/`）仅剩几十 G，数据盘在 `/data`（TB 级） | conda env、pip/HF 缓存默认落在 `~`（系统盘）；pip 装 torch 时还会往 `/tmp` 解包好几 G 的 CUDA 库，都可能撑爆系统盘 | conda env、pip 缓存、HF 缓存**统一重定向到 `/data`**；跑 `install.sh` 时前面带 `TMPDIR=/data/wangqiao/tmp` 把 pip 解包也顶到数据盘 |
| conda 默认连境外源，新版还要求接受服务条款 | `conda-forge` 安装慢，`conda create` 直接被 `CondaToSNonInteractiveError` 拦下 | 把 conda 源换成国内镜像（见第 2 步），既提速又免 ToS |
| `requirements.txt` 走 pip→PyPI，跟 conda 镜像无关 | 国内拉 PyPI 包体可能慢（体检里的 `200 OK` 只是握手通，不代表下载快） | 单独给 pip 配国内源（清华 PyPI），见第 3 步「requirements 慢」 |
| 驱动 CUDA 12.4，`install.sh` 只提供 CU126/CU128 | 没有正好匹配 12.4 的选项 | 选 **CU126**：CUDA 12.x 小版本向前兼容，驱动 ≥525 即可跑 12.6 runtime（本机驱动 550，**已实测 `torch 2.12.1+cu126` 返回 `True`**） |
| `nvcc` 缺失（无 CUDA toolkit） | 会让人以为装不了 | 不影响：PyTorch wheel 自带 CUDA runtime，训练/推理不需要系统 `nvcc` |
| 2 张卡，其中一张常被别人占用 | 不指定卡会撞到别人的显存 | 运行时用 `CUDA_VISIBLE_DEVICES` 显式选空闲卡 |
| 多人共用、`conda` 不一定装好 | `install.sh` 硬依赖 conda，缺了直接报错退出 | 安装前先确认 conda；缺失则装 Miniconda 到 `/data` |

一句话心智：**镜像源走 HF-Mirror，一切缓存落 /data，设备选 CU126，运行显式选卡。**

## 第 1 步：前置检查

先跑仓库自带的体检脚本，确认基础依赖齐、并看清网络与磁盘现状：

```bash
cd /data/wangqiao/GPT-SoVITS-Pro
bash scripts/check_server_env.sh
```

重点看四处：**登录 Shell 类型**（决定环境变量写进哪个 rc 文件，见下）、HuggingFace 连通性（预期超时）、`/data` 剩余空间（安装约需 15~20G 富余）、`nvidia-smi` 里两张卡各自的占用。

脚本的「Conda 环境/包管理器 conda (仅信息)」一节会报告 conda 现状（仅信息，不影响 exit code）：

- 有 `conda --version` 输出 → 已装好，跳到第 2 步。
- 显示 `MISSING: 未安装 conda` → 按下面装 Miniconda 到数据盘（避免占系统盘）。

> **先看脚本报出的「登录 Shell」再选命令**：`bash` 用 `~/.bashrc` + `conda init bash`；`zsh` 用 `~/.zshrc` + `conda init zsh`。下面以 bash 为例，zsh 用户把两处 `bash` 换成 `zsh`。

```bash
cd /data/wangqiao
curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p /data/wangqiao/miniconda3
rm /data/wangqiao/miniconda.sh
/data/wangqiao/miniconda3/bin/conda init bash   # zsh 用户改为 conda init zsh
exec bash                                           # zsh 用户改为 exec zsh；重载使 conda 生效
```

装完可再跑一次 `bash scripts/check_server_env.sh`，确认 conda 一节已显示版本与路径。

## 第 2 步：配置 conda（国内源 + 缓存落数据盘）

建环境之前先做两件事，否则会踩两个坑：conda 默认连境外源，又慢又会被服务条款拦下；依赖和缓存默认写进只剩几十 G 的系统盘。

**先把 conda 源换成国内镜像（清华 TUNA）。** 这一步同时解决「conda-forge 装依赖慢」和「`conda create` 报 `CondaToSNonInteractiveError`」两个问题——默认源不再指向 `repo.anaconda.com`，conda 就不要求接受它的条款：

```bash
cat > ~/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
conda clean -i -y
```

**再把 conda 环境、包缓存、pip / HuggingFace 缓存全部落到 `/data`：**

```bash
# conda 环境与包缓存放数据盘（追加进上面的 ~/.condarc）
conda config --add envs_dirs /data/wangqiao/conda/envs
conda config --add pkgs_dirs /data/wangqiao/conda/pkgs

# pip 与 HuggingFace 缓存放数据盘（写进 shell 配置，长期生效）
# 注意：zsh 用户把下面三处 ~/.bashrc 换成 ~/.zshrc（以第 1 步脚本报出的登录 Shell 为准）
echo 'export PIP_CACHE_DIR=/data/wangqiao/.cache/pip'  >> ~/.bashrc
echo 'export HF_HOME=/data/wangqiao/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc
```

> 底模不受影响：`install.sh` 会把权重下到仓库内的 `GPT_SoVITS/pretrained_models/`，仓库本身在 `/data`，天然落在数据盘。

## 第 3 步：建环境并安装

`install.sh` 依赖 conda，内部用 conda 装系统级依赖（gcc、ffmpeg、cmake），再用 pip 装 PyTorch 和 `requirements.txt`，最后从镜像拉底模与词典。它**自己不建环境、也不激活环境**，装进的是你运行它时**当前激活的那个环境**——所以务必先 `conda activate GPTSoVits`，确认提示符是 `(GPTSoVits)` 再跑，否则会灌进 `base`。

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits          # 确认提示符变成 (GPTSoVits) 再往下

cd /data/wangqiao/GPT-SoVITS-Pro
TMPDIR=/data/wangqiao/tmp bash install.sh --device CU126 --source HF-Mirror
```

命令按本机约束固定为：

- `TMPDIR=/data/wangqiao/tmp`：pip 会把 torch 的一堆 CUDA 库（cudnn、cublas… 解包后好几 G）默认写到系统盘 `/tmp`，撑爆只剩几十 G 的 `/`。用 `TMPDIR` 顶到 `/data`。**`install.sh` 自身不设 `TMPDIR`，必须你在命令前带上。**
- `--device CU126`：匹配驱动 CUDA 12.4（理由见上表）。若安装或首次 `torch.cuda` 调用报驱动版本不兼容，再降级到与驱动更贴合的 PyTorch 轮子（见排查）。
- `--source HF-Mirror`：绕开被墙的 HuggingFace。镜像偶发不稳时可改 `--source ModelScope`。
- **不加** `--download-uvr5`：人声分离模型只在「从歌曲里剥人声做微调素材」时才需要，纯推理/常规微调不用，省几个 G。需要时再补跑一次带该参数的安装。

装完出现 `Installation Completed` 即成功。成功的输出大致长这样（底模已下过时会看到 `Exists` 跳过）：

```text
[SUCCESS]: libstdcxx-ng=11 Installed...
[SUCCESS]: FFmpeg & CMake Installed
[SUCCESS]: unzip Installed
[INFO]: Pretrained Model Exists          # 已下过，跳过
[INFO]: G2PWModel Exists                  # 已下过，跳过
[SUCCESS]: PyTorch Installed
[SUCCESS]: Python Dependencies Installed
[SUCCESS]: NLTK Data Downloaded
[SUCCESS]: Open JTalk Dic Downloaded
[SUCCESS]: Installation Completed         # ← 看到这行才算全装完
```

### 看不到进度？install.sh 吞了输出

`install.sh` 用 `run_pip_quiet` **把 pip 输出整个捕获，只在失败时才打印**。所以装 torch（cu126 的 wheel 是 GB 级）时，屏幕会长时间停在 `Installing PyTorch For CUDA 12.6...` 一动不动——**这通常是在下载，不是卡死**。另开一个窗口确认它在动：

```bash
# ① 进程在不在
pgrep -af "install.sh"; pgrep -af "pip install"

# ② 临时下载文件在不在长大（数字持续涨 = 在下）
watch -n2 'du -sh /data/wangqiao/tmp/* 2>/dev/null'
```

判断：`pip install torch` 进程在、且 `/data/wangqiao/tmp` 下临时文件持续变大 → 正常下载中，等它。进程在但无临时文件增长、网络也静 → 多半在把已缓存依赖解包安装（纯磁盘活，正常）。进程完全没有 → 回 `install.sh` 窗口看是走完了还是报错。

> `/proc/net/dev` 只能看「有没有在下、多快」，看不了某个文件「下了多少 / 还剩多少」。要看具体进度用上面的 `du` 盯临时文件，或改用下面手动 pip（有进度条）。

### torch 下载太慢的兜底

官方源 `download.pytorch.org` 从国内可能只有几百 kB/s。嫌慢就 `Ctrl-C`，在 `GPTSoVits` 环境用**上海交大镜像**手动装 torch（**已下载的依赖在 pip 缓存里不会重下**，只重下主 wheel），装完再重跑 `install.sh` 续上：

```bash
TMPDIR=/data/wangqiao/tmp pip install torch torchcodec \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126
# 装好后重跑，install.sh 检测到 torch 已在会跳过，继续装 requirements 与词典
TMPDIR=/data/wangqiao/tmp bash install.sh --device CU126 --source HF-Mirror
```

> 只有 SJTU（`mirror.sjtu.edu.cn/pytorch-wheels`）和官方 `download.pytorch.org` 确定有 cu126 wheel；**阿里云、清华没有 pytorch-wheels 镜像，别用**（会报 `No matching distribution found`）。两个源都可能慢，别反复切换——每次 `Ctrl-C` 都会让主 wheel 从头下。

### requirements.txt 慢？换 pip 源 + 单独装（能看到进度）

torch 装完后 `install.sh` 会装 `requirements.txt`，这步走的是 **pip → PyPI**，**不吃 conda 的 `~/.condarc` 镜像**（那只管 conda 装的 gcc/ffmpeg 等）。国内拉 PyPI 可能慢，两个办法都能解决，其中办法 B 还能让你看到进度条：

**A. 换 pip 源后重跑 install.sh（最省事）**

```bash
# 持久化清华 PyPI（这台机器以后所有 pip 都走它）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 重跑，torch/模型已装会跳过，只快速补 requirements 与词典
TMPDIR=/data/wangqiao/tmp bash install.sh --device CU126 --source HF-Mirror
```

**B. 自己手动装 pip 依赖（有进度条）**，复刻 `install.sh` 的两步，再回 `install.sh` 收尾下词典：

```bash
conda activate GPTSoVits
cd /data/wangqiao/GPT-SoVITS-Pro

PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple TMPDIR=/data/wangqiao/tmp \
  pip install -r extra-req.txt --no-deps
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple TMPDIR=/data/wangqiao/tmp \
  pip install -r requirements.txt

# 依赖装完回 install.sh 补最后的 NLTK / JTalk 词典
TMPDIR=/data/wangqiao/tmp bash install.sh --device CU126 --source HF-Mirror
```

> 直接跑 pip 天然带 `━━━` 进度条；`install.sh` 的 `run_pip_quiet` 会吞掉输出，想看进度就手动跑。已装好的包 pip 判定「已满足」会跳过，不会从零重来。

> **先分清「慢」的原因**：换源只治**下载慢**。若 `/data/wangqiao/tmp` 里冒出大量 `pip-build-env-*` / `pip-modern-metadata-*` 目录，说明有包在**从源码编译 wheel**（CPU 活，本项目有几个），换源无效，只能等它编完——这属正常，不是卡死。

## 第 4 步：验证

分两层验证，先证 GPU 可用，再证服务能起。

```bash
# 1) PyTorch 能看到卡（验证 CU126 与驱动 12.4 兼容）
conda activate GPTSoVits
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 实测输出：2.12.1+cu126 True 2
#   True → CU126 runtime 在 12.4 驱动上正常；2 → 两张卡都识别到
```

```bash
# 2) 起推理服务：PYTHONPATH=. 让它找到根目录的 config.py；选一张空闲卡
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python GPT_SoVITS/inference_webui.py
```

> **必须带 `PYTHONPATH=.`**：`inference_webui.py` 直接 `from config import`，而 `config.py` 在仓库根目录，脚本自己不把根目录加进 `sys.path`。不带就报 `ModuleNotFoundError: No module named 'config'`。且脚本用 `os.getcwd()` 拼模型路径，所以**必须在仓库根目录跑**。

看到 `Running on local URL: http://0.0.0.0:9872`、且不再刷报错即成功（`To create a public link, set share=True` 是正常提示，不是错）。加载底模无报错，即整条链路可用。

> **本机实测跑通的版本组合**（供对照）：`torch 2.12.1+cu126` / `torchaudio 2.11.0+cu126` / `fastapi 0.115.2` / `starlette 0.40.0` / `gradio 4.44.1` / `pydantic 2.10.6`。其中 torchaudio、fastapi/starlette 都是 `install.sh` 装完后需按「排查」手动对齐的——根因是 `requirements.txt` 多处不锁版本，在 2026 的新依赖环境下装出了未测组合。若你也踩到，照「排查」两条修即可。跑通第一条零样本合成的页面操作见 [快速上手 · 第 5 步](quickstart.md)。

> 共享机器铁律：**任何训练/推理命令都带 `CUDA_VISIBLE_DEVICES` 显式选卡**。先 `nvidia-smi` 看哪张卡显存空，再把序号填进去，避免撞掉别人的任务。

### 选卡：不是「必须空卡」，是「留够显存」

`CUDA_VISIBLE_DEVICES=0` 只是让程序「只看得见 0 号卡」。选哪张的判断标准是**剩余显存够不够**（推理约需几个 G），不是「必须没人用」。用 `nvidia-smi` 对照：

```text
GPU 0  1548MiB / 23028MiB   → 空余 ~21.5G，够，选它（那点占用是别人的小进程，不影响）
GPU 1 19446MiB / 23028MiB   → 只剩 ~3.5G，跑推理易 OOM，别选
```

## 第 5 步：从本机浏览器访问 WebUI（SSH 端口转发）

服务器是无界面机器，WebUI 起在服务器的 9872 端口，但云服务器这个端口通常没对外开放，本机直连服务器 IP 多半不通。可靠做法是用 **SSH 本地端口转发**——走你已有的 SSH 通道把 9872 隧道到本机，不需要开放任何防火墙端口。

两个窗口：

```bash
# ① 服务器上（当前 SSH 会话）起服务，等它打印 Running on local URL: http://0.0.0.0:9872
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python GPT_SoVITS/inference_webui.py
```

```bash
# ② 本机另开一个终端建隧道（-N = 只转发不执行命令，保持开着别关）
ssh -N -L 9872:localhost:9872 47.94.108.140
```

然后**本机浏览器打开 http://localhost:9872**。

> `-L 9872:localhost:9872` = 把本机 9872 转发到「服务器上的 localhost:9872」，所以 WebUI 在服务器上监听 127.0.0.1 或 0.0.0.0 都能通。隧道不通时先在服务器确认它在监听：`ss -tlnp | grep 9872`。
> 想同时开总控(9874)/UVR5(9873) 等别的页，就多加 `-L`：`ssh -N -L 9872:localhost:9872 -L 9874:localhost:9874 <地址>`。

## 排查

按现象对号入座（通用现象见 [快速上手 · 常见问题排查](quickstart.md)，这里只列服务器特有的）：

- **`conda create` 报 `CondaToSNonInteractiveError`（服务条款未接受）** → 按第 2 步配好 `~/.condarc` 国内源即可绕过（默认源不再指向 `repo.anaconda.com`）；临时办法是 `conda create -n GPTSoVits python=3.10 -y -c conda-forge --override-channels`。
- **下载卡住 / 超时** → 先分清卡在哪一步：conda 装系统依赖卡住 → conda 源没换（第 2 步）；模型下载卡住 → 确认用了 `--source HF-Mirror` 而非 `HF`，镜像抖动时切 `ModelScope`；torch 下载「看不见动静」→ 见第 3 步「看不到进度」与「torch 太慢的兜底」；`requirements.txt` 慢 → 它走 pip→PyPI（不是 conda 源），换 pip 源，见第 3 步「requirements 慢」。
- **`/data` 或 `/` 空间不足** → 检查缓存重定向是否生效：`conda config --show envs_dirs`、`echo $HF_HOME`；清理系统盘上旧的 `~/.cache`。
- **`torch.cuda.is_available()` 为 False，或报驱动/CUDA 版本不兼容** → 多半是 CU126 runtime 与驱动错配。先 `nvidia-smi` 确认驱动仍是 12.4；如确需贴合，卸载 torch 后按 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 装匹配 CUDA 12.4 的轮子（`pip install torch --index-url https://download.pytorch.org/whl/cu124`）再重试。
- **起 `inference_webui.py` 报 `OSError: libcudart.so.13`（`import torchaudio` 时）** → `requirements.txt` 里的 `torchaudio` **未锁版本也未指定源**，被从 PyPI 装成了按 CUDA 13 编译的版本，与 cu126 的 torch 不匹配（`ldd .../_torchaudio.abi3.so | grep cudart` 会看到 `libcudart.so.13 => not found`）。从 cu126 源装一个 torchaudio，链接 `libcudart.so.12`。注意 **cu126 源上 torchaudio 版本可能落后于 torch**（本机 torch 是 `2.12.1`，但 cu126 最高只有 `torchaudio 2.11.0`），先查可用版本再装最高的、用 `--no-deps` 不动 torch：
  ```bash
  pip index versions torchaudio --index-url https://download.pytorch.org/whl/cu126   # 看最高版本
  TMPDIR=/data/wangqiao/tmp pip install --force-reinstall --no-deps \
    torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
  ```
  重装后 `import torchaudio` 不再报错即可（本机 torch 2.12.1 + torchaudio 2.11.0+cu126 实测可正常 import；若报 `undefined symbol` 说明 ABI 不兼容，需把 torch 降到与 torchaudio 同版本一起装）。
- **起 `inference_webui.py` 时页面刷 `TypeError: unhashable type: 'dict'`，随后 `localhost is not accessible` 退出** → `gradio 4.x` 与过新的 `starlette/fastapi` 签名不兼容（`fastapi[standard]>=0.115.2` 无上限，会装到 fastapi 0.139 / starlette 1.x）。把 fastapi 钉回 `0.115.2`，pip 会连带把 starlette 降到 `<0.42`：
  ```bash
  TMPDIR=/data/wangqiao/tmp pip install "fastapi[standard]==0.115.2"
  # 验证：fastapi 0.115.2 / starlette 0.40.x
  pip show fastapi starlette | grep -iE "^Name|^Version"
  ```
- **`conda: command not found`** → 回第 1 步装 Miniconda 并 `conda init`。
- **显存不足（OOM）** → 先 `nvidia-smi` 看是否选到了被占用的卡；换空闲卡，或调小 batch / 用半精度。

## 维护

- 本文只覆盖「这类国内 GPU 服务器」的部署。跨平台通用安装仍以 [快速上手](quickstart.md) 为准；两者出现冲突时，以本文的服务器约束为准。
- 体检脚本 `scripts/check_server_env.sh` 现已报告 conda（仅信息）与登录 Shell 类型；若其检查项再变化（如磁盘阈值、conda 改为必要依赖），同步更新本文第 1 步。
- 若默认设备版本、镜像源选项或 `install.sh` 参数变化，同步更新第 3 步。
