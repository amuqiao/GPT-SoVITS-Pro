# 声音克隆端到端教程：从 0 做出一个可复用音色

本文带你从一台空环境开始，准备一批目标声音素材，完成标注、训练和推理，最后得到一组可复用的 GPT / SoVITS 微调权重。

## 这篇文档带你完成什么

完成后你应该得到五类产物：

```text
原始素材        -> 目标声音的音频文件
标注文件        -> 每段音频对应一句文本，格式为 .list
实验中间产物     -> logs/<实验名>/ 下的文本、BERT、HuBERT、语义等文件
微调权重        -> GPT_weights_*/...ckpt + SoVITS_weights_*/...pth
最终合成音频     -> 用微调权重生成的目标声音
```

本文是完整声音克隆流程，不要求你先读其他文档。你会先跑通环境，再用 WebUI 完成素材处理、训练集格式化、两个模型微调和最终推理。

## 先建立心智模型：声音克隆不是一个按钮

声音克隆可以拆成四个连续阶段：

```text
阶段 1：装好环境和底模
  -> 让 WebUI、底模、FFmpeg 能正常工作

阶段 2：把原始声音变成训练集
  -> 切片、ASR 转写、人工校对，得到 .list 标注文件

阶段 3：把训练集变成模型输入
  -> 1A 训练集格式化，生成 logs/<实验名>/ 中间产物

阶段 4：训练并推理
  -> 1B 训练 SoVITS + GPT，1C 选择新权重并合成
```

这四个阶段的产物关系如下：

```text
目标声音长音频
  │
  ├─ 0b 切分
  ▼
短音频片段目录
  │
  ├─ 0c ASR + 0d 人工校对
  ▼
标注文件 .list
  │    每行：音频路径|说话人名|语言|文本
  │
  ├─ 1Aabc 训练集格式化一键三连
  ▼
logs/<实验名>/
  ├─ 2-name2text.txt
  ├─ 3-bert/
  ├─ 4-cnhubert/
  ├─ 5-wav32k/
  ├─ 6-name2semantic.tsv
  └─ 7-sv_cn/               # v2Pro / v2ProPlus 会生成
  │
  ├─ 1Ba 训练 SoVITS
  ├─ 1Bb 训练 GPT
  ▼
SoVITS_weights_<版本>/<实验名>_...pth
GPT_weights_<版本>/<实验名>-...ckpt
  │
  ├─ 1C 推理页选择这两个新权重
  ▼
目标声音说出你的文本
```

还有一个贯穿全流程的事实：

> 即使你已经训练出目标声音模型，最终推理时仍然需要一段 3~10 秒目标声音参考音频和它对应的文本。

微调让模型更熟悉这个音色；参考音频在推理时继续负责锚定当前这次合成的音色、语气和发音状态。

## 第 0 步：定义本次克隆任务

先确定一个实验名。下面用 `myvoice` 作为示例，你可以换成自己的名字，但不要使用空格。

```text
实验名：myvoice
说话人名：myvoice
语言：zh
```

建议把素材放在固定目录，便于 WebUI 填路径：

```bash
mkdir -p raw/myvoice
```

素材要求：

| 项目 | 建议 |
| --- | --- |
| 总时长 | 新手先准备 3~10 分钟干净人声；认真做音色可到 30 分钟左右 |
| 单段质量 | 单人说话、清晰、无明显混响、无背景音乐、无其他人插话 |
| 内容 | 尽量覆盖常见语气和发音，不要全是喘息、笑声、喊叫或极端情绪 |
| 格式 | wav 更稳；mp3 也可先导入再切片 |
| 合规 | 只处理你有权使用的声音，不用于冒充、欺骗、骚扰或未授权商用 |

如果素材带背景音乐或伴奏，后面要先用 UVR5 分离人声；如果本来就是干净讲话，可以跳过 UVR5。

## 第 1 步：创建 Python 环境

进入项目根目录：

```bash
cd /Users/admin/Downloads/GPT-SoVITS-Pro
```

创建并激活环境：

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
```

确认 Python 版本：

```bash
python --version
```

## 第 2 步：安装依赖、底模和可选 UVR5

声音克隆微调通常会用到数据处理工具。建议安装时把 UVR5 一起下载，后续如果素材不带背景音乐，可以不用打开它。

Linux：

```bash
bash install.sh --device <CU126|CU128|ROCM|CPU> --source <HF|HF-Mirror|ModelScope> --download-uvr5
```

macOS：

```bash
bash install.sh --device <MPS|CPU> --source <HF|HF-Mirror|ModelScope> --download-uvr5
```

Windows PowerShell：

```pwsh
pwsh -F install.ps1 --Device <CU126|CU128|CPU> --Source <HF|HF-Mirror|ModelScope> --DownloadUVR5
```

参数选择：

| 参数 | 什么时候选 |
| --- | --- |
| `CU126` / `CU128` | NVIDIA GPU，CUDA 版本匹配对应 PyTorch wheel |
| `ROCM` | AMD GPU 的 ROCm 环境 |
| `MPS` | Apple silicon Mac |
| `CPU` | 没有可用 GPU，能跑但训练会很慢 |
| `HF` | 能稳定访问 Hugging Face |
| `HF-Mirror` | 国内网络优先尝试 |
| `ModelScope` | 国内网络优先尝试 |

安装 FFmpeg：

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

## 第 3 步：确认模型文件存在

训练和推理都依赖底模。安装成功后应能看到：

```text
GPT_SoVITS/pretrained_models/
GPT_SoVITS/text/G2PWModel/
```

本文默认使用 `v2Pro` 训练版本。常用底模路径是：

```text
GPT_SoVITS/pretrained_models/s1v3.ckpt
GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth
GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
```

如果缺少主底模，把 GPT / SoVITS 底模放到 `GPT_SoVITS/pretrained_models/`。如果做中文声音，`G2PWModel` 必须解压并放在 `GPT_SoVITS/text/G2PWModel/`。

## 第 4 步：启动总控 WebUI

声音克隆微调主要在总控 WebUI 里完成：

```bash
conda activate GPTSoVits
python webui.py zh_CN
```

本机浏览器访问：

```text
http://127.0.0.1:9874/
```

如果 WebUI 跑在远程开发服务器，本地需要转发多个端口：

```bash
ssh -N \
  -L 9874:localhost:9874 \
  -L 9872:localhost:9872 \
  -L 9873:localhost:9873 \
  -L 9871:localhost:9871 \
  <user>@<server-ip>
```

浏览器访问本地地址：

```text
http://127.0.0.1:9874/
```

端口用途：

| 端口 | 用途 | 什么时候打开 |
| --- | --- | --- |
| `9874` | 总控 WebUI | 全流程必需 |
| `9872` | TTS 推理 WebUI | 最终推理必需 |
| `9873` | UVR5 人声分离 | 素材带 BGM 时需要 |
| `9871` | 语音文本校对标注 | 做 0d 校对时需要 |

`0.0.0.0` 是服务监听地址，浏览器侧优先使用 `127.0.0.1` 或 `localhost`。

## 第 5 步：进入 0-前置数据集获取工具

打开总控页后，先进入：

```text
0-前置数据集获取工具
```

这里负责把原始音频变成可训练的短音频和标注文件。

## 第 6 步：0a 处理带背景音乐的素材

如果你的素材已经是干净人声，跳过本步。

如果素材来自歌曲、视频、广播剧或带配乐片段，打开：

```text
0a-UVR5人声伴奏分离&去混响去延迟工具
```

点击开启后会打开 UVR5 页面，默认端口是 `9873`。在 UVR5 中选择输入音频或目录，输出目标声音的人声版本。输出后，把干净人声作为后续切分输入。

不要把带背景音乐的素材直接拿去训练。模型会把背景音乐、混响和噪声一起学进去，后面很难靠参数修好。

## 第 7 步：0b 把长音频切成短句

打开：

```text
0b-语音切分工具
```

填写：

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `音频自动切分输入路径，可文件可文件夹` | `raw/myvoice` | 原始干声文件或目录 |
| `切分后的子音频的输出根目录` | `output/slicer_opt` | 切片输出目录 |
| `threshold` | `-34` | 音量低于该值可作为静音切点 |
| `min_length` | `4000` | 每段最短长度，默认即可 |
| `min_interval` | `300` | 最短切割间隔，默认即可 |
| `hop_size` | `10` | 音量曲线计算步长，默认即可 |
| `max_sil_kept` | `500` | 切完后最多保留的静音，默认即可 |
| `max` | `0.9` | 归一化后最大值，默认即可 |
| `alpha_mix` | `0.25` | 归一化混合比例，默认即可 |
| `切割使用的进程数` | `4` | 按 CPU 情况调整 |

点击开启切分。完成后，`output/slicer_opt` 里应出现很多短音频片段。

切片质量检查：

```text
每段最好是一句或半句
不要大量小于 1 秒
不要大量超过 10 秒
不要把两个人的对话切在同一段里
```

## 第 8 步：0c 自动语音识别生成标注

打开：

```text
0c-语音识别工具
```

填写：

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `输入文件夹路径` | `output/slicer_opt` | 第 7 步切出来的短音频目录 |
| `输出文件夹路径` | `output/asr_opt` | ASR 标注输出目录 |
| `ASR 模型` | 达摩中文 ASR 项 | 中文普通话选择页面里的达摩中文 ASR；不同版本可能显示为 `达摩 ASR (中文)` 或 `达摩 ASR (中文经典)` |
| `ASR 模型尺寸` | `large` | 默认即可 |
| `ASR 语言设置` | `zh` | 中文普通话用 `zh`，粤语用 `yue` |
| `数据类型精度` | `float32` | 默认即可 |

点击开启 ASR。完成后，`output/asr_opt` 下会生成一个 `.list` 标注文件。

ASR 输出文件名通常来自输入目录名，例如输入目录是 `output/slicer_opt`，输出目录下会生成类似 `slicer_opt.list` 的文件。

`.list` 每行格式是：

```text
音频路径|说话人名|语言|文本
```

示例：

```text
/abs/path/output/slicer_opt/000001.wav|myvoice|zh|今天我们先完成第一条测试语音。
```

ASR 自动生成时，第二列说话人名通常取输入目录名，语言码可能写成大写，例如：

```text
/abs/path/output/slicer_opt/000001.wav|slicer_opt|ZH|今天我们先完成第一条测试语音。
```

训练脚本能识别常见大小写语言码。`0d` 校对页主要用于改文本，不提供说话人名和语言码输入框；如果你想把第二列统一成 `myvoice`，或把 `ZH` 统一成 `zh`，请在进入 1A 之前用文本编辑器直接批量修改 `.list` 文件。

常用语言码：

| 语言 | `.list` 语言码 |
| --- | --- |
| 中文普通话 | `zh` |
| 粤语 | `yue` |
| 英文 | `en` |
| 日文 | `ja` |
| 韩文 | `ko` |

## 第 9 步：0d 人工校对标注

ASR 一定会有错字、漏字或断句问题。训练前必须校对。

打开：

```text
0d-语音文本校对标注工具
```

填写：

```text
标注文件路径 (含文件后缀 *.list)：第 8 步生成的 .list 文件
```

点击开启后会打开校对页面，默认端口是 `9871`。逐条听音频并修改文本：

```text
音频说了什么，文本就写什么
不要加音频里没有的字
不要省略音频里说出的字
明显错切、多人混说、噪声过大的片段可以删除
```

校对页有两个保存动作，顺序不要省：

```text
1. 改完当前页面的文本后，先点 Submit Text
   - 这个按钮会把当前页文本框内容写回内存和文件
   - 不点就翻页或退出，修改可能回滚

2. 完成一批校对后，再点 Save File
   - 这个按钮把当前 .list 文件保存下来
```

如果需要统一说话人名或语言码，退出 `0d` 后再直接编辑 `.list` 文件的第二列和第三列。保存后的 `.list` 就是后续训练集格式化的输入。

## 第 10 步：进入 1-GPT-SoVITS-TTS 并设置实验信息

回到总控 WebUI，进入：

```text
1-GPT-SoVITS-TTS
```

先在 `微调模型信息` 中填写：

| 字段 | 示例 |
| --- | --- |
| `*实验/模型名` | `myvoice` |
| `训练模型的版本` | `v2Pro` |

`预训练模型路径` 默认会随版本切换。本文使用 `v2Pro`，通常保持默认即可。

## 第 11 步：1A 训练集格式化

进入：

```text
1A-训练集格式化工具
```

填写公共输入：

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `*文本标注文件` | `output/asr_opt/xxx.list` | 第 9 步校对后的 `.list` |
| `*训练集音频文件目录` | `output/slicer_opt` | 切片音频所在目录 |

路径规则很重要：

```text
如果 .list 第一列是绝对音频路径，训练集音频文件目录可以留空。
如果 .list 第一列只是文件名或相对路径，就填写切片音频目录。
WebUI 会用“训练集音频文件目录 + .list 里的音频文件名”去找音频。
```

GPU 卡号字段使用 `-` 分割：

```text
单卡：0
双卡：0-1
```

1A 的默认值在单卡环境下可能显示成 `0-0`，表示开两个进程都用同一张卡。第一次跑不确定时，可以改成 `0`，先用单进程降低排错复杂度。

新手建议直接点击：

```text
1Aabc-训练集格式化一键三连
```

它会按顺序执行：

```text
1Aa 文本分词与特征提取
1Ab 语音自监督特征提取
1Ac 语义Token提取
```

完成后检查 `logs/myvoice/`：

```text
logs/myvoice/2-name2text.txt
logs/myvoice/3-bert/
logs/myvoice/4-cnhubert/
logs/myvoice/5-wav32k/
logs/myvoice/6-name2semantic.tsv
logs/myvoice/7-sv_cn/        # v2Pro / v2ProPlus 会生成
```

缺少这些产物时，不要继续训练。先回到对应步骤看报错，通常是 `.list` 路径、音频路径或底模路径有问题。

## 第 12 步：1B 训练 SoVITS

进入：

```text
1B-微调训练 -> 1Ba-SoVITS 训练: 模型权重文件在 SoVITS_weights/
```

填写或保持默认：

| 字段 | 第一次建议 |
| --- | --- |
| `每张显卡的batch_size` | 先用默认；显存不足再调小 |
| `总训练轮数total_epoch，不建议太高` | 先用默认 |
| `文本模块学习率权重` | 默认 |
| `保存频率save_every_epoch` | 默认 |
| `是否仅保存最新的权重文件以节省硬盘空间` | 开启 |
| `是否在每次保存时间点将最终小模型保存至weights文件夹` | 开启 |
| `GPU卡号以-分割，每个卡号一个进程` | 单卡填 `0` |

点击开启 SoVITS 训练。训练完成后，`v2Pro` 权重会保存到：

```text
SoVITS_weights_v2Pro/
```

文件名通常包含实验名、轮数和步数，例如：

```text
SoVITS_weights_v2Pro/myvoice_e<轮数>_s<步数>.pth
```

## 第 13 步：1B 训练 GPT

仍在 `1B-微调训练` 中，打开：

```text
1Bb-GPT 训练: 模型权重文件在 GPT_weights/
```

填写或保持默认：

| 字段 | 第一次建议 |
| --- | --- |
| `每张显卡的batch_size` | 先用默认；显存不足再调小 |
| `总训练轮数total_epoch` | 默认 `15` |
| `保存频率save_every_epoch` | 默认 `5` |
| `是否开启DPO训练选项(实验性)` | 不开启 |
| `是否仅保存最新的权重文件以节省硬盘空间` | 开启 |
| `是否在每次保存时间点将最终小模型保存至weights文件夹` | 开启 |
| `GPU卡号以-分割，每个卡号一个进程` | 单卡填 `0` |

点击开启 GPT 训练。训练完成后，`v2Pro` 权重会保存到：

```text
GPT_weights_v2Pro/
```

文件名通常包含实验名和轮数，例如：

```text
GPT_weights_v2Pro/myvoice-e<轮数>.ckpt
```

SoVITS 和 GPT 两个训练都要完成。只训练其中一个，最终效果通常不完整。

## 第 14 步：打开 1C 推理页

进入：

```text
1C-推理
```

在总控页里选择刚训练出的权重：

| 字段 | 选择 |
| --- | --- |
| `GPT模型列表` | `GPT_weights_v2Pro/myvoice-e*.ckpt` |
| `SoVITS模型列表` | `SoVITS_weights_v2Pro/myvoice_e*.pth` |
| `GPU卡号,只能填1个整数` | 单卡填 `0` |
| `启用并行推理版本` | 第一次不启用 |

如果列表里没有新权重，点击：

```text
刷新模型路径
```

然后点击开启 TTS 推理 WebUI。推理页默认端口是：

```text
http://127.0.0.1:9872/
```

## 第 15 步：用微调权重合成目标声音

在 `9872` 推理页按顺序填写：

| 顺序 | 中文字段 | 英文字段 | 填什么 |
| --- | --- | --- | --- |
| 1 | `GPT模型列表` | `GPT weight list` | 选择 `myvoice` 的 `.ckpt` |
| 2 | `SoVITS模型列表` | `SoVITS weight list` | 选择 `myvoice` 的 `.pth` |
| 3 | `请上传3~10秒内参考音频` | `Please upload a reference audio within the 3-10 second range` | 上传一段目标声音参考音频 |
| 4 | `参考音频的文本` | `Text for reference audio` | 填参考音频逐字对应文本 |
| 5 | `参考音频的语种` | `Language for reference audio` | 中文选 `中文` / `Chinese` |
| 6 | `需要合成的文本` | `Inference text` | 填你想让目标声音说的话 |
| 7 | `需要合成的语种` | `Inference text language` | 中文选 `中文` / `Chinese` |
| 8 | `怎么切` | `How to slice the sentence` | 第一次保持默认 |
| 9 | `合成语音` | `Start inference` | 点击生成 |

参考音频要求仍然是 3~10 秒。最好从你的训练素材里选一段干净、自然、文本准确的片段。

## 第 16 步：验收这次克隆是否成功

用下面标准检查：

```text
能播放        -> 推理链路成功
内容正确      -> 目标文本、语种和切分基本正确
音色接近      -> 训练权重和参考音频生效
长句稳定      -> GPT 训练和标注质量基本可用
无明显噪声/BGM -> 数据清洗合格
```

第一次微调不要只听一条。建议准备 5 条测试文本：

```text
短句：今天我们先做一个测试。
长句：如果这段声音能够稳定地读完较长的句子，说明模型的韵律和发音都比较可靠。
数字：订单编号是一二三四五六。
中英混合：今天我们测试 GPT-SoVITS 的 voice cloning。
情绪句：这件事情比我想象中更重要。
```

## 常见问题

### 训练集格式化找不到音频

检查 `.list` 第一列和 `*训练集音频文件目录` 的关系：

```text
.list 第一列是绝对路径 -> 训练集音频文件目录可以留空
.list 第一列是文件名   -> 训练集音频文件目录必须填切片目录
```

### `logs/myvoice/6-name2semantic.tsv` 没生成

说明 1Ac 没完成。先确认 1Ab 已生成 `4-cnhubert/` 和 `5-wav32k/`，再检查预训练 SoVITS-G 路径是否存在。

### v2Pro 没有 `7-sv_cn/`

`v2Pro` / `v2ProPlus` 会额外提取说话人向量。检查 `GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt` 是否存在。

### 显存不足

先降低 SoVITS 和 GPT 的 `batch_size`。不要为了省显存跳过训练集格式化产物，也不要随意删 `logs/<实验名>/` 中间产物。

### 音色不够像

优先检查数据，而不是先加训练轮数：

```text
素材是否只有目标说话人
是否有背景音乐、混响、噪声
ASR 文本是否逐条校对
训练素材是否太少
参考音频是否清晰且在 3~10 秒内
```

### 发音漏字、重复或乱读

优先检查 `.list` 文本和目标文本语言：

```text
中文文本使用 zh / 中文
英文文本使用 en / 英文
混合文本再选择对应混合语种
长文本先用默认切分，必要时手动加标点或换行
```

### 音色发闷或像被盖住

常见原因是素材带噪声、带混响、训练轮数过高或参考音频质量差。先换更干净的参考音频，再尝试更早轮次的权重。

### 权重列表没有刷新

确认训练时开启了“是否在每次保存时间点将最终小模型保存至weights文件夹”，然后点击 `刷新模型路径`。`v2Pro` 应查看：

```text
GPT_weights_v2Pro/
SoVITS_weights_v2Pro/
```

## 最终检查清单

在认为任务完成前，逐项确认：

```text
[ ] 能打开 http://127.0.0.1:9874/
[ ] 远程服务器已转发 9874、9872，以及按需转发 9873、9871
[ ] 原始素材已经清洗成单人干声
[ ] 0b 已切出短音频
[ ] 0c 已生成 .list
[ ] 0d 已人工校对 .list
[ ] 1Aabc 已生成 logs/<实验名>/ 下的中间产物
[ ] 1Ba 已生成 SoVITS 权重
[ ] 1Bb 已生成 GPT 权重
[ ] 1C 已选择新权重并打开 9872 推理页
[ ] 最终输出音频内容正确、音色接近、可重复生成
```
