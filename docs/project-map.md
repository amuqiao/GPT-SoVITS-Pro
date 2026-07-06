# Project Map

本文用于快速判断 GPT-SoVITS-Pro 的入口、目录职责和优先阅读路径，减少每次任务都从完整目录扫描开始。

## 文档边界

本文负责回答“改某类功能应该先看哪里”。它不替代 README、安装文档、API 文档或源码细节，也不记录临时排查结论。

当新增、移动或删除核心入口、训练/推理流程、主要配置、验证命令或顶层目录职责时，同步更新本文。

## 整体模型

仓库主干可以按四层理解：

```text
根目录入口脚本
  -> WebUI / API / CLI / 安装与 Docker
GPT_SoVITS/
  -> TTS 推理、GPT 语义模型、SoVITS 声学模型、文本处理、训练和数据准备
tools/
  -> WebUI 调用的音频处理、ASR、UVR5、人声分离、i18n 等辅助工具
docs/
  -> 多语言 README、changelog 和少量维护入口文档
```

优先从根目录入口和 `GPT_SoVITS/` 内的流程文件进入，不要从模型组件或第三方子模块开始全量阅读。

## 常用入口

- WebUI 主入口：`webui.py`；Windows 快捷入口是 `go-webui.bat` / `go-webui.ps1`。
- 推理 WebUI：`GPT_SoVITS/inference_webui.py`，快速版本为 `GPT_SoVITS/inference_webui_fast.py`，GUI 辅助入口为 `GPT_SoVITS/inference_gui.py`。
- API v1：`api.py`，默认端口来自 `config.py`，接口说明写在文件顶部注释中。
- API v2：`api_v2.py`，默认配置为 `GPT_SoVITS/configs/tts_infer.yaml`，核心管线来自 `GPT_SoVITS/TTS_infer_pack/TTS.py`。
- CLI 推理：`GPT_SoVITS/inference_cli.py`。
- V2Pro 流式 CLI：`GPT_SoVITS/stream_v2pro.py`。
- GPT 语义模型训练：`GPT_SoVITS/s1_train.py`，主要读取 `GPT_SoVITS/configs/s1*.yaml`。
- SoVITS 声学模型训练：`GPT_SoVITS/s2_train.py`、`GPT_SoVITS/s2_train_v3.py`、`GPT_SoVITS/s2_train_v3_lora.py`，主要读取 `GPT_SoVITS/configs/s2*.json`。
- 数据准备流水线：`GPT_SoVITS/prepare_datasets/1-get-text.py`、`2-get-hubert-wav32k.py`、`2-get-sv.py`、`3-get-semantic.py`。
- ASR 工具：中文/粤语看 `tools/asr/funasr_asr.py`，多语种看 `tools/asr/fasterwhisper_asr.py`。
- UVR5 工具：`tools/uvr5/webui.py`。
- 安装脚本：`install.sh`、`install.ps1`。
- Docker：`Dockerfile`、`docker-compose.yaml`、`docker_build.sh`、`Docker/`。

## 顶层目录职责

- `GPT_SoVITS/`：核心 TTS、训练、推理和模型实现。
- `GPT_SoVITS/TTS_infer_pack/`：API v2 和推理管线使用的 TTS 封装、文本预处理和文本切分。
- `GPT_SoVITS/AR/`：GPT/Text2Semantic 相关数据、模型、模块和训练工具。
- `GPT_SoVITS/module/`：SoVITS/VITS 模型、判别器、损失、mel 处理和数据加载等底层模块。
- `GPT_SoVITS/BigVGAN/`：vocoder 相关实现、配置和测试。
- `GPT_SoVITS/eres2net/`：speaker embedding 相关模型。
- `GPT_SoVITS/f5_tts/`：F5-TTS 相关模块。
- `GPT_SoVITS/text/`：多语言文本清洗、分词、音素、符号表和中文归一化。
- `GPT_SoVITS/feature_extractor/`：HuBERT、Whisper 等特征提取封装。
- `GPT_SoVITS/prepare_datasets/`：训练数据标注、HuBERT 特征、说话人向量和 semantic 提取脚本。
- `GPT_SoVITS/configs/`：训练和推理配置。
- `tools/`：音频切片、降噪、超分、ASR、UVR5、人声分离、i18n 和 WebUI 静态资源辅助。
- `docs/`：多语言用户文档、变更日志和少量维护入口文档。
- `Docker/`：容器内安装辅助脚本。

## 常见修改应该先看哪里

- 改 WebUI 工作流：先看 `webui.py`，再按功能进入 `GPT_SoVITS/inference_webui.py`、`tools/uvr5/webui.py`、`tools/subfix_webui.py` 或数据准备脚本。
- 改 API 行为：v1 先看 `api.py`；v2 先看 `api_v2.py` 和 `GPT_SoVITS/TTS_infer_pack/TTS.py`。
- 改推理质量、切分、流式或批处理：先看 `GPT_SoVITS/TTS_infer_pack/`，再看 `GPT_SoVITS/inference_webui.py`。
- 改 GPT/Text2Semantic：先看 `GPT_SoVITS/AR/models/`、`GPT_SoVITS/AR/modules/`、`GPT_SoVITS/s1_train.py`。
- 改 SoVITS/VITS：先看 `GPT_SoVITS/module/models.py`、`GPT_SoVITS/module/data_utils.py`、`GPT_SoVITS/s2_train.py`。
- 改文本清洗或多语言发音：先看 `GPT_SoVITS/text/`，再看 `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`。
- 改数据集准备：先看 `GPT_SoVITS/prepare_datasets/` 和 WebUI 中启动这些脚本的调用点。
- 改安装、依赖或容器：先看 `requirements.txt`、`extra-req.txt`、`install.sh`、`install.ps1`、`Dockerfile`、`docker-compose.yaml`。
- 改工具链能力：ASR 看 `tools/asr/`，UVR5/人声分离看 `tools/uvr5/`，音频切片看 `tools/slice_audio.py` 和 `tools/slicer2.py`，音频超分看 `tools/audio_sr.py` 和 `tools/AP_BWE_main/`。

## 不优先扫描的目录和产物

以下路径通常是模型权重、下载内容、缓存、日志或生成产物。除非任务明确相关，不要把它们作为第一轮扫描目标：

- `GPT_SoVITS/pretrained_models/`
- `tools/asr/models/`
- `tools/uvr5/uvr5_weights/`
- `tools/denoise-model/`
- `tools/AP_BWE_main/24kto48k/`
- `output/`
- `logs/`
- `TEMP/`
- `SoVITS_weights*/`
- `GPT_weights*/`
- `runtime/`
- `env/`
- `venv/`
- `ref_audios/`
- `ffmpeg*`
- `ffprobe*`
- `weight.json`
- `cfg.json`
- `speakers.json`
- `*.pth`、`*.ckpt`、`*.onnx`、`*.pt`

`GPT_SoVITS/text/` 下的 `cmudict*.rep`、`engdict*.rep`、`*_cache.pickle`、`g2pw/*.rep`、`g2pw/*.pickle` 属于文本前端词典或缓存数据；只有排查文本处理逻辑时才优先查看。

## 验证入口

本仓库没有单一通用的轻量测试入口。按修改范围选择最小验证：

- 文档或规则改动：检查 Markdown 链接和 `git diff` 范围。
- API 改动：至少运行对应文件的语法编译；有模型权重和运行环境时，再启动 `api.py` 或 `api_v2.py` 做一次 smoke 请求。
- WebUI 改动：有完整依赖和模型时启动 `webui.py`，验证对应页面流程。
- 训练或数据准备改动：优先用小样本配置跑目标脚本，避免直接启动完整训练。
- 安装或 Docker 改动：优先验证脚本参数解析或 Docker Compose 配置；完整构建只在任务需要时执行。
