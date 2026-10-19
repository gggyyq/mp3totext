# 本地音频转文字

一个面向 Windows 的简洁 Python 桌面工具：选择 MP3 等音频文件，指定 TXT 保存位置，然后在本机完成语音转写。

## 功能

- 支持 MP3、WAV、M4A、AAC、FLAC、OGG、OPUS、WMA、WEBM。
- 自动检测中文、英文及其他常见语言。
- 只转写原语言，不进行翻译。
- 使用 `faster-whisper` 在本机识别，音频和转写文本不会上传。
- 首次下载模型后可断网使用。
- 自动选择 NVIDIA GPU；不可用时回退到 CPU。
- 长音频在后台处理，窗口保持响应。
- 支持取消；失败或取消不会留下不完整 TXT，也不会破坏原有输出文件。

## 环境要求

- Windows 10/11
- 64 位 Python 3.11 或更高版本（已在 Python 3.13 验证）
- 项目已包含本地 CTranslate2 `large-v3` 模型时无需联网；缺少模型时需联网下载约 3 GB

Tkinter 通常随官方 Python 一起安装。可以先运行以下命令确认：

```bat
py -m tkinter
```

如果弹出一个测试窗口，说明 Tkinter 可用。

## 安装

在项目目录打开终端：

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 启动

```bat
python app.py
```

使用步骤：

1. 单击“选择音频”。
2. 软件会默认把 TXT 放在音频旁边，也可以单击“更改路径”。
3. 单击“开始转写”。
4. 程序会优先使用项目目录中的 `whisper-large-v3-ct2`，不会重复下载；仅在该目录缺失或不完整时提示下载。

模型优先从项目中的 `whisper-large-v3-ct2` 加载；日志默认保存在：

```text
%LOCALAPPDATA%\LocalSpeechTranscriber\
```

## 准确率与速度

默认使用多语言 `large-v3` 模型，优先保证中文、口音和嘈杂录音中的转写准确率。模型和显存占用较大；建议使用至少 6 GB 显存的 NVIDIA GPU。

- NVIDIA GPU 通常明显快于 CPU。
- 仅使用 CPU 时，长音频可能需要较长时间。
- 很短的语音、较强背景噪声、多人重叠说话或频繁混合语言会降低识别准确率。
- 程序启用了静音检测，以减少长时间静音导致的错误文字。

## 隐私说明

- 音频识别过程完全在本机进行。
- 音频内容和转写文本不会发送给服务商。
- 首次使用只会从 Hugging Face 下载语音识别模型文件。
- 模型下载完成后，识别不再需要网络。
- 本地日志不会记录音频内容或转写文本。

## 常见问题

### 模型下载失败

检查网络、代理设置和磁盘空间，然后再次单击“开始转写”。模型目录中的未完成下载由依赖库管理，重新尝试通常会继续下载。

### 无法解码音频

文件可能损坏，或实际编码与扩展名不一致。可先用常见播放器确认文件能够正常播放，再转换为 MP3 或 WAV 后重试。

### CUDA 不可用

程序会自动回退到 CPU，不要求安装 PyTorch。如果 NVIDIA 驱动或 CUDA 运行库不兼容，转写仍可继续，但速度会变慢。

### 无法保存 TXT

确认目标目录存在、具有写入权限，并关闭可能正在占用同名 TXT 的编辑器。

### 查看日志

日志位于：

```text
%LOCALAPPDATA%\LocalSpeechTranscriber\logs\app.log
```

## 运行测试

安装开发依赖：

```bat
python -m pip install -r requirements-dev.txt
python -m pytest
```

单元测试使用假模型，不会下载真实模型。
