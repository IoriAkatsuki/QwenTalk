# QwenTalk

QwenTalk 是一个专为 Intel® NPU 优化的桌面聊天应用，基于 OpenVINO™ 工具套件开发，旨在充分发挥新一代 AI PC 的硬件性能。本项目特别为 **Lunar Lake**、**Arrow Lake** 及后续系列处理器设计，推荐在搭载 **Intel® Core™ Ultra 2** 或更高性能处理器的设备上运行以获得最佳体验。

应用优先利用 NPU 进行推理，同时支持 GPU 和 CPU，并能在 NPU 不可用时自动回退。项目包含图形用户界面 (GUI) 和命令行界面 (CLI) 两种模式。

## ✨ 功能特性

- **专为 Intel NPU 优化**: 基于 OpenVINO™ 开发，为下一代 AI PC 提供硬件加速支持。
- **双模式操作**: 提供现代化的图形用户界面 (GUI) 和轻量级的命令行界面 (CLI)。
- **硬件自动回退**: 优先使用 Intel® NPU/GPU 加速，并支持在不可用时自动切换至 CPU。
- **会话管理**: 支持创建、切换和删除多个独立的聊天会话。
- **流式响应**: 对话响应以流的形式逐字生成，提升交互体验。
- **联网搜索 (CLI)**: 在 CLI 模式下，可通过 `/search` 命令调用搜索引擎获取实时信息。

## ⚙️ 环境准备

在开始之前，请确保您的系统已安装以下软件：

- **Conda**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/products/distribution)。
- **Git**: 用于克隆本仓库。
- **Python**: 推荐使用 Python 3.10。

## 🚀 安装与配置

**1. 克隆仓库**

```bash
git clone https://github.com/IoriAkatsuki/QwenTalk.git
cd QwenTalk
```

**2. 创建并激活 Conda 环境**

```bash
conda create -n openvino python=3.10 -y
conda activate openvino
```

**3. 安装依赖**

```bash
pip install -r requirements.txt
```

**4. 获取模型**

本项目需要使用 OpenVINO IR 格式的模型。您可以使用 `optimum-cli` 工具从 Hugging Face Hub 导出所需的模型。

执行以下命令来导出 `Qwen/Qwen3-8B` 模型：

```bash
optimum-cli export openvino --model Qwen/Qwen3-8B --task text-generation-with-past --weight-format nf4 --sym --group-size -1 Qwen3-8B-nf4-ov --backup-precision int8_sym
```

该命令会创建一个名为 `Qwen3-8B-nf4-ov` 的文件夹，其中包含了 OpenVINO 推理所需的 `.xml` 和 `.bin` 文件。请确保此文件夹位于项目的根目录下。

## ▶️ 如何运行

**图形用户界面 (GUI)**

- **Windows 用户**:
  最简单的方式是直接运行 `run.bat` 批处理文件。它会自动检查您的 Conda 环境和模型文件，然后启动应用程序。

- **其他用户 (或手动启动)**:
  确保您已激活 `openvino` Conda 环境，然后执行：
  ```bash
  python QwenTalkGUI.py
  ```

**命令行界面 (CLI)**

确保您已激活 `openvino` Conda 环境，然后执行以下命令：

```bash
python talkCLI.py -m ./Qwen3-8B-nf4-ov
```

CLI 模式支持以下命令:
- `/help`: 显示帮助信息。
- `/clear`: 清除当前对话历史。
- `/search <关键词>`: 进行联网搜索。
- `/quit` 或 `/exit`: 退出程序。

## 📂 文件结构

```
QwenTalk/
├── Qwen3-8B-nf4-ov/    # 存放 OpenVINO 模型文件
├── QwenTalkGUI.py      # GUI 应用主程序
├── talkCLI.py          # CLI 应用主程序
├── QwenTalk.py         # 核心聊天引擎 (模型加载与推理)
├── run.bat             # Windows 快速启动脚本
├── requirements.txt    # Python 依赖列表
└── README.md           # 本文档
```

## 📝 未来计划 (TODO List)

- [ ] 接入 Microsoft Copilot (MCP) 实现更深度的信息流整合。
- [ ] 优化模型量化方案，进一步降低资源占用。
- [ ] 增加对更多本地模型的支持。
- [ ] GUI 界面增加更多可配置选项。
