---
title: AI Agent 作业批改系统
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
license: mit
---

# 🎓 AI Agent 作业批改系统

面向一线教师的智能作业批改助手，基于**阿里云通义千问**大模型。

## ✨ 功能特色

- **多学科分支**：数学严谨校验 / 语文专项批改 / 英语专项批改
- **Agent 化展示**：数学模式 JSON 结构化解析，✅/❌ 直观反馈
- **流式输出**：文科 / 英语模式打字机效果
- **学情分析**：薄弱点统计 + AI 学情综述
- **Prompt 引擎**：可视化编辑 System Prompt 模板
- **PDF 作业提交**：支持上传 PDF 自动提取文字，可与手动输入合并
- **批量批改**：支持文件夹 / 多 PDF / ZIP 导入，批处理后生成个性化邮件与班级报告

## 🔑 使用前配置

本 Space 需要在 **Settings → Repository secrets** 中添加：

| Secret 名称 | 说明 |
|-------------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key |

获取方式：[阿里云百炼控制台](https://bailian.console.aliyun.com/) → API-KEY 管理

## 🛠️ 本地运行

```bash
pip install -r requirements.txt
echo DASHSCOPE_API_KEY=your_key > .env
streamlit run app.py
```

## 📄 技术栈

Python · Streamlit · OpenAI SDK（DashScope 兼容模式）· 通义千问 qwen-turbo / qwen-plus
