
# 📄 产品需求文档 (PRD)：AI教师作业批改智能体 (高级定制版)

## 一、 产品目标与核心理念
*   **Prompt 有效性与分支管理**：针对不同学科（如数学的逻辑校验、语文的情感与结构），构建完全独立的分支 Prompt 模板，并允许用户二次修改。
*   **结构化输出与前端解析 (Agent 化核心)**：利用大模型输出 JSON，通过代码解析后，重构为高视觉体验的 UI 组件，体现“从接口调用到产品化展示”的跨越。
*   **极致用户体验 (UX)**：隐藏底层复杂的 JSON 和 Prompt 逻辑，向教师展示直观的“正确/错误”图标、分步解析卡片。

## 二、 技术栈与底层约束
*   **框架**：`Python` + `Streamlit` (利用 Tabs 和 Columns 实现高级排版)。
*   **AI 接口**：阿里云百炼平台 (`qwen-turbo`, `qwen-plus`, `qwen-max`)，兼容 OpenAI SDK 调用。
*   **JSON 鲁棒性要求**：代码必须包含健壮的 JSON 解析逻辑（使用 `json.loads`，并使用正则表达式清理大模型可能带有的 ` ```json ` 标记）。

---

## 三、 页面交互与功能架构设计 (UI/UX)

系统采用 **“侧边栏 + 主体多标签页 (Tabs)”** 的布局。

### 3.1 侧边栏 (Sidebar) - 全局与模型控制
*   **应用 Logo/标题**：🎓 AI Agent 作业批改系统
*   **模型选择**：下拉框选择 `qwen-turbo` (极速)、`qwen-plus` (深度)。
*   **学科分支切换 (核心)**：单选框 (Radio Button) -> `数学严谨校验模式` / `文科综合批改模式`。
    *   *交互逻辑：切换此处，将动态改变主界面的输入框和底层的 Prompt 模板。*

### 3.2 主区域 (Main Area) - 双标签页设计
为了体现产品化思维，主区域分为两个 Tab 页：
*   **Tab 1：📝 作业批改台 (用户态)** - 教师日常操作区。
*   **Tab 2：⚙️ 提示词引擎 (开发者/高级设置态)** - 展示和修改当前学科的 System Prompt 模板。

#### ▶️ 场景 A：当侧边栏选择【数学严谨校验模式】时
**Tab 1: 作业批改台**的动态输入框：
1.  输入框 1 (`st.text_area`)：`题目内容`
2.  输入框 2 (`st.text_area`)：`标准答案或参考思路`
3.  输入框 3 (`st.text_area`, 高度更大)：`学生的实际解答步骤`
4.  按钮：`🔍 开始执行逻辑校验`

**结果渲染逻辑 (UX 重点)**：
*   拦截大模型的 JSON 输出，不直接显示给用户。
*   如果 `is_correct == true`，使用 `st.success("✅ 该题解答完全正确！")`。
*   如果 `is_correct == false`，使用 `st.error("❌ 发现逻辑错误或计算失误")`，并用两个卡片 (`st.info` / `st.warning`) 分别展示 `error_step` 和 `error_reason`。
*   最后使用一个代码块或 Markdown 框展示 `corrected_solution`。

#### ▶️ 场景 B：当侧边栏选择【文科综合批改模式】时
**Tab 1: 作业批改台**的动态输入框：
1.  输入框 1 (`st.text_input`)：`本次批改侧重点（如：过去时态、议论文结构）`
2.  输入框 2 (`st.text_area`)：`学生作业正文`
3.  按钮：`🚀 开始智能批改`

**结果渲染逻辑**：
*   文科采用流式文本打字机输出 (`st.write_stream`)。

---

## 四、 核心 Prompt 架构设计 (Branching Prompts)

系统需要在 `session_state` 中维护不同学科的默认 Prompt。在 **Tab 2 (提示词引擎)** 中，用户可以使用 `st.text_area` 自由修改这些模板。

### 4.1 数学模式专用 Prompt (强制 JSON)
**变量绑定**：`{question}` (题目), `{standard_answer}` (标准答案), `{student_answer}` (学生解答)

**模板内容**：
```text
# 角色：
你是一位严谨、专业的数学老师，拥有多年的中学数学教学与阅卷经验。

# 上下文：
你需要根据提供的题目和标准参考，对学生的解答进行严格的逻辑校验。
题目：{question}
参考证明思路或关键步骤：{standard_answer}

# 任务：
请严格审阅以下学生的解答。你需要抛弃主观猜测，像编译器一样逐行检查其推导公式、逻辑链条的严密性和每一步骤的正确性。

# 学生解答：
{student_answer}

# 输出要求：
请严格按照以下JSON格式输出，绝对不要包含任何额外的解释性文本、Markdown标记（如```json）或问候语，确保输出可直接被Python的json.loads()解析：
{{
  "is_correct": true/false,
  "error_step": "如完全正确则填'无'，如有误则明确指明第几步或摘录错误原句",
  "error_reason": "详细分析错误原因，例如'此处逻辑跳跃'、'公式记忆错误'、'计算失误'等，如正确填'无'",
  "corrected_solution": "提供正确的步骤或完整解法"
}}
```
*(注：JSON的大括号在 Python string format 中需要双写 `{{` 和 `}}` 以防报错)*

### 4.2 文科模式专用 Prompt
**变量绑定**：`{rubric}` (侧重点), `{student_answer}` (学生作业)

**模板内容**：
```text
你现在是一位专业的语言文学教师。你需要帮我批改学生的作业。
【本次批改侧重点】：{rubric}

【输出格式】：
1. 🏷️ 薄弱知识点标签：(提取1-3个标签)
2. 📊 综合评分：(百分制或评级)
3. ❌ 错误指正详情
4. 👧 给学生的鼓励评语

【学生作业内容】：
{student_answer}
```

---

## 🛠️ 五、 给 Cursor/代码生成助手的具体执行指令

**@Cursor, 请严格阅读上述 PRD，并使用 Python 和 Streamlit 生成 `app.py`。**

**关键技术要求与实现细节：**
1.  **API 接入**：使用 `openai` 库，配置 `base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"`，并从系统环境变量或 `.env` 读取 `DASHSCOPE_API_KEY`。
2.  **动态 UI 渲染**：利用 Streamlit 的 `st.radio` 切换学科，基于 `if` 判断动态渲染相应的输入框组合。
3.  **状态管理 (Session State)**：
    *   将两套 Prompt 模板保存在 `st.session_state` 中，确保在 Tab 2 中修改 Prompt 后，Tab 1 调用的也是修改后的最新 Prompt。
    *   必须处理 Python `str.format()` 遇到 JSON 大括号时的冲突（建议将模板里的 JSON `{` 写为 `{{`，或者使用 `string.Template` 替换机制）。
4.  **JSON 解析与容错机制 (极其重要)**：
    *   当处于数学模式时，获取到大模型的响应后，**不要直接打印**。
    *   请写一个 `parse_json_response(text)` 函数。因为大模型可能会不听话地输出 ` ```json\n{...}\n``` `，你需要用正则提取出纯粹的 `{...}` 部分，再使用 `json.loads`。
    *   解析成功后，使用 Streamlit 的组件（如 `st.success`, `st.error`, `st.expander`）将 `is_correct`, `error_step`, `error_reason`, `corrected_solution` 美观地展示出来。如果 JSON 解析失败，则直接 `st.error` 显示原始文本。
5.  **文科流式输出**：当处于文科模式时，保持使用流式请求 (`stream=True`) 和 `st.write_stream`。

请提供包含完整注释的、健壮可运行的 `app.py` 代码。