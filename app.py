"""
AI 教师作业批改智能体 (高级定制版)

依赖安装：
    pip install streamlit openai python-dotenv

环境变量：
    在项目根目录创建 .env 文件，写入：
    DASHSCOPE_API_KEY=你的阿里云百炼 API Key

运行方式：
    streamlit run app.py
"""

import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Generator

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

load_dotenv()

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_OPTIONS = {
    "qwen-turbo (极速)": "qwen-turbo",
    "qwen-plus (深度)": "qwen-plus",
}

SUBJECT_MATH = "数学严谨校验模式"
SUBJECT_ARTS = "文科综合批改模式"
SUBJECT_ENGLISH = "英语专项批改模式"
ALL_SUBJECTS = [SUBJECT_MATH, SUBJECT_ARTS, SUBJECT_ENGLISH]

# 数学模式默认 Prompt
DEFAULT_MATH_PROMPT = """# 角色：
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
  "corrected_solution": "提供正确的步骤或完整解法",
  "knowledge_tags": ["#知识点1", "#知识点2"],
  "score": 85
}}"""

DEFAULT_LIBERAL_ARTS_PROMPT = """你现在是一位专业的语言文学教师。你需要帮我批改学生的作业。
【本次批改侧重点】：{rubric}

【输出格式】：
1. 🏷️ 薄弱知识点标签：(提取1-3个标签，格式 #标签名)
2. 📊 综合评分：(百分制或评级)
3. ❌ 错误指正详情
4. 👧 给学生的鼓励评语

【学生作业内容】：
{student_answer}"""

DEFAULT_ENGLISH_PROMPT = """You are an experienced English teacher grading student homework.

【Grading Focus / 本次批改侧重点】：{rubric}

【Output Format / 输出格式】（请用中文输出，保留英文原文引用）：
1. 🏷️ Weak Point Tags / 薄弱知识点标签：(1-3 tags, format #TagName, e.g. #PastTense #SubjectVerbAgreement)
2. 📊 Overall Score / 综合评分：(percentage or letter grade)
3. ❌ Error Details / 错误指正详情：(grammar, vocabulary, tense, structure — cite original sentences)
4. ✏️ Improved Version / 修改示范：(rewrite key problematic sentences)
5. 👧 Encouragement / 给学生的鼓励评语

【Student Homework / 学生作业】：
{student_answer}"""

DEFAULT_ANALYTICS_PROMPT = """你是一位资深教学数据分析专家。请根据以下学生的历次批改记录，生成一份简洁的学情分析报告。

【学生姓名】：{student_name}
【批改记录汇总】：
{records_summary}

请输出以下模块（使用 Emoji 标题）：
1. 📌 学情总览（一句话概括当前水平）
2. 🎯 高频薄弱点（按出现频率排序，Top 3-5）
3. 📈 进步与风险（哪些方面有改善趋势，哪些需 urgent 关注）
4. 💡 教学建议（给教师 3 条可操作的辅导建议）
5. 📝 下次练习方向（推荐针对性练习类型）"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def get_openai_client() -> OpenAI:
    """创建 OpenAI 兼容客户端，从环境变量读取 API Key。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY。请在 .env 文件或系统环境变量中配置该密钥。"
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def parse_json_response(text: str) -> dict:
    """从大模型响应中提取并解析 JSON。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    json_str = match.group(0) if match else cleaned
    return json.loads(json_str)


def extract_tags_from_text(text: str) -> list[str]:
    """从文本中提取 #标签 格式的知识点。"""
    tags = re.findall(r"#[\w\u4e00-\u9fff\-]+", text)
    return list(dict.fromkeys(tags))


def extract_score_from_text(text: str) -> int | None:
    """尝试从批改文本中提取百分制分数。"""
    patterns = [
        r"综合评分[：:]\s*(\d{1,3})\s*分",
        r"(\d{1,3})\s*/\s*100",
        r"Score[：:]\s*(\d{1,3})",
        r"(\d{1,3})%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score
    return None


def build_math_prompt(question: str, standard_answer: str, student_answer: str) -> str:
    template = st.session_state.math_prompt
    return template.format(
        question=question,
        standard_answer=standard_answer,
        student_answer=student_answer,
    )


def build_arts_prompt(rubric: str, student_answer: str) -> str:
    template = st.session_state.liberal_arts_prompt
    rubric_text = rubric.strip() if rubric.strip() else "按常规标准进行全面批改"
    return template.format(rubric=rubric_text, student_answer=student_answer)


def build_english_prompt(rubric: str, student_answer: str) -> str:
    template = st.session_state.english_prompt
    rubric_text = rubric.strip() if rubric.strip() else "全面检查语法、词汇、时态与篇章结构"
    return template.format(rubric=rubric_text, student_answer=student_answer)


def call_llm(model: str, prompt: str, user_msg: str, stream: bool = False):
    """统一 LLM 调用入口。"""
    client = get_openai_client()
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        stream=stream,
    )


def call_math_grading(model: str, prompt: str) -> str:
    response = call_llm(model, prompt, "请开始逻辑校验，仅输出 JSON。", stream=False)
    return response.choices[0].message.content or ""


def stream_text_grading(model: str, prompt: str) -> Generator[str, None, None]:
    stream = call_llm(model, prompt, "请开始批改这份作业。", stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def add_grading_record(
    subject: str,
    student_name: str,
    tags: list[str],
    score: int | None,
    is_correct: bool | None,
    summary: str,
) -> None:
    """将一次批改结果写入 session_state 学情历史。"""
    st.session_state.grading_history.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "subject": subject,
            "student_name": student_name or "未命名学生",
            "tags": tags,
            "score": score,
            "is_correct": is_correct,
            "summary": summary[:200],
        }
    )


def render_math_result(raw_text: str, student_name: str) -> None:
    """解析数学 JSON 并展示，同时写入学情记录。"""
    try:
        result = parse_json_response(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"JSON 解析失败：{e}")
        st.markdown("**原始响应：**")
        st.code(raw_text, language="text")
        return

    is_correct = result.get("is_correct", False)
    error_step = result.get("error_step", "无")
    error_reason = result.get("error_reason", "无")
    corrected_solution = result.get("corrected_solution", "")
    tags = result.get("knowledge_tags", [])
    if isinstance(tags, str):
        tags = extract_tags_from_text(tags)
    score = result.get("score")
    if isinstance(score, str) and score.isdigit():
        score = int(score)

    if is_correct:
        st.success("✅ 该题解答完全正确！")
    else:
        st.error("❌ 发现逻辑错误或计算失误")
        st.info(f"**出错步骤：**\n\n{error_step}")
        st.warning(f"**错误原因：**\n\n{error_reason}")

    if corrected_solution and corrected_solution != "无":
        st.markdown("**✅ 正确解法参考**")
        st.code(corrected_solution, language="text")

    if tags:
        st.markdown("**🏷️ 涉及知识点**")
        st.markdown(" ".join(tags))

    summary = error_reason if not is_correct else "解答完全正确"
    add_grading_record(
        subject=SUBJECT_MATH,
        student_name=student_name,
        tags=tags,
        score=score,
        is_correct=is_correct,
        summary=summary,
    )
    st.caption("✅ 已记录至学情分析")


def save_stream_grading_record(
    subject: str, student_name: str, full_text: str
) -> None:
    """流式批改完成后，解析标签/分数并写入学情。"""
    tags = extract_tags_from_text(full_text)
    score = extract_score_from_text(full_text)
    add_grading_record(
        subject=subject,
        student_name=student_name,
        tags=tags,
        score=score,
        is_correct=None,
        summary=full_text[:200],
    )
    st.caption("✅ 已记录至学情分析")


def build_records_summary(history: list[dict], student_filter: str) -> str:
    """将历史记录格式化为供 AI 分析的文本。"""
    filtered = [
        r
        for r in history
        if student_filter == "全部学生" or r["student_name"] == student_filter
    ]
    if not filtered:
        return "（暂无记录）"

    lines = []
    for i, r in enumerate(filtered, 1):
        tag_str = "、".join(r["tags"]) if r["tags"] else "无"
        score_str = f"{r['score']}分" if r["score"] is not None else "未提取"
        correct_str = (
            "正确" if r["is_correct"] is True
            else "错误" if r["is_correct"] is False
            else "-"
        )
        lines.append(
            f"{i}. [{r['time']}] {r['subject']} | 分数:{score_str} | "
            f"正误:{correct_str} | 标签:{tag_str} | 摘要:{r['summary']}"
        )
    return "\n".join(lines)


def render_analytics_tab(selected_model: str) -> None:
    """学情分析 Tab：统计图表 + AI 综述。"""
    st.subheader("📊 学情分析仪表盘")
    st.caption("自动汇总历次批改数据，洞察学生薄弱点与进步趋势")

    history = st.session_state.grading_history

    if not history:
        st.info("暂无批改记录。请先在「作业批改台」完成至少一次批改。")
        return

    students = sorted({r["student_name"] for r in history})
    student_filter = st.selectbox(
        "👤 筛选学生",
        options=["全部学生"] + students,
        key="analytics_student_filter",
    )

    filtered = [
        r
        for r in history
        if student_filter == "全部学生" or r["student_name"] == student_filter
    ]

    # ---- 指标卡片 ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("批改次数", len(filtered))
    with col2:
        scores = [r["score"] for r in filtered if r["score"] is not None]
        avg = round(sum(scores) / len(scores), 1) if scores else "-"
        st.metric("平均分", avg)
    with col3:
        math_records = [r for r in filtered if r["subject"] == SUBJECT_MATH]
        if math_records:
            correct_rate = round(
                sum(1 for r in math_records if r["is_correct"]) / len(math_records) * 100, 1
            )
            st.metric("数学正确率", f"{correct_rate}%")
        else:
            st.metric("数学正确率", "-")
    with col4:
        all_tags = [t for r in filtered for t in r["tags"]]
        st.metric("薄弱标签数", len(set(all_tags)))

    st.divider()

    # ---- 薄弱知识点频次 ----
    tag_counter = Counter(t for r in filtered for t in r["tags"])
    if tag_counter:
        st.markdown("### 🏷️ 高频薄弱知识点")
        chart_data = dict(tag_counter.most_common(10))
        st.bar_chart(chart_data)

    # ---- 学科分布 ----
    subject_counter = Counter(r["subject"] for r in filtered)
    st.markdown("### 📚 批改学科分布")
    st.bar_chart(dict(subject_counter))

    # ---- 历史明细 ----
    st.markdown("### 📋 批改历史明细")
    st.dataframe(
        [
            {
                "时间": r["time"],
                "学生": r["student_name"],
                "学科": r["subject"],
                "分数": r["score"] if r["score"] is not None else "-",
                "正误": (
                    "✅" if r["is_correct"] is True
                    else "❌" if r["is_correct"] is False
                    else "-"
                ),
                "标签": " ".join(r["tags"]) if r["tags"] else "-",
            }
            for r in reversed(filtered)
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ---- AI 学情综述 ----
    st.markdown("### 🧠 AI 学情综述")
    if st.button("生成 AI 学情分析报告", type="primary", key="btn_analytics"):
        records_summary = build_records_summary(history, student_filter)
        prompt = DEFAULT_ANALYTICS_PROMPT.format(
            student_name=student_filter,
            records_summary=records_summary,
        )
        st.markdown("#### 分析报告")
        try:
            stream = call_llm(
                selected_model, prompt, "请生成学情分析报告。", stream=True
            )

            def _stream() -> Generator[str, None, None]:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

            st.write_stream(_stream)
        except Exception as e:
            st.error(f"学情分析生成失败：{e}")

    if st.button("🗑️ 清空学情记录", key="clear_history"):
        st.session_state.grading_history = []
        st.rerun()


def init_session_state() -> None:
    """初始化 session_state。"""
    defaults = {
        "math_prompt": DEFAULT_MATH_PROMPT,
        "liberal_arts_prompt": DEFAULT_LIBERAL_ARTS_PROMPT,
        "english_prompt": DEFAULT_ENGLISH_PROMPT,
        "grading_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_prompt_engine(subject_mode: str) -> None:
    """Tab 3：提示词引擎，按学科分支展示可编辑模板。"""
    st.subheader("⚙️ 提示词引擎")
    st.caption("修改当前学科分支的 System Prompt，批改时将自动使用最新版本")

    if subject_mode == SUBJECT_MATH:
        st.markdown("**当前分支：数学严谨校验模式**")
        st.markdown("变量：`{question}` · `{standard_answer}` · `{student_answer}`")
        edited = st.text_area(
            "数学模式 Prompt",
            value=st.session_state.math_prompt,
            height=450,
            key="math_prompt_editor",
        )
        st.session_state.math_prompt = edited
        if st.button("🔄 恢复默认数学 Prompt", key="reset_math"):
            st.session_state.math_prompt = DEFAULT_MATH_PROMPT
            st.rerun()

    elif subject_mode == SUBJECT_ARTS:
        st.markdown("**当前分支：文科综合批改模式**")
        st.markdown("变量：`{rubric}` · `{student_answer}`")
        edited = st.text_area(
            "文科模式 Prompt",
            value=st.session_state.liberal_arts_prompt,
            height=450,
            key="arts_prompt_editor",
        )
        st.session_state.liberal_arts_prompt = edited
        if st.button("🔄 恢复默认文科 Prompt", key="reset_arts"):
            st.session_state.liberal_arts_prompt = DEFAULT_LIBERAL_ARTS_PROMPT
            st.rerun()

    else:
        st.markdown("**当前分支：英语专项批改模式**")
        st.markdown("变量：`{rubric}` · `{student_answer}`")
        edited = st.text_area(
            "英语模式 Prompt",
            value=st.session_state.english_prompt,
            height=450,
            key="english_prompt_editor",
        )
        st.session_state.english_prompt = edited
        if st.button("🔄 恢复默认英语 Prompt", key="reset_english"):
            st.session_state.english_prompt = DEFAULT_ENGLISH_PROMPT
            st.rerun()


def render_stream_grading_ui(
    subject: str,
    title: str,
    caption: str,
    rubric_label: str,
    rubric_placeholder: str,
    homework_label: str,
    homework_placeholder: str,
    button_label: str,
    model: str,
    prompt_builder,
    rubric_key: str,
    homework_key: str,
    button_key: str,
) -> None:
    """文科 / 英语共用的流式批改 UI。"""
    st.subheader(title)
    st.caption(caption)

    student_name = st.text_input(
        "👤 学生姓名（选填，用于学情分析）",
        placeholder="如：张三",
        key=f"student_{button_key}",
    )

    rubric = st.text_input(
        rubric_label,
        placeholder=rubric_placeholder,
        key=rubric_key,
    )

    homework = st.text_area(
        homework_label,
        height=300,
        placeholder=homework_placeholder,
        key=homework_key,
    )

    if st.button(button_label, type="primary", use_container_width=True, key=button_key):
        if not homework.strip():
            st.warning("请输入学生作业内容")
        else:
            st.markdown("### 📋 批改结果")
            try:
                prompt = prompt_builder(rubric, homework.strip())

                def _stream() -> Generator[str, None, None]:
                    yield from stream_text_grading(model, prompt)

                full_text = st.write_stream(_stream)
                if full_text:
                    save_stream_grading_record(subject, student_name, full_text)
            except ValueError as e:
                st.error(f"配置错误：{e}")
            except Exception as e:
                st.error(f"API 调用失败：{e}")


# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Agent 作业批改系统",
    page_icon="🎓",
    layout="wide",
)

init_session_state()

# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎓 AI Agent 作业批改系统")
    st.caption("多学科分支 · 学情分析 · 智能批改")
    st.divider()

    model_label = st.selectbox(
        "🤖 模型选择",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
    )
    selected_model = MODEL_OPTIONS[model_label]

    subject_mode = st.radio(
        "📚 学科分支切换",
        options=ALL_SUBJECTS,
        index=0,
        help="切换后将动态改变输入框与 Prompt 模板",
    )

    st.divider()
    history_count = len(st.session_state.grading_history)
    st.markdown(
        f"**快速指引**\n\n"
        f"- 当前模式：**{subject_mode}**\n"
        f"- 学情记录：**{history_count}** 条\n"
        f"- Tab 1：作业批改\n"
        f"- Tab 2：学情分析\n"
        f"- Tab 3：Prompt 引擎"
    )

# ---------------------------------------------------------------------------
# 主区域 - 三标签页
# ---------------------------------------------------------------------------

tab_grading, tab_analytics, tab_prompt = st.tabs(
    ["📝 作业批改台", "📊 学情分析", "⚙️ 提示词引擎"]
)

with tab_grading:
    if subject_mode == SUBJECT_MATH:
        st.subheader("🔢 数学严谨校验")
        st.caption("逐行校验推导逻辑，输出结构化校验报告")

        student_name = st.text_input(
            "👤 学生姓名（选填，用于学情分析）",
            placeholder="如：张三",
            key="math_student_name",
        )

        col1, col2 = st.columns(2)
        with col1:
            question = st.text_area(
                "📋 题目内容",
                height=120,
                placeholder="请输入数学题目…",
                key="math_question",
            )
        with col2:
            standard_answer = st.text_area(
                "✅ 标准答案或参考思路",
                height=120,
                placeholder="请输入标准答案或关键解题步骤…",
                key="math_standard",
            )

        student_answer = st.text_area(
            "📝 学生的实际解答步骤",
            height=250,
            placeholder="请粘贴学生的完整解答过程…",
            key="math_student",
        )

        if st.button(
            "🔍 开始执行逻辑校验",
            type="primary",
            use_container_width=True,
            key="btn_math",
        ):
            if not question.strip() or not student_answer.strip():
                st.warning("请填写题目内容和学生解答步骤")
            else:
                st.markdown("### 📋 校验结果")
                try:
                    prompt = build_math_prompt(
                        question.strip(),
                        standard_answer.strip(),
                        student_answer.strip(),
                    )
                    raw_response = call_math_grading(selected_model, prompt)
                    render_math_result(raw_response, student_name)
                except ValueError as e:
                    st.error(f"配置错误：{e}")
                except Exception as e:
                    st.error(f"API 调用失败：{e}")

    elif subject_mode == SUBJECT_ARTS:
        render_stream_grading_ui(
            subject=SUBJECT_ARTS,
            title="📖 文科综合批改",
            caption="支持自定义侧重点，流式输出批改报告",
            rubric_label="🎯 本次批改侧重点（如：议论文结构、修辞手法）",
            rubric_placeholder="留空则按常规标准全面批改",
            homework_label="📝 学生作业正文",
            homework_placeholder="请粘贴学生的作文或作业内容…",
            button_label="🚀 开始智能批改",
            model=selected_model,
            prompt_builder=build_arts_prompt,
            rubric_key="arts_rubric",
            homework_key="arts_homework",
            button_key="btn_arts",
        )

    else:
        render_stream_grading_ui(
            subject=SUBJECT_ENGLISH,
            title="🇬🇧 英语专项批改",
            caption="聚焦语法、时态、词汇与篇章结构，流式输出双语批改报告",
            rubric_label="🎯 本次批改侧重点（如：一般过去时、定语从句、书信格式）",
            rubric_placeholder="留空则全面检查语法、词汇、时态与结构",
            homework_label="📝 学生英语作业",
            homework_placeholder="Paste student's English essay or exercises here…",
            button_label="🚀 开始英语批改",
            model=selected_model,
            prompt_builder=build_english_prompt,
            rubric_key="english_rubric",
            homework_key="english_homework",
            button_key="btn_english",
        )

with tab_analytics:
    render_analytics_tab(selected_model)

with tab_prompt:
    render_prompt_engine(subject_mode)
