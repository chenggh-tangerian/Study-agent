"""
AI 教师作业批改智能体 (高级定制版)

依赖安装：
    pip install streamlit openai python-dotenv pypdf

环境变量：
    在项目根目录创建 .env 文件，写入：
    DASHSCOPE_API_KEY=你的阿里云百炼 API Key

运行方式：
    streamlit run app.py
"""

import io
import json
import os
import re
import zipfile
from collections import Counter
from datetime import datetime
from typing import Generator

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

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
SUBJECT_ARTS = "语文专项批改模式"
SUBJECT_ENGLISH = "英语专项批改模式"
ALL_SUBJECTS = [SUBJECT_MATH, SUBJECT_ARTS, SUBJECT_ENGLISH]

# 各学科标签前缀（学情分析、图表按此前缀区分）
SUBJECT_TAG_PREFIX = {
    SUBJECT_MATH: "数学",
    SUBJECT_ARTS: "语文",
    SUBJECT_ENGLISH: "英语",
}

# 兼容旧 session 中的学科名称
SUBJECT_ALIASES: dict[str, list[str]] = {
    SUBJECT_MATH: [SUBJECT_MATH],
    SUBJECT_ARTS: [SUBJECT_ARTS, "文科综合批改模式"],
    SUBJECT_ENGLISH: [SUBJECT_ENGLISH],
}

# Prompt 版本号：递增后会自动刷新 session 中缓存的旧模板
PROMPT_VERSION = 5
# 批量批改模块版本（部署到 HF 后可在界面底部核对是否最新）
BATCH_MODULE_VERSION = "batch-v3-compat"

# 数学模式 JSON 示例中曾使用的占位分数，模型易照搬
MATH_TEMPLATE_ANCHOR_SCORE = 85
# 文科/英语常见“偷懒”锚定分数
HOMEWORK_ANCHOR_SCORES = {85, 88, 90, 92, 95}

ARTS_DIMENSION_CAPS = {
    "情感理解": 40,
    "个人感悟": 30,
    "语言表达": 20,
    "结构逻辑": 10,
}

ENGLISH_DIMENSION_CAPS = {
    "Grammar": 30,
    "Vocabulary": 25,
    "Structure": 25,
    "Content": 20,
}

# 数学模式默认 Prompt
DEFAULT_MATH_PROMPT = """# 角色：
你是一位严谨、专业的数学老师，拥有多年的中学数学教学与阅卷经验。

# 上下文：
你需要根据提供的题目和标准参考，对学生的解答进行严格的逻辑校验。
题目：{question}
参考证明思路或关键步骤：{standard_answer}

# 任务：
请严格审阅以下学生的解答。你需要抛弃主观猜测，像编译器一样逐行检查其推导公式、逻辑链条的严密性和每一步骤的正确性。

# 任务与严格约束（V2.0 底线核查）：
【第一步：底线核查（熔断规则）】
在常规打分前，必须先进行以下三项核查，若触犯须记录到 fatal_errors 并执行降级扣分：
1. 定理/公式张冠李戴：使用错误公式、混淆定理条件（如把勾股定理用于非直角三角形），总分扣除15-25分。
2. 严重答非所问：解答内容与题目要求完全无关，is_correct 必须为 false，score 不得超过40分。
3. 套步骤无推导：只有结论或套模板步骤，缺少关键推导过程，须在 fatal_errors 中标注，score 不得超过55分。

【第二步：精准评价】
- 禁止「基本正确」「还可以」等模糊评价。
- 优点：必须指出具体哪一步推导正确、哪个等式成立。
- 缺点：必须摘录错误原句/原步骤，说明错因，给出修改建议。

# 学生解答：
{student_answer}

# 输出要求：
请严格按照以下JSON格式输出，绝对不要包含任何额外的解释性文本、Markdown标记（如```json）或问候语，确保输出可直接被Python的json.loads()解析。
字符串值中禁止出现未转义的单个反斜杠；数学符号请用 plain text（如 x∈[1,2]）或将反斜杠写为双反斜杠（\\\\in）。
{{
  "is_correct": true/false,
  "fatal_errors": ["触发的熔断规则说明，无则为空数组[]"],
  "error_step": "如完全正确则填'无'，如有误则明确指明第几步或摘录错误原句",
  "error_reason": "精准分析错误原因，禁止模糊用语",
  "corrected_solution": "提供正确的步骤或完整解法",
  "knowledge_tags": ["#数学·知识点1", "#数学·知识点2"],
  "score": "<0-100整数，必填>"
}}

# 标签规则：knowledge_tags 每一项必须以 #数学· 开头

# 打分规则（必须严格遵守）：
- is_correct 为 true 时：score 必须在 95-100 之间（完全正确给 100）
- is_correct 为 false 时：score 必须在 0-75 之间，按错误严重程度给分：
  · 思路完全错误 / 未作答 / 严重答非所问：0-30
  · 方法对部分但关键步骤错误：31-55
  · 思路正确但计算或符号失误：56-75
- 若 fatal_errors 非空，score 必须与熔断规则一致，禁止给高分
- 禁止无论对错都给出相同分数；score 必须与 is_correct 和 error_reason 一致"""

DEFAULT_LIBERAL_ARTS_PROMPT = """# 角色：
你是一位资深的语言文学教师，擅长阅读教学与作文精批，对《背影》等经典篇目有深入研究。

# 上下文：
你需要批改以下学生作业。
【本次批改侧重点】：{rubric}

# 任务与严格约束（V2.0 新增增强指令）：

【第一步：底线核查（熔断规则）】
在常规打分前，你必须先进行以下三项核查，若触犯，需直接触发降级扣分，并写入 fatal_errors：
1. 事实性错误核查：如果学生写错了文章作者（如把朱自清写成鲁迅）或弄错了核心人物关系，总分直接扣除20分。
2. 严重偏题核查：如果文章主体在讨论「交通安全」「买橘子攻防」等与「父爱/感恩」无关的硬逻辑问题，属于严重偏题。[情感理解]维度不得超过10分，总分不得超过60分。
3. 反套话侦测：如果文章大量使用「父爱如山、母爱如水」等烂大街词汇，且没有写出任何一件具体的、个人的小事，[个人感悟]维度不得超过15分。

【第二步：精准评价】
避免使用「很好」「不错」「文笔优美」等模糊评价。
- 对优点：必须指出具体好在哪个词、哪个细节。
- 对缺点：不要过度宽容！复述课文不是感悟，华丽辞藻不代表真情。必须明确指出其空洞或病句所在。

【学生作业内容】：
{student_answer}

# 输出要求：
请严格按照以下JSON格式输出，绝对不要包含任何额外的解释性文本、Markdown标记（如```json）或问候语，确保输出可直接被Python的json.loads()解析。
字符串值中禁止出现未转义的单个反斜杠；数学符号请用 plain text（如 x∈[1,2]）或将反斜杠写为双反斜杠（\\\\in）。
{{
  "fatal_errors": ["触发的熔断规则说明，无则为空数组[]"],
  "score": "<0-100整数，必填>",
  "dimension_scores": {{
    "情感理解": "<0-40整数>",
    "个人感悟": "<0-30整数>",
    "语言表达": "<0-20整数>",
    "结构逻辑": "<0-10整数>"
  }},
  "knowledge_tags": ["#语文·情感理解偏差", "#语文·套话堆砌"],
  "strengths": "具体优点，须引用原文词句",
  "weaknesses": "具体不足，须指出空洞/病句/偏题位置",
  "detailed_feedback": "错误指正与批改详情",
  "student_comment": "给学生的鼓励评语，亲切委婉"
}}

# 标签规则（语文专用）：
- knowledge_tags 每一项 **必须以 #语文· 开头**，使用中文标签名
- 示例：#语文·作者事实错误 #语文·严重偏题 #语文·套话堆砌 #语文·情感理解薄弱 #语文·病句
- **禁止** 使用英语标签（如 #PastTense #Grammar）或无前缀标签

# 打分规则：
- dimension_scores 四项之和 **必须等于** score（满分100），以维度分为准反推总分
- fatal_errors 非空时，score 必须与熔断规则一致（偏题≤60，事实错误扣20，套话扣感悟分）
- **严禁**不同作业给出相同总分（尤其禁止都写 92、90、88 等“安全分”）
- **严禁** dimension_scores 每次都给相同组合（如 38/28/18/8）；必须随作业缺陷数量浮动
- 有明显缺点（weaknesses 非空）时，总分通常应低于 85；仅当几乎无硬伤时才给 90+"""

DEFAULT_ENGLISH_PROMPT = """# Role:
You are an experienced English teacher with rigorous grading standards.

# Context:
Grade the following student homework.
【Grading Focus / 本次批改侧重点】：{rubric}

# Task & Strict Constraints (V2.0):

【Step 1: Bottom-line Checks (Circuit Breaker)】
Before scoring, check these three rules. If violated, record in fatal_errors and apply penalty:
1. Factual/Language Misuse: Serious misuse of key grammar patterns (e.g. consistent tense chaos) or wrong meaning of core vocabulary — deduct 15-20 points from total score.
2. Severe Off-topic: Essay does not address the assigned topic/prompt at all — [Content] dimension ≤ 10, total score ≤ 60.
3. Template/Cliché Detection: Heavy use of generic phrases ("English is important", "I like English very much") with no personal examples or original sentences — [Personal Expression] dimension ≤ 15.

【Step 2: Precise Evaluation】
Avoid vague praise like "good" or "nice". Cite original English sentences.
- Strengths: point to specific words/phrases that work well.
- Weaknesses: quote erroneous sentences, explain why wrong, suggest fixes.

【Student Homework / 学生作业】：
{student_answer}

# Output:
Output ONLY valid JSON parseable by Python json.loads(), no markdown fences.
Do not use unescaped backslashes in strings; use plain text or double backslashes (\\\\).
{{
  "fatal_errors": ["rule triggered, or empty []"],
  "score": "<0-100 integer, required>",
  "dimension_scores": {{
    "Grammar": "<0-30 integer>",
    "Vocabulary": "<0-25 integer>",
    "Structure": "<0-25 integer>",
    "Content": "<0-20 integer>"
  }},
  "knowledge_tags": ["#英语·PastTense", "#英语·SubjectVerbAgreement"],
  "strengths": "Specific strengths with quoted English",
  "weaknesses": "Specific errors with quoted English",
  "detailed_feedback": "Error details in Chinese, keep English citations",
  "improved_sentences": "Rewrite 1-3 problematic sentences",
  "student_comment": "Encouraging comment in Chinese"
}}

# Tag rules (English only):
- Every knowledge_tags item **must start with #英语·**
- Examples: #英语·PastTense #英语·Grammar #英语·Vocabulary #英语·OffTopic
- **Do NOT** use Chinese-subject tags (#语文·) or unprefixed tags

# Scoring rules:
- dimension_scores must sum to score (max 100); **use dimension sum as the total**
- If fatal_errors is not empty, score must reflect penalties (off-topic ≤60, etc.)
- **Never** assign the same total score to different essays (especially 92, 90, 88)
- Never reuse identical dimension splits (e.g. 28/22/22/20) for every student
- If weaknesses is non-empty, total score should usually be below 85"""

DEFAULT_ANALYTICS_PROMPT = """你是一位资深 **{subject_label}** 学科教研员。请 **仅基于以下 {subject_label} 批改记录** 生成学情分析，勿混入其他学科标签或评价维度。

【分析学科】：{subject_label}（只分析标签以 #{tag_prefix}· 开头的薄弱点）
【学生姓名】：{student_name}
【批改记录汇总】：
{records_summary}

请输出以下模块（使用 Emoji 标题）：
1. 📌 学情总览（一句话概括该生在 **{subject_label}** 方面的水平）
2. 🎯 高频薄弱点（仅统计 #{tag_prefix}· 标签，Top 3-5）
3. 📈 进步与风险（针对 {subject_label} 维度）
4. 💡 教学建议（给 **{subject_label}** 教师 3 条可操作建议）
5. 📝 下次练习方向（{subject_label} 针对性练习）

**禁止** 出现英语语法/时态分析（除非当前学科为英语）；**禁止** 出现数学解题分析（除非当前学科为数学）。"""

DEFAULT_PARENT_EMAIL_PROMPT = """# 角色
你是一位拥有多年教学经验、极其关心学生心理与成长的班主任。语气友善、充满鼓励，善于发现闪光点，也能委婉指出不足。

# 上下文
以下是一位学生的作业 AI 批改结果：
学生姓名：{student_name}
学科：{subject}
批改详情：
{grading_result}

# 任务
请为该学生写一封一对一的个性化反馈邮件，可直接复制发给家长或学生。

# 输出要求
1. 纯文本格式，不要 Markdown 代码块
2. 结构必须包含：
   - 邮件主题：（温馨且有针对性）
   - 称呼：
   - 正文第一段：公布分数（若有），给予肯定和鼓励
   - 正文第二段：具体表扬表现最好的一点
   - 正文第三段：指出需提升之处，给出可操作的改进建议（期望口吻，非指责）
   - 落款：关心你的老师 + {today}
3. 语调像真实老师面对面交流，不要机械念数据"""

DEFAULT_CLASS_REPORT_PROMPT = """# 角色
你是一位专业的教学数据分析师，擅长从批量作业数据中发现规律，为任课老师提供教学建议。

# 上下文
本次批量批改学科：{subject}
批改侧重点：{rubric}
共 {count} 份作业，汇总数据如下：
{batch_summary}

# 任务
请生成一份《班级学情分析报告》。

# 输出要求
以 Markdown 格式输出，必须包含：
1. **整体成绩概况**（最高/最低/平均分，整体达标情况）
2. **成绩明细表**（Markdown 表格：姓名、分数、薄弱标签）
3. **共性优点与典型问题**（各 2-3 条，可提及典型学生姓名）
4. **后续教学改进建议**（至少 3 条可落地建议）"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def extract_text_from_pdf_bytes(raw: bytes) -> tuple[str | None, str]:
    """从 PDF 字节流提取文本，返回 (文本, 错误信息)。"""
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        text = "\n\n".join(pages).strip()
        if not text:
            return None, "未能提取文字（可能是扫描版 PDF，需 OCR）"
        return text, ""
    except Exception as e:
        return None, str(e)


def parse_pdf_upload(uploaded_file) -> str | None:
    """解析 PDF 上传组件，返回提取文本；失败时展示提示。"""
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    text, err = extract_text_from_pdf_bytes(raw)
    if err:
        st.error(f"PDF 解析失败：{err}")
        return None
    reader = PdfReader(io.BytesIO(raw))
    st.success(f"✅ PDF 已解析：{len(reader.pages)} 页，共 {len(text)} 个字符")
    with st.expander("📖 PDF 提取预览", expanded=False):
        preview = text if len(text) <= 3000 else text[:3000] + "\n\n...(内容过长，已截断预览)"
        st.text(preview)
    return text


def merge_text_and_pdf(manual_text: str, pdf_text: str | None) -> str:
    """合并手动输入与 PDF 提取文本；两者都有则拼接。"""
    parts = []
    if manual_text.strip():
        parts.append(manual_text.strip())
    if pdf_text and pdf_text.strip():
        parts.append(pdf_text.strip())
    return "\n\n".join(parts)


def render_pdf_uploader(label: str, key: str) -> str | None:
    """渲染 PDF 上传控件并返回提取文本。"""
    uploaded = st.file_uploader(
        label,
        type=["pdf"],
        key=key,
        help="支持文字版 PDF；也可与下方文本框同时使用，内容会自动合并",
    )
    return parse_pdf_upload(uploaded)


def get_openai_client() -> OpenAI:
    """创建 OpenAI 兼容客户端，从环境变量读取 API Key。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY。请在 .env 文件或系统环境变量中配置该密钥。"
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def _backslash_count_before(text: str, index: int) -> int:
    """统计 index 位置之前连续反斜杠数量。"""
    count = 0
    j = index - 1
    while j >= 0 and text[j] == "\\":
        count += 1
        j -= 1
    return count


def sanitize_json_escapes(json_str: str) -> str:
    """
    修复大模型 JSON 中的无效转义（常见于 LaTeX：\\in、\\frac 等）。
    JSON 合法转义仅：\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX
    """
    valid_single = set('"\\/bfnrt')
    out: list[str] = []
    i = 0
    in_string = False

    while i < len(json_str):
        ch = json_str[i]

        if ch == '"':
            if _backslash_count_before(json_str, i) % 2 == 0:
                in_string = not in_string
            out.append(ch)
            i += 1
            continue

        if ch == "\\" and in_string and i + 1 < len(json_str):
            nxt = json_str[i + 1]
            if nxt in valid_single:
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            if nxt == "u" and i + 5 < len(json_str):
                hex_part = json_str[i + 2 : i + 6]
                if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                    out.append(json_str[i : i + 6])
                    i += 6
                    continue
            out.append("\\\\")
            out.append(nxt)
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def normalize_latex_delimiters(text: str) -> str:
    """统一 LaTeX 定界符为 Streamlit Markdown 支持的 $ / $$ 格式。"""
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return text


def render_latex_rich_text(text: str) -> None:
    """
    渲染含 LaTeX 的混合文本（Streamlit Markdown + KaTeX）。
    支持 $...$ 行内公式、$$...$$ 块级公式，以及自动识别无 $ 包裹的公式行。
    """
    if not text or str(text).strip() in ("无", "-", ""):
        st.caption("无")
        return

    content = normalize_latex_delimiters(str(text).strip())
    latex_cmd = re.compile(r"\\(frac|sqrt|in|left|right|infty|leq|geq|cdot|sum|int|pi)")

    rendered_lines: list[str] = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            rendered_lines.append("")
            continue
        if line.count("$") == 0 and latex_cmd.search(line):
            rendered_lines.append(f"$${line}$$")
        else:
            rendered_lines.append(line)

    # Markdown 硬换行：行尾两空格
    st.markdown("  \n".join(rendered_lines))


def render_math_content_block(title: str, content: str) -> None:
    """数学模式专用：带边框的内容块 + LaTeX 渲染。"""
    if not content or str(content).strip() in ("无", "-", ""):
        return
    st.markdown(f"**{title}**")
    with st.container(border=True):
        render_latex_rich_text(str(content))


def parse_json_response(text: str) -> dict:
    """从大模型响应中提取并解析 JSON（含 LaTeX 转义容错）。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    json_str = match.group(0) if match else cleaned

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return json.loads(sanitize_json_escapes(json_str))


def extract_tags_from_text(text: str) -> list[str]:
    """从文本中提取 #标签 格式的知识点。"""
    tags = re.findall(r"#[\w\u4e00-\u9fff\-·]+", text)
    return list(dict.fromkeys(tags))


def record_matches_subject(record_subject: str, filter_subject: str) -> bool:
    """判断一条记录的学科是否匹配筛选（含旧名称兼容）。"""
    if filter_subject == "全部学科":
        return True
    aliases = SUBJECT_ALIASES.get(filter_subject, [filter_subject])
    return record_subject in aliases


def normalize_tags_by_subject(tags: list[str], subject: str) -> list[str]:
    """统一为 #学科·标签名 格式，避免语文/英语标签混用。"""
    prefix = SUBJECT_TAG_PREFIX.get(subject)
    if not prefix:
        return list(dict.fromkeys(str(t).strip() for t in tags if t and str(t).strip()))

    english_to_chinese = {
        "pasttense": "时态混用",
        "grammar": "语法问题",
        "subjectverbagreement": "主谓一致",
        "wordchoice": "用词不当",
        "vocabulary": "词汇薄弱",
        "offtopic": "偏题",
        "letterformat": "格式问题",
    }

    result: list[str] = []
    for raw in tags:
        tag = str(raw).strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = "#" + tag

        body = tag.lstrip("#")
        if "·" in body:
            head, rest = body.split("·", 1)
            if head in SUBJECT_TAG_PREFIX.values():
                body = rest

        if subject == SUBJECT_ARTS and re.match(r"^[A-Za-z][A-Za-z0-9]*$", body):
            body = english_to_chinese.get(body.lower(), "表达问题")

        normalized = f"#{prefix}·{body}"
        result.append(normalized)

    return list(dict.fromkeys(result))


def tag_belongs_to_subject(tag: str, subject: str) -> bool:
    """判断标签是否属于指定学科前缀。"""
    prefix = SUBJECT_TAG_PREFIX.get(subject)
    if not prefix:
        return True
    return str(tag).startswith(f"#{prefix}·")


def get_subject_label(subject: str) -> str:
    """学科显示名（用于学情综述 Prompt）。"""
    return SUBJECT_TAG_PREFIX.get(subject, subject)


def extract_score_from_text(text: str) -> int | None:
    """从批改文本中提取百分制分数，优先匹配「综合评分」行，避免误匹配正文中的数字。"""
    priority_patterns = [
        r"综合评分[：:\s]*(\d{1,3})\s*分",
        r"Overall Score[^\n]{0,40}?(\d{1,3})\s*分",
        r"Overall Score[：:\s]*(\d{1,3})",
        r"Score[：:\s]*(\d{1,3})\s*(?:分|/100)?",
        r"(\d{1,3})\s*/\s*100",
    ]
    for pattern in priority_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score
    return None


def _coerce_score(raw_score) -> int | None:
    """将模型返回的 score 字段转为整数。"""
    if raw_score is None:
        return None
    if isinstance(raw_score, bool):
        return None
    if isinstance(raw_score, (int, float)):
        s = int(raw_score)
        return s if 0 <= s <= 100 else None
    if isinstance(raw_score, str):
        match = re.search(r"\d{1,3}", raw_score)
        if match:
            s = int(match.group())
            return s if 0 <= s <= 100 else None
    return None


def normalize_math_score(is_correct: bool, raw_score) -> int:
    """
    数学模式分数规范化：修正模型照搬 JSON 示例值 85 的问题。
    - 完全正确 → 100 分（若模型给分 < 90 或为示例值 85，强制纠正）
    - 解答错误 → 最高 75 分；若为示例值 85 或无有效分数，给合理默认低分
    """
    score = _coerce_score(raw_score)

    if is_correct:
        if score is None or score < 90 or score == MATH_TEMPLATE_ANCHOR_SCORE:
            return 100
        return min(score, 100)

    # 解答错误
    if score is None or score == MATH_TEMPLATE_ANCHOR_SCORE:
        return 35
    if score >= 90:
        return 55
    return max(0, min(score, 75))


def _parse_dimension_scores(dims) -> dict[str, int]:
    """解析 dimension_scores 为整数 dict。"""
    if not isinstance(dims, dict):
        return {}
    parsed: dict[str, int] = {}
    for key, val in dims.items():
        s = _coerce_score(val)
        if s is not None:
            parsed[str(key)] = s
    return parsed


def _scale_dimensions_to_total(dims: dict[str, int], target: int) -> dict[str, int]:
    """按比例缩放各维度分，使之和等于 target。"""
    if not dims or target is None:
        return dims
    current = sum(dims.values())
    if current <= 0:
        return dims
    if current == target:
        return dims
    scaled = {k: max(0, int(v * target / current)) for k, v in dims.items()}
    diff = target - sum(scaled.values())
    if diff != 0:
        first = next(iter(scaled))
        scaled[first] = max(0, scaled[first] + diff)
    return scaled


def normalize_homework_json(result: dict, subject: str) -> dict:
    """
    文科/英语分数服务端校验：
    1. 以 dimension_scores 求和为基准
    2. 强制执行 fatal_errors 熔断上限
    3. 修正模型偷懒锚定分（如人人 92 分）
    """
    result = dict(result)
    fatal = result.get("fatal_errors", [])
    if isinstance(fatal, str):
        fatal = [fatal] if fatal.strip() and fatal.strip() not in ("[]", "无") else []
    fatal_text = " ".join(map(str, fatal)) if fatal else ""

    caps = ARTS_DIMENSION_CAPS if subject == SUBJECT_ARTS else ENGLISH_DIMENSION_CAPS
    dims = _parse_dimension_scores(result.get("dimension_scores", {}))

    for key, cap in caps.items():
        if key in dims:
            dims[key] = min(dims[key], cap)

    # 熔断规则：强制压维度/总分上限
    if subject == SUBJECT_ARTS and fatal_text:
        if any(x in fatal_text for x in ("偏题", "无关", "交通安全", "买橘子攻防")):
            dims["情感理解"] = min(dims.get("情感理解", 40), 10)
        if any(x in fatal_text for x in ("套话", "父爱如山", "母爱如水", "千篇一律")):
            dims["个人感悟"] = min(dims.get("个人感悟", 30), 15)

    if dims:
        score = sum(dims.values())
    else:
        score = _coerce_score(result.get("score"))

    if score is None:
        result["dimension_scores"] = dims
        return result

    if fatal_text:
        if subject == SUBJECT_ARTS:
            if any(x in fatal_text for x in ("偏题", "无关", "交通安全")):
                score = min(score, 60)
            if any(x in fatal_text for x in ("事实", "作者", "朱自清", "鲁迅", "写错")):
                score = max(0, score - 20)
            if any(x in fatal_text for x in ("套话", "父爱如山")):
                score = min(score, 75)

    weaknesses = str(result.get("weaknesses", ""))
    detailed = str(result.get("detailed_feedback", ""))
    defect_text = weaknesses + detailed

    # 有明显缺陷却给 88-95 锚定分 → 下调
    if score in HOMEWORK_ANCHOR_SCORES and len(defect_text.strip()) > 30:
        penalty = 12 + min(18, len(defect_text) // 40)
        score = max(40, score - penalty)

    # 无熔断但缺陷描述含严重问题词 → 上限 78
    if not fatal_text and score >= 88:
        if any(w in defect_text for w in ("偏题", "错误", "病句", "空洞", "复述", "套话", "作者", "无关")):
            score = min(score, 78)

    # 几乎无缺陷才允许 90+
    if score >= 90 and len(defect_text.strip()) > 15:
        if any(w in defect_text for w in ("不足", "缺点", "问题", "改进", "欠缺", "空洞", "病句")):
            score = min(score, 84)

    if dims:
        dims = _scale_dimensions_to_total(dims, score)
        result["dimension_scores"] = dims

    result["score"] = score
    return result


def parse_tags_from_result(result: dict) -> list[str]:
    """从 JSON 结果中提取知识点标签。"""
    tags = result.get("knowledge_tags", [])
    if isinstance(tags, str):
        return extract_tags_from_text(tags)
    if isinstance(tags, list):
        return [str(t) for t in tags if t]
    return []


def render_fatal_errors(fatal_errors) -> None:
    """展示 V2.0 熔断规则触发情况。"""
    if not fatal_errors:
        return
    if isinstance(fatal_errors, str):
        fatal_errors = [fatal_errors]
    if fatal_errors:
        st.error("⚠️ 触发底线核查（熔断规则）")
        for item in fatal_errors:
            st.warning(str(item))


def parse_grading_json(raw: str, subject: str) -> dict:
    """解析文科/英语 JSON 批改结果，提取学情字段（含分数校验）。"""
    result = parse_json_response(raw)
    raw_score = _coerce_score(result.get("score"))
    result = normalize_homework_json(result, subject)
    tags = normalize_tags_by_subject(parse_tags_from_result(result), subject)
    result["knowledge_tags"] = tags
    score = _coerce_score(result.get("score"))
    score_adjusted = raw_score is not None and score != raw_score
    fatal = result.get("fatal_errors", [])
    parts = []
    if fatal:
        parts.append("熔断:" + "; ".join(map(str, fatal)))
    for key in ("weaknesses", "detailed_feedback", "student_comment"):
        if result.get(key):
            parts.append(str(result[key])[:120])
            break
    summary = " | ".join(parts) if parts else raw[:200]
    return {
        "result": result,
        "tags": tags,
        "score": score,
        "score_adjusted": score_adjusted,
        "raw_score": raw_score,
        "summary": summary,
        "grading_text": raw,
    }


def render_homework_json_result(raw_text: str, student_name: str, subject: str) -> None:
    """渲染文科/英语 JSON 批改结果并写入学情。"""
    try:
        parsed = parse_grading_json(raw_text, subject)
        result = parsed["result"]
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"JSON 解析失败：{e}")
        st.markdown("**原始响应：**")
        st.code(raw_text, language="text")
        return

    render_fatal_errors(result.get("fatal_errors"))

    score = parsed["score"]
    if score is not None:
        st.metric("📊 综合评分", f"{score} 分")
        if parsed.get("score_adjusted"):
            st.caption(
                f"ℹ️ 已对模型原始分数 {parsed.get('raw_score')} 分进行服务端校验"
                "（维度求和 / 熔断规则 / 反锚定分）"
            )

    dims = result.get("dimension_scores", {})
    if isinstance(dims, dict) and dims:
        st.markdown("**📐 各维度得分**")
        cols = st.columns(min(len(dims), 4))
        for col, (name, val) in zip(cols, dims.items()):
            col.metric(str(name), f"{val}分")

    tags = parsed["tags"]
    if tags:
        st.markdown("**🏷️ 薄弱知识点标签**")
        st.markdown(" ".join(tags))

    if result.get("strengths"):
        st.success(f"**✅ 具体优点**\n\n{result['strengths']}")
    if result.get("weaknesses"):
        st.warning(f"**❌ 具体不足**\n\n{result['weaknesses']}")
    if result.get("detailed_feedback"):
        st.info(f"**📝 批改详情**\n\n{result['detailed_feedback']}")
    if result.get("improved_sentences"):
        st.markdown("**✏️ 修改示范**")
        st.code(result["improved_sentences"], language="text")
    if result.get("student_comment"):
        st.markdown(f"**👧 给学生的评语**\n\n{result['student_comment']}")

    add_grading_record(
        subject=subject,
        student_name=student_name,
        tags=tags,
        score=score,
        is_correct=None,
        summary=parsed["summary"],
    )
    st.caption("✅ 已记录至学情分析")


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


def call_text_grading(model: str, prompt: str) -> str:
    """非流式批改，供批量模式及文科/英语 JSON 模式使用。"""
    response = call_llm(model, prompt, "请开始批改，仅输出 JSON。", stream=False)
    return response.choices[0].message.content or ""


def student_name_from_filename(filename: str) -> str:
    """从文件名推断学生姓名（PDF 内无个人信息时的兜底）。"""
    base = os.path.splitext(os.path.basename(filename))[0]
    for sep in ("_", "-", " "):
        if sep in base:
            return base.split(sep)[0].strip()
    return base.strip() or "未命名学生"


def extract_student_info_from_text(text: str) -> dict:
    """
    从 PDF 正文头部提取学生个人信息。
    支持常见格式：姓名、学号、班级（中英文标签均可）。
    """
    header = text[:1500]
    info = {"name": "", "student_id": "", "class_name": ""}

    name_patterns = [
        r"姓\s*名\s*[：:]\s*([\u4e00-\u9fffA-Za-z·\s]{2,20})",
        r"学生姓名\s*[：:]\s*([\u4e00-\u9fffA-Za-z·\s]{2,20})",
        r"Name\s*[：:]\s*([A-Za-z][A-Za-z\s]{1,30})",
    ]
    id_patterns = [
        r"学\s*号\s*[：:]\s*([A-Za-z0-9\-]{4,20})",
        r"Student\s*ID\s*[：:]\s*([A-Za-z0-9\-]{4,20})",
    ]
    class_patterns = [
        r"班\s*级\s*[：:]\s*([^\n\r]{2,30})",
        r"Class\s*[：:]\s*([^\n\r]{2,30})",
    ]

    for pat in name_patterns:
        m = re.search(pat, header, re.IGNORECASE)
        if m:
            info["name"] = m.group(1).strip()
            break
    for pat in id_patterns:
        m = re.search(pat, header, re.IGNORECASE)
        if m:
            info["student_id"] = m.group(1).strip()
            break
    for pat in class_patterns:
        m = re.search(pat, header, re.IGNORECASE)
        if m:
            info["class_name"] = m.group(1).strip().rstrip("，,。.")
            break

    return info


def resolve_student_identity(text: str, filename: str) -> dict:
    """
    综合 PDF 内容与文件名，确定学生身份。
    优先使用 PDF 内的个人信息；姓名缺失时回退到文件名。
    """
    info = extract_student_info_from_text(text)
    fallback_name = student_name_from_filename(filename)
    name = info["name"] or fallback_name
    source = "PDF正文" if info["name"] else "文件名"

    # 展示标签：姓名 + 班级 + 学号（用于列表区分同名学生）
    label_parts = [name]
    if info["class_name"]:
        label_parts.append(info["class_name"])
    if info["student_id"]:
        label_parts.append(f"学号{info['student_id']}")
    display_label = " · ".join(label_parts)

    # 学情/邮件使用的唯一键：有学号则附加学号避免同名冲突
    unique_key = f"{name}({info['student_id']})" if info["student_id"] else name

    return {
        "name": name,
        "student_id": info["student_id"],
        "class_name": info["class_name"],
        "display_label": display_label,
        "unique_key": unique_key,
        "info_source": source,
    }


def is_pdf_filename(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def collect_pdf_files_from_folder(folder_path: str) -> tuple[list[tuple[str, bytes]], list[str]]:
    """扫描本地文件夹，仅收集 PDF；返回 (pdf列表, 被跳过的非PDF文件名)。"""
    folder = folder_path.strip().strip('"').strip("'")
    if not folder or not os.path.isdir(folder):
        raise ValueError(f"文件夹不存在或路径无效：{folder}")
    files = []
    skipped = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            continue
        if is_pdf_filename(name):
            with open(path, "rb") as f:
                files.append((name, f.read()))
        else:
            skipped.append(name)
    if not files:
        raise ValueError(
            f"文件夹中未找到 PDF 文件：{folder}\n"
            "批量模式仅支持 .pdf 格式，请将作业保存为 PDF 后放入文件夹。"
        )
    return files, skipped


def collect_pdf_files_from_zip(uploaded_zip) -> tuple[list[tuple[str, bytes]], list[str]]:
    """从 ZIP 中仅提取 PDF 文件。"""
    files = []
    skipped = []
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue
            base = os.path.basename(name)
            if is_pdf_filename(base):
                files.append((base, zf.read(name)))
            else:
                skipped.append(base)
    if not files:
        raise ValueError("ZIP 中未找到 PDF 文件。批量模式仅支持 .pdf 格式。")
    return files, skipped


def collect_pdf_files_from_uploads(uploaded_files) -> list[tuple[str, bytes]]:
    """从多文件上传组件收集 PDF（拒绝非 PDF）。"""
    files = []
    for f in uploaded_files:
        if not is_pdf_filename(f.name):
            raise ValueError(f"文件「{f.name}」不是 PDF，批量模式仅支持 .pdf 格式。")
        f.seek(0)
        files.append((f.name, f.read()))
    return files


def grade_batch_item(
    subject: str,
    model: str,
    rubric: str,
    content: str,
    question: str = "",
    standard_answer: str = "",
) -> dict:
    """批改单份作业，返回结构化结果。"""
    if subject == SUBJECT_MATH:
        prompt = build_math_prompt(question, standard_answer, content)
        raw = call_math_grading(model, prompt)
        try:
            result = parse_json_response(raw)
            is_correct = bool(result.get("is_correct", False))
            score = normalize_math_score(is_correct, result.get("score"))
            tags = result.get("knowledge_tags", [])
            if isinstance(tags, str):
                tags = extract_tags_from_text(tags)
            tags = normalize_tags_by_subject(tags, SUBJECT_MATH)
            summary = result.get("error_reason", "无") if not is_correct else "解答完全正确"
            fatal = result.get("fatal_errors", [])
            if fatal:
                summary = f"熔断:{'；'.join(map(str, fatal))} | {summary}"
            grading_text = raw
        except (json.JSONDecodeError, ValueError):
            is_correct = None
            score = None
            tags = []
            summary = raw[:200]
            grading_text = raw
    elif subject == SUBJECT_ARTS:
        prompt = build_arts_prompt(rubric, content)
        raw = call_text_grading(model, prompt)
        try:
            parsed = parse_grading_json(raw, subject)
            tags = parsed["tags"]
            score = parsed["score"]
            summary = parsed["summary"]
            grading_text = parsed["grading_text"]
            is_correct = None
        except (json.JSONDecodeError, ValueError):
            tags = normalize_tags_by_subject(extract_tags_from_text(raw), subject)
            score = extract_score_from_text(raw)
            is_correct = None
            summary = raw[:200]
            grading_text = raw
    else:
        prompt = build_english_prompt(rubric, content)
        raw = call_text_grading(model, prompt)
        try:
            parsed = parse_grading_json(raw, subject)
            tags = parsed["tags"]
            score = parsed["score"]
            summary = parsed["summary"]
            grading_text = parsed["grading_text"]
            is_correct = None
        except (json.JSONDecodeError, ValueError):
            tags = normalize_tags_by_subject(extract_tags_from_text(raw), subject)
            score = extract_score_from_text(raw)
            is_correct = None
            summary = raw[:200]
            grading_text = raw

    return {
        "score": score,
        "tags": tags,
        "is_correct": is_correct,
        "summary": summary,
        "grading_text": grading_text,
    }


def process_one_batch_pdf(
    filename: str,
    raw: bytes,
    subject: str,
    model: str,
    rubric: str,
    question: str = "",
    standard_answer: str = "",
) -> dict:
    """批改单份 PDF，返回一条批量结果记录。"""
    text, err = extract_text_from_pdf_bytes(raw)
    if err:
        identity = resolve_student_identity("", filename)
        return {
            "filename": filename,
            "student_name": identity["unique_key"],
            "display_label": identity["display_label"],
            "student_id": identity["student_id"],
            "class_name": identity["class_name"],
            "info_source": identity["info_source"],
            "status": "失败",
            "error": err,
            "score": None,
            "tags": [],
            "grading_text": "",
            "summary": err,
        }

    identity = resolve_student_identity(text, filename)
    student = identity["unique_key"]
    try:
        graded = grade_batch_item(
            subject, model, rubric, text, question, standard_answer
        )
        add_grading_record(
            subject=subject,
            student_name=student,
            tags=graded["tags"],
            score=graded["score"],
            is_correct=graded["is_correct"],
            summary=graded["summary"],
        )
        return {
            "filename": filename,
            "student_name": student,
            "display_label": identity["display_label"],
            "student_id": identity["student_id"],
            "class_name": identity["class_name"],
            "info_source": identity["info_source"],
            "status": "成功",
            "error": "",
            **graded,
        }
    except Exception as e:
        return {
            "filename": filename,
            "student_name": student,
            "display_label": identity["display_label"],
            "student_id": identity["student_id"],
            "class_name": identity["class_name"],
            "info_source": identity["info_source"],
            "status": "失败",
            "error": str(e),
            "score": None,
            "tags": [],
            "grading_text": "",
            "summary": str(e),
        }


def run_batch_grading(
    pdf_files: list[tuple[str, bytes]],
    subject: str,
    model: str,
    rubric: str,
    question: str = "",
    standard_answer: str = "",
) -> list[dict]:
    """批量批改 PDF 列表（本地同步模式，一次跑完）。"""
    results = []
    progress = st.progress(0, text="准备批量批改…")
    status = st.empty()

    for i, (filename, raw) in enumerate(pdf_files):
        identity_hint = resolve_student_identity("", filename)
        status.info(
            f"正在批改 ({i + 1}/{len(pdf_files)})：**{identity_hint['display_label']}** — `{filename}`"
        )
        results.append(
            process_one_batch_pdf(
                filename, raw, subject, model, rubric, question, standard_answer
            )
        )
        progress.progress((i + 1) / len(pdf_files), text=f"已完成 {i + 1}/{len(pdf_files)}")

    progress.empty()
    status.empty()
    return results


def _clear_batch_job() -> None:
    """清理批量任务 session 状态。"""
    for key in (
        "batch_active",
        "batch_queue",
        "batch_index",
        "batch_results_partial",
        "batch_meta",
    ):
        st.session_state.pop(key, None)


def _batch_grading_worker() -> None:
    """
    逐份批改（兼容 Hugging Face 旧版 Streamlit，不依赖 st.fragment）。
    PDF 字节已缓存在 session_state.batch_queue，整页 rerun 不会重复读 ZIP。
    """
    if not st.session_state.get("batch_active"):
        return

    queue = st.session_state.get("batch_queue", [])
    idx = st.session_state.get("batch_index", 0)
    meta = st.session_state.get("batch_meta", {})
    partial: list = st.session_state.get("batch_results_partial", [])
    total = len(queue)

    if total == 0:
        _clear_batch_job()
        st.error("批量队列为空，请重新上传 ZIP 并开始批改。")
        return

    st.info(
        f"⏳ 批量批改进行中：**{idx}/{total}**（云端逐份处理，请勿关闭页面）"
    )

    if idx >= total:
        return

    filename, raw = queue[idx]
    identity = resolve_student_identity("", filename)
    progress = st.progress(idx / total, text=f"正在批改第 {idx + 1}/{total} 份…")
    st.caption(
        f"**{identity['display_label']}**（识别来源：{identity['info_source']}）— `{filename}`"
    )

    result = process_one_batch_pdf(
        filename,
        raw,
        meta.get("subject", SUBJECT_ARTS),
        meta.get("model", "qwen-turbo"),
        meta.get("rubric", ""),
        meta.get("question", ""),
        meta.get("standard_answer", ""),
    )
    partial.append(result)
    st.session_state.batch_results_partial = partial
    st.session_state.batch_index = idx + 1

    progress.progress(
        (idx + 1) / total,
        text=f"已完成 {idx + 1}/{total} — {result['status']}",
    )

    if st.session_state.batch_index >= total:
        st.session_state.batch_results = partial
        ok = sum(1 for r in partial if r["status"] == "成功")
        st.session_state.batch_finish_msg = (
            f"批量批改完成：成功 {ok}/{len(partial)} 份"
        )
        _clear_batch_job()
        return

    st.rerun()


def _render_batch_results_table(results: list[dict]) -> None:
    """展示批量批改结果表与锚定分提示。"""
    st.markdown("### 📋 批量批改结果")
    st.dataframe(
        [
            {
                "文件名": r["filename"],
                "学生": r.get("display_label") or r["student_name"],
                "学号": r.get("student_id") or "-",
                "班级": r.get("class_name") or "-",
                "识别来源": r.get("info_source") or "-",
                "状态": r["status"],
                "分数": r.get("score") if r.get("score") is not None else "-",
                "标签": " ".join(r.get("tags", [])) or "-",
                "备注": r.get("error") or r.get("summary", "")[:50],
            }
            for r in results
        ],
        use_container_width=True,
        hide_index=True,
    )

    fallback_count = sum(1 for r in results if r.get("info_source") == "文件名")
    if fallback_count:
        st.warning(
            f"有 **{fallback_count}** 份 PDF 未识别到正文「姓名」，已用文件名兜底。"
            "请确保 PDF 开头包含：姓名、学号、班级。"
        )

    success_scores = [
        r["score"] for r in results
        if r["status"] == "成功" and r.get("score") is not None
    ]
    if len(success_scores) >= 2 and len(set(success_scores)) == 1:
        st.warning(
            f"⚠️ 批量批改中 **{len(success_scores)}** 份作业总分均为 **{success_scores[0]} 分**，"
            "可能存在模型锚定分问题。已启用服务端校验；若仍相同，请换用 qwen-plus 或检查 Prompt。"
        )


def _render_batch_emails_and_report(
    results: list[dict], subject_mode: str, selected_model: str, rubric: str
) -> None:
    """生成并展示反馈邮件与班级报告（仅用户点击按钮时调用）。"""
    success_items = [r for r in results if r["status"] == "成功"]
    if not success_items:
        st.warning("没有成功的批改记录，无法生成邮件")
        return

    st.markdown("### 📧 个性化反馈邮件")
    all_emails = []
    mail_progress = st.progress(0, text="生成邮件中…")

    for i, item in enumerate(success_items):
        label = item.get("display_label") or item["student_name"]
        with st.spinner(f"生成 {label} 的邮件…"):
            email_body = generate_parent_email(
                selected_model,
                item["student_name"],
                subject_mode,
                item["grading_text"],
                display_label=item.get("display_label", ""),
                student_id=item.get("student_id", ""),
                class_name=item.get("class_name", ""),
            )
        all_emails.append(
            f"{'=' * 50}\n"
            f"【{label}】\n"
            f"{'=' * 50}\n\n"
            f"{email_body}\n"
        )
        with st.expander(f"📨 {label} 的反馈邮件"):
            st.text(email_body)
        mail_progress.progress(
            (i + 1) / len(success_items),
            text=f"邮件 {i + 1}/{len(success_items)}",
        )

    mail_progress.empty()
    combined = "\n".join(all_emails)
    st.download_button(
        "⬇️ 下载全部邮件（TXT）",
        combined,
        file_name=f"反馈邮件_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        key="download_emails",
    )

    st.markdown("### 📊 班级学情分析报告")
    with st.spinner("生成班级学情报告…"):
        class_report = generate_class_report(
            selected_model, subject_mode, rubric, results
        )
    st.markdown(class_report)
    st.download_button(
        "⬇️ 下载班级报告（MD）",
        class_report,
        file_name=f"班级学情报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        key="download_report",
    )


def build_batch_summary(results: list[dict]) -> str:
    """格式化批量结果供班级报告使用。"""
    lines = []
    for r in results:
        label = r.get("display_label") or r["student_name"]
        if r["status"] != "成功":
            lines.append(f"- {label}（{r['filename']}）：批改失败 — {r['error']}")
            continue
        score_str = f"{r['score']}分" if r["score"] is not None else "未提取"
        tag_str = " ".join(r["tags"]) if r["tags"] else "无"
        id_str = f"学号{r['student_id']} | " if r.get("student_id") else ""
        class_str = f"班级{r['class_name']} | " if r.get("class_name") else ""
        lines.append(
            f"- {label} | {id_str}{class_str}分数:{score_str} | 标签:{tag_str} | "
            f"摘要:{r['summary'][:120]}"
        )
    return "\n".join(lines)


def generate_parent_email(
    model: str, student_name: str, subject: str, grading_result: str,
    display_label: str = "", student_id: str = "", class_name: str = "",
) -> str:
    """为单个学生生成个性化反馈邮件。"""
    student_context = student_name
    if display_label:
        student_context = display_label
    if student_id or class_name:
        extra = []
        if class_name:
            extra.append(f"班级：{class_name}")
        if student_id:
            extra.append(f"学号：{student_id}")
        student_context += "\n" + "，".join(extra)

    prompt = DEFAULT_PARENT_EMAIL_PROMPT.format(
        student_name=student_context,
        subject=subject,
        grading_result=grading_result,
        today=datetime.now().strftime("%Y年%m月%d日"),
    )
    return call_text_grading(model, prompt)


def generate_class_report(
    model: str, subject: str, rubric: str, results: list[dict]
) -> str:
    """生成班级学情分析报告。"""
    success = [r for r in results if r["status"] == "成功"]
    prompt = DEFAULT_CLASS_REPORT_PROMPT.format(
        subject=subject,
        rubric=rubric or "常规标准",
        count=len(success),
        batch_summary=build_batch_summary(results),
    )
    return call_text_grading(model, prompt)


def render_batch_tab(subject_mode: str, selected_model: str) -> None:
    """批量批改 Tab：文件夹 / 多 PDF / ZIP 导入 + 邮件生成。"""
    st.subheader("📦 批量批改 & 反馈邮件")
    st.caption(
        "批量模式 **仅支持 PDF**。系统会从每份 PDF 正文头部读取学生个人信息用于区分人员。"
    )

    st.info(
        "**PDF 格式要求**\n\n"
        "1. 文件格式：只能是 `.pdf`（文件夹 / ZIP 中的其他格式会自动跳过）\n"
        "2. 每份 PDF 开头须包含学生个人信息，例如：\n"
        "   ```\n"
        "   姓名：张三\n"
        "   学号：20230101\n"
        "   班级：初二(3)班\n"
        "   ```\n"
        "3. 系统优先按 PDF 内「姓名 / 学号 / 班级」识别学生；"
        "若无姓名则回退到文件名。"
    )

    import_mode = st.radio(
        "导入方式",
        ["📁 本地文件夹路径（仅本地运行）", "📄 多 PDF 上传", "🗜️ ZIP 压缩包（仅含 PDF）"],
        horizontal=True,
        key="batch_import_mode",
    )

    pdf_files: list[tuple[str, bytes]] = []
    skipped_non_pdf: list[str] = []
    folder_path = ""

    if import_mode.startswith("📁"):
        folder_path = st.text_input(
            "文件夹路径",
            placeholder=r"如：C:\Users\86186\Desktop\homework_pdfs",
            key="batch_folder_path",
        )
        st.caption("仅读取文件夹内的 `.pdf` 文件，Word/图片等会自动忽略")
    elif import_mode.startswith("📄"):
        multi_pdfs = st.file_uploader(
            "选择多个 PDF 文件（仅 PDF）",
            type=["pdf"],
            accept_multiple_files=True,
            key="batch_multi_pdf",
        )
        if multi_pdfs:
            try:
                pdf_files = collect_pdf_files_from_uploads(multi_pdfs)
                st.info(f"已选择 **{len(pdf_files)}** 个 PDF 文件")
            except ValueError as e:
                st.error(str(e))
    else:
        zip_file = st.file_uploader(
            "上传 ZIP 压缩包（包内仅保留 PDF 参与批改）",
            type=["zip"],
            key="batch_zip",
        )
        if zip_file:
            try:
                pdf_files, skipped_non_pdf = collect_pdf_files_from_zip(zip_file)
                st.info(f"ZIP 中共 **{len(pdf_files)}** 个 PDF 文件")
                if skipped_non_pdf:
                    st.warning(
                        f"已跳过 {len(skipped_non_pdf)} 个非 PDF 文件："
                        + "、".join(skipped_non_pdf[:8])
                        + ("…" if len(skipped_non_pdf) > 8 else "")
                    )
            except Exception as e:
                st.error(str(e))

    rubric = st.text_input(
        "🎯 批改侧重点（批量共用）",
        placeholder="如：议论文结构、过去时态（留空则按常规标准）",
        key="batch_rubric",
    )

    batch_question = ""
    batch_standard = ""
    if subject_mode == SUBJECT_MATH:
        st.warning("数学批量模式：所有 PDF 视为「学生解答」，请填写统一的题目与标准答案。")
        c1, c2 = st.columns(2)
        with c1:
            batch_question = st.text_area("📋 统一题目", height=100, key="batch_math_q")
        with c2:
            batch_standard = st.text_area(
                "✅ 统一标准答案", height=100, key="batch_math_std"
            )

    col_run, col_mail, col_cancel = st.columns([2, 2, 1])
    with col_run:
        start_batch = st.button(
            "🚀 开始批量批改",
            type="primary",
            use_container_width=True,
            key="btn_batch_run",
            disabled=st.session_state.get("batch_active", False),
        )
    with col_mail:
        gen_emails = st.button(
            "📧 生成反馈邮件 & 班级报告",
            use_container_width=True,
            key="btn_batch_email",
            disabled=st.session_state.get("batch_active", False),
        )
    with col_cancel:
        if st.session_state.get("batch_active"):
            if st.button("⏹ 取消", use_container_width=True, key="btn_batch_cancel"):
                _clear_batch_job()
                st.session_state.pop("batch_finish_msg", None)
                st.rerun()

    if start_batch:
        try:
            if import_mode.startswith("📁"):
                if not folder_path.strip():
                    st.warning("请输入文件夹路径")
                else:
                    pdf_files, skipped_non_pdf = collect_pdf_files_from_folder(folder_path)
                    st.success(f"从文件夹读取 **{len(pdf_files)}** 个 PDF")
                    if skipped_non_pdf:
                        st.warning(
                            f"已跳过 {len(skipped_non_pdf)} 个非 PDF 文件："
                            + "、".join(skipped_non_pdf[:8])
                            + ("…" if len(skipped_non_pdf) > 8 else "")
                        )
            if not pdf_files:
                st.warning("请先导入 PDF 文件")
            elif subject_mode == SUBJECT_MATH and not batch_question.strip():
                st.warning("数学批量模式需填写统一题目")
            elif st.session_state.get("batch_active"):
                st.warning("批量批改进行中，请等待完成或点击「取消」")
            else:
                st.session_state.batch_results = []
                st.session_state.pop("batch_finish_msg", None)
                st.session_state.batch_queue = list(pdf_files)
                st.session_state.batch_index = 0
                st.session_state.batch_results_partial = []
                st.session_state.batch_meta = {
                    "subject": subject_mode,
                    "model": selected_model,
                    "rubric": rubric,
                    "question": batch_question,
                    "standard_answer": batch_standard,
                }
                st.session_state.batch_active = True
                st.rerun()
        except Exception as e:
            st.error(f"批量批改失败：{e}")

    _batch_grading_worker()

    finish_msg = st.session_state.pop("batch_finish_msg", None)
    if finish_msg:
        st.success(finish_msg)

    if gen_emails:
        results = st.session_state.get("batch_results")
        if not results:
            st.warning("请先完成批量批改")
        else:
            _render_batch_emails_and_report(results, subject_mode, selected_model, rubric)

    if not st.session_state.get("batch_active"):
        results = st.session_state.get("batch_results")
        if results:
            _render_batch_results_table(results)

    st.caption(f"批量模块版本：{BATCH_MODULE_VERSION}（HF 部署后应显示此标识）")


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
    tags = normalize_tags_by_subject(tags, subject)
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
    if isinstance(is_correct, str):
        is_correct = is_correct.lower() in ("true", "1", "yes")
    render_fatal_errors(result.get("fatal_errors"))
    error_step = result.get("error_step", "无")
    error_reason = result.get("error_reason", "无")
    corrected_solution = result.get("corrected_solution", "")
    tags = result.get("knowledge_tags", [])
    if isinstance(tags, str):
        tags = extract_tags_from_text(tags)
    tags = normalize_tags_by_subject(tags, SUBJECT_MATH)
    score = result.get("score")
    score = normalize_math_score(is_correct, score)

    if is_correct:
        st.success("✅ 该题解答完全正确！")
    else:
        st.error("❌ 发现逻辑错误或计算失误")
        render_math_content_block("📍 出错步骤", error_step)
        render_math_content_block("💡 错误原因", error_reason)

    st.metric("📊 本题得分", f"{score} 分")

    if corrected_solution and corrected_solution != "无":
        st.markdown("**✅ 正确解法参考**")
        with st.container(border=True):
            render_latex_rich_text(corrected_solution)

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
    if score is not None:
        st.metric("📊 提取得分（已记入学情）", f"{score} 分")
    else:
        st.caption("⚠️ 未能从批改结果中提取分数，请确认模型输出了「XX分」格式")
    st.caption("✅ 已记录至学情分析")


def build_records_summary(
    history: list[dict], student_filter: str, subject_filter: str
) -> str:
    """将历史记录格式化为供 AI 分析的文本。"""
    filtered = [
        r
        for r in history
        if (student_filter == "全部学生" or r["student_name"] == student_filter)
        and record_matches_subject(r["subject"], subject_filter)
    ]
    if not filtered:
        return "（暂无记录）"

    lines = []
    for i, r in enumerate(filtered, 1):
        tag_str = "、".join(
            normalize_tags_by_subject(r["tags"], r["subject"])
        ) if r["tags"] else "无"
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


def render_analytics_tab(subject_mode: str, selected_model: str) -> None:
    """学情分析 Tab：统计图表 + AI 综述（按学科隔离标签）。"""
    st.subheader("📊 学情分析仪表盘")
    st.caption("自动汇总历次批改数据，洞察学生薄弱点与进步趋势（语文/英语/数学标签互不混用）")

    history = st.session_state.grading_history

    if not history:
        st.info("暂无批改记录。请先在「作业批改台」完成至少一次批改。")
        return

    students = sorted({r["student_name"] for r in history})
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        student_filter = st.selectbox(
            "👤 筛选学生",
            options=["全部学生"] + students,
            key="analytics_student_filter",
        )
    with filter_col2:
        subject_options = ALL_SUBJECTS + ["全部学科"]
        default_idx = (
            subject_options.index(subject_mode)
            if subject_mode in subject_options
            else 0
        )
        subject_filter = st.selectbox(
            "📚 筛选学科",
            options=subject_options,
            index=default_idx,
            key="analytics_subject_filter",
            help="默认跟随侧边栏当前分支；语文与英语标签分开统计",
        )

    filtered = [
        r
        for r in history
        if (student_filter == "全部学生" or r["student_name"] == student_filter)
        and record_matches_subject(r["subject"], subject_filter)
    ]

    if not filtered:
        st.warning(f"当前筛选条件下暂无 **{subject_filter}** 批改记录。")
        return

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
        if subject_filter == "全部学科":
            all_tags = [
                t
                for r in filtered
                for t in normalize_tags_by_subject(r["tags"], r["subject"])
            ]
        else:
            all_tags = [
                t
                for r in filtered
                for t in normalize_tags_by_subject(r["tags"], r["subject"])
                if tag_belongs_to_subject(t, subject_filter)
            ]
        st.metric("薄弱标签数", len(set(all_tags)))

    st.divider()

    # ---- 薄弱知识点频次 ----
    tag_counter = Counter(
        t
        for r in filtered
        for t in normalize_tags_by_subject(r["tags"], r["subject"])
        if subject_filter == "全部学科"
        or tag_belongs_to_subject(t, subject_filter)
    )
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
                "标签": " ".join(
                    normalize_tags_by_subject(r["tags"], r["subject"])
                ) if r["tags"] else "-",
            }
            for r in reversed(filtered)
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ---- AI 学情综述 ----
    subject_label = get_subject_label(subject_filter)
    tag_prefix = SUBJECT_TAG_PREFIX.get(subject_filter, subject_label)
    st.markdown(
        f"### 🧠 AI 学情综述（{subject_label}）"
        if subject_filter != "全部学科"
        else "### 🧠 AI 学情综述"
    )
    if st.button("生成 AI 学情分析报告", type="primary", key="btn_analytics"):
        if subject_filter == "全部学科":
            st.warning("请先选择具体学科（语文/英语/数学），再生成学情综述，避免标签混用。")
        else:
            records_summary = build_records_summary(
                history, student_filter, subject_filter
            )
            prompt = DEFAULT_ANALYTICS_PROMPT.format(
                subject_label=subject_label,
                tag_prefix=tag_prefix,
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
    """初始化 session_state；Prompt 版本升级时自动刷新默认模板。"""
    if st.session_state.get("prompt_version", 0) < PROMPT_VERSION:
        st.session_state.math_prompt = DEFAULT_MATH_PROMPT
        st.session_state.liberal_arts_prompt = DEFAULT_LIBERAL_ARTS_PROMPT
        st.session_state.english_prompt = DEFAULT_ENGLISH_PROMPT
        st.session_state.prompt_version = PROMPT_VERSION

    defaults = {
        "math_prompt": DEFAULT_MATH_PROMPT,
        "liberal_arts_prompt": DEFAULT_LIBERAL_ARTS_PROMPT,
        "english_prompt": DEFAULT_ENGLISH_PROMPT,
        "grading_history": [],
        "batch_results": [],
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
        st.markdown("变量：`{question}` · `{standard_answer}` · `{student_answer}` · JSON 含 `fatal_errors`")
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
        st.markdown("**当前分支：语文专项批改模式（V2.0 JSON）**")
        st.markdown(
            "变量：`{rubric}` · `{student_answer}` · JSON 含 `fatal_errors` · `dimension_scores` 等"
        )
        edited = st.text_area(
            "语文模式 Prompt",
            value=st.session_state.liberal_arts_prompt,
            height=450,
            key="arts_prompt_editor",
        )
        st.session_state.liberal_arts_prompt = edited
        if st.button("🔄 恢复默认语文 Prompt", key="reset_arts"):
            st.session_state.liberal_arts_prompt = DEFAULT_LIBERAL_ARTS_PROMPT
            st.rerun()

    else:
        st.markdown("**当前分支：英语专项批改模式（V2.0 JSON）**")
        st.markdown(
            "变量：`{rubric}` · `{student_answer}` · JSON 含 `fatal_errors` · `dimension_scores` 等"
        )
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


def render_json_grading_ui(
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
    """语文 / 英语共用的 JSON 结构化批改 UI。"""
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

    pdf_text = render_pdf_uploader("📄 上传 PDF 作业（选填）", key=f"pdf_{button_key}")

    homework = st.text_area(
        homework_label,
        height=300,
        placeholder=homework_placeholder + "\n（也可仅上传 PDF，或两者同时使用）",
        key=homework_key,
    )

    if st.button(button_label, type="primary", use_container_width=True, key=button_key):
        content = merge_text_and_pdf(homework, pdf_text)
        if not content:
            st.warning("请输入作业内容或上传 PDF 文件")
        else:
            st.markdown("### 📋 批改结果")
            try:
                prompt = prompt_builder(rubric, content)
                raw = call_text_grading(model, prompt)
                render_homework_json_result(raw, student_name, subject)
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
        f"- Tab 2：批量批改 & 邮件\n"
        f"- Tab 3：学情分析\n"
        f"- Tab 4：Prompt 引擎"
    )

# ---------------------------------------------------------------------------
# 主区域 - 三标签页
# ---------------------------------------------------------------------------

tab_grading, tab_batch, tab_analytics, tab_prompt = st.tabs(
    ["📝 作业批改台", "📦 批量批改 & 邮件", "📊 学情分析", "⚙️ 提示词引擎"]
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
            question_pdf = render_pdf_uploader(
                "📄 上传题目 PDF（选填）", key="pdf_math_question"
            )
            question = st.text_area(
                "📋 题目内容",
                height=120,
                placeholder="请输入数学题目，或上传 PDF…",
                key="math_question",
            )
        with col2:
            standard_pdf = render_pdf_uploader(
                "📄 上传标准答案 PDF（选填）", key="pdf_math_standard"
            )
            standard_answer = st.text_area(
                "✅ 标准答案或参考思路",
                height=120,
                placeholder="请输入标准答案，或上传 PDF…",
                key="math_standard",
            )

        student_pdf = render_pdf_uploader(
            "📄 上传学生解答 PDF（选填）", key="pdf_math_student"
        )
        student_answer = st.text_area(
            "📝 学生的实际解答步骤",
            height=250,
            placeholder="请粘贴学生解答，或上传 PDF…",
            key="math_student",
        )

        if st.button(
            "🔍 开始执行逻辑校验",
            type="primary",
            use_container_width=True,
            key="btn_math",
        ):
            final_question = merge_text_and_pdf(question, question_pdf)
            final_standard = merge_text_and_pdf(standard_answer, standard_pdf)
            final_student = merge_text_and_pdf(student_answer, student_pdf)
            if not final_question.strip() or not final_student.strip():
                st.warning("请填写题目和学生解答（文本或 PDF 至少提供一项）")
            else:
                st.markdown("### 📋 校验结果")
                try:
                    prompt = build_math_prompt(
                        final_question.strip(),
                        final_standard.strip(),
                        final_student.strip(),
                    )
                    raw_response = call_math_grading(selected_model, prompt)
                    render_math_result(raw_response, student_name)
                except ValueError as e:
                    st.error(f"配置错误：{e}")
                except Exception as e:
                    st.error(f"API 调用失败：{e}")

    elif subject_mode == SUBJECT_ARTS:
        render_json_grading_ui(
            subject=SUBJECT_ARTS,
            title="📖 语文专项批改",
            caption="V2.0 底线核查 + 多维度 JSON 结构化输出",
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
        render_json_grading_ui(
            subject=SUBJECT_ENGLISH,
            title="🇬🇧 英语专项批改",
            caption="V2.0 底线核查 + 多维度 JSON 结构化输出",
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

with tab_batch:
    render_batch_tab(subject_mode, selected_model)

with tab_analytics:
    render_analytics_tab(subject_mode, selected_model)

with tab_prompt:
    render_prompt_engine(subject_mode)
