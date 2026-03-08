"""
Batch 模式 VLM 提示模板。

System prompt 指示 VLM 返回 JSON 数组，User prompt 列出物体和问题。
"""


BATCH_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of spatial questions.

Question types:
1. QRR (distance comparison): Compare the 3D distance between two pairs of objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
For QRR: answer is a string "<", "~=", or ">".
For TRR: answer is an integer 1-12.

Example:
[{"qid": "qrr_0001", "answer": "<"}, {"qid": "trr_0001", "answer": 7}]"""


MULTI_VIEW_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene from multiple viewpoints.
You will receive {n_views} images of the same scene taken from different camera angles,
followed by a list of objects visible in the scene and a set of spatial questions.

Analyze ALL provided views to determine spatial relationships more accurately.

Question types:
1. QRR (distance comparison): Compare the 3D distance between two pairs of objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
For QRR: answer is a string "<", "~=", or ">".
For TRR: answer is an integer 1-12.

Example:
[{{"qid": "qrr_0001", "answer": "<"}}, {{"qid": "trr_0001", "answer": 7}}]"""


def format_batch_user_prompt(objects: list, questions: list) -> str:
    """
    构造 batch 模式的 user prompt。

    列出场景中的物体描述，然后逐条列出本 batch 的问题。
    QRR 问题用自然语言描述距离比较，TRR 问题用钟面方向描述。
    """
    lines = ["Objects in the image:"]
    for obj in objects:
        lines.append(f"  - {obj['id']}: {obj['desc']}")
    lines.append("")
    lines.append("Questions:")

    for q in questions:
        if q["type"] == "qrr":
            p1a, p1b = q["pair1"]
            p2a, p2b = q["pair2"]
            lines.append(
                f"[{q['qid']}] Compare the distance between {p1a} and {p1b} "
                f"vs the distance between {p2a} and {p2b}. "
                f"Answer: < / ~= / >"
            )
        elif q["type"] == "trr":
            lines.append(
                f"[{q['qid']}] Standing at {q['ref1']}, facing {q['ref2']} "
                f"(12 o'clock), what clock hour (1-12) is {q['target']} at?"
            )

    return "\n".join(lines)


# ReAct 纠正提示：当解析失败或缺失过多时，追加此 prompt 要求模型重新输出
REACT_CORRECTION_PROMPT = """\
Your previous response could not be fully parsed. {n_missing} out of {n_total} answers are missing.

Missing question IDs: {missing_qids}

Please output ONLY a valid JSON array containing the answers for the missing questions.
Do NOT include any explanation, markdown fences, or extra text.
Format: [{{"qid": "...", "answer": ...}}, ...]"""
