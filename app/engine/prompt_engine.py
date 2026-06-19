STYLE_PROMPT_TEMPLATES = {
    "formal_business": """请将以下文本改写为正式商务风格。改写要求：
1. 使用正式、规范的商务用语，避免口语化表达
2. 句式以长句为主，适当使用被动语态
3. 多用专业术语和书面词汇
4. 段落结构严谨，逻辑清晰
5. 保持原文的核心信息和语义不变

原文：{source_text}

请直接输出改写后的文本，不要添加任何解释。""",

    "colloquial": """请将以下文本改写为口语化风格。改写要求：
1. 使用轻松、自然的口语表达
2. 多用短句，语气词可以适当使用（如啊、呢、嘛、吧）
3. 避免使用专业术语和书面语
4. 表达活泼生动，像日常聊天一样
5. 保持原文的核心信息和语义不变

原文：{source_text}

请直接输出改写后的文本，不要添加任何解释。""",

    "academic": """请将以下文本改写为学术论文风格。改写要求：
1. 使用严谨、客观的学术语言
2. 长句为主，多用被动语态和复杂句式
3. 大量使用专业术语和学术词汇
4. 段落结构清晰，论证严密
5. 保持原文的核心信息和语义不变

原文：{source_text}

请直接输出改写后的文本，不要添加任何解释。""",

    "literary": """请将以下文本改写为文学散文风格。改写要求：
1. 使用优美、富有表现力的文学语言
2. 句式长短结合，注重节奏和韵律
3. 善用修辞手法（比喻、拟人、排比等）
4. 意境深远，情感丰富
5. 保持原文的核心信息和语义不变

原文：{source_text}

请直接输出改写后的文本，不要添加任何解释。""",

    "humorous": """请将以下文本改写为幽默诙谐风格。改写要求：
1. 使用风趣、诙谐的表达方式
2. 短句为主，节奏明快
3. 适当使用夸张、反讽等幽默手法
4. 语言活泼有趣，让人忍俊不禁
5. 保持原文的核心信息和语义不变

原文：{source_text}

请直接输出改写后的文本，不要添加任何解释。""",

    "concise": """请将以下文本改写为极简精炼风格。改写要求：
1. 用最少的文字表达最核心的信息
2. 删除所有冗余和修饰性内容
3. 短句为主，每句话都有实质内容
4. 结构紧凑，逻辑清晰
5. 保持原文的核心信息和语义不变

原文：{source_text}

请直接输出改写后的文本，不要添加任何解释。""",
}


def build_prompt(source_text, target_style_key, style_name=None, custom_description=None):
    if target_style_key in STYLE_PROMPT_TEMPLATES:
        template = STYLE_PROMPT_TEMPLATES[target_style_key]
        return template.format(source_text=source_text)
    if custom_description:
        prompt = f"""请将以下文本改写为"{style_name or target_style_key}"风格。改写要求：
{custom_description}
5. 保持原文的核心信息和语义不变

原文：{{source_text}}

请直接输出改写后的文本，不要添加任何解释。"""
        return prompt.format(source_text=source_text)
    return f"""请将以下文本改写为"{style_name or target_style_key}"风格，保持原文核心语义不变。

原文：{source_text}

请直接输出改写后的文本，不要添加任何解释。"""


def call_llm_mock(prompt):
    import re
    source_match = re.search(r'原文[：:]\s*(.+?)(?:\n|$)', prompt, re.DOTALL)
    source_text = source_match.group(1).strip() if source_match else "未知原文"

    if "正式商务" in prompt or "商务" in prompt:
        processed = source_text
        processed = processed.replace("今天", "本日")
        processed = processed.replace("我们", "本司")
        processed = processed.replace("大家", "各位同仁")
        processed = processed.replace("说说", "发表意见")
        processed = processed.replace("想法", "建议")
        processed = processed.replace("讨论", "研讨")
        processed = processed.replace("公司", "本公司")
        return f"根据会议议程安排，现就有关事项进行如下说明：{processed}。上述事宜，请各位认真审议并提出宝贵意见。"
    elif "口语化" in prompt:
        processed = source_text
        processed = processed.replace("根据", "说起")
        processed = processed.replace("应当", "得")
        processed = processed.replace("执行", "照着做")
        processed = processed.replace("本企业", "咱们公司")
        processed = processed.replace("企业", "咱们公司")
        processed = processed.replace("要求", "规矩")
        return f"嗨，跟你说个事儿啊，{processed}！这道理很简单嘛，大家心里都明白的对吧～"
    elif "学术" in prompt:
        processed = source_text
        processed = processed.replace("很快", "呈指数级增长态势")
        processed = processed.replace("很多", "众多")
        processed = processed.replace("公司", "机构")
        processed = processed.replace("用", "予以采用")
        processed = processed.replace("不错", "表现出良好的应用前景")
        processed = processed.replace("值得", "具有重要的")
        return f"近年来，相关研究表明，{processed}。综上所述，该领域的研究成果对于推动相关理论发展具有重要的学术价值与实践意义。"
    elif "文学" in prompt or "散文" in prompt:
        return f"时光悠悠，岁月如歌，{source_text}。这一切如诗如画，在生命的长河中泛起层层涟漪，留下了难以磨灭的印记。"
    elif "幽默" in prompt or "诙谐" in prompt:
        processed = source_text
        processed = processed.replace("发展", "折腾")
        processed = processed.replace("技术", "黑科技")
        processed = processed.replace("研究", "琢磨")
        return f"话说{processed}，这事儿说起来也是挺有意思的哈！不过话说回来，生活不就是这样嘛，笑一笑十年少~"
    elif "极简" in prompt or "精炼" in prompt:
        sents = re.split(r'[，。！？；]', source_text)
        sents = [s.strip() for s in sents if s.strip()]
        core = sents[0] if sents else source_text
        if len(core) > 20:
            core = core[:20] + "..."
        return f"{core}。"
    else:
        return f"[风格改写] {source_text}"


def migrate_prompt_based(source_text, target_style_key, style_name=None, custom_description=None):
    prompt = build_prompt(source_text, target_style_key, style_name, custom_description)
    result = call_llm_mock(prompt)
    return result
