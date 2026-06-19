import re
import random
from app.core.synonym_dict import SYNONYM_MAP, FORMALITY_MAP
from app.core.fingerprint import tokenize, split_sentences


AMBIGUOUS_SINGLE_CHARS = {'走', '说', '看', '想', '做', '问', '听', '吃', '喝', '来', '去', '起', '下', '上', '过', '打', '开', '关', '拿', '放', '找', '叫', '让', '给', '把', '被', '对', '为', '以', '及', '并', '而', '且', '或', '与', '同', '从', '向', '到', '在', '于', '自', '至', '由', '凭', '沿', '顺', '逆', '经', '过', '进', '出', '入', '回', '离'}

FIXED_PATTERNS = [
    (r'怎么\S?', {'走', '说', '看', '想', '做', '办', '弄', '搞'}),
    (r'大家\S?', {'说', '想', '看', '做', '议', '谈'}),
    (r'一起\S?', {'走', '说', '看', '做', '吃', '喝'}),
    (r'可以\S?', {'说', '看', '想', '做', '用'}),
    (r'下来\S?', {'走', '说', '看', '做'}),
    (r'下一步\S?', {'走', '说', '看', '做', '计划'}),
]


def _is_ambiguous_context(token, idx, tokens):
    if len(token) == 1 and token in AMBIGUOUS_SINGLE_CHARS:
        if idx > 0:
            prev = tokens[idx - 1]
            for pattern, blocked in FIXED_PATTERNS:
                if re.match(pattern, prev + token):
                    return True
        if idx < len(tokens) - 1:
            nxt = tokens[idx + 1]
            two_char = token + nxt
            if two_char in SYNONYM_MAP or two_char in FORMALITY_MAP:
                return True
        return True
    return False


def _split_long_sentence(sentence, max_len=25):
    if len(sentence) <= max_len:
        return sentence
    clauses = re.split(r'[，、；]', sentence)
    if len(clauses) <= 1:
        mid = len(sentence) // 2
        for i in range(mid, len(sentence)):
            if sentence[i] in '，、；':
                return sentence[:i+1] + '\n' + sentence[i+1:]
        return sentence[:mid] + '，\n' + sentence[mid:]
    result = []
    current = ""
    for clause in clauses:
        if len(current) + len(clause) > max_len and current:
            result.append(current.strip('，、；'))
            current = clause
        else:
            current += clause
    if current:
        result.append(current.strip('，、；'))
    return '。'.join(result) + ('。' if not sentence.endswith('。') else '')


def _merge_short_sentences(text, min_len=10):
    sentences = split_sentences(text)
    if not sentences:
        return text
    merged = []
    buffer = ""
    for sent in sentences:
        if len(buffer) + len(sent) < min_len:
            buffer = buffer + '，' + sent if buffer else sent
        else:
            if buffer:
                merged.append(buffer)
            buffer = sent
    if buffer:
        merged.append(buffer)
    return '。'.join(merged)


def _replace_by_formality(text, target_formality):
    tokens = tokenize(text)
    result_tokens = []
    for idx, token in enumerate(tokens):
        current_score = FORMALITY_MAP.get(token, None)
        if current_score is None:
            result_tokens.append(token)
            continue

        if _is_ambiguous_context(token, idx, tokens):
            result_tokens.append(token)
            continue

        synonyms = SYNONYM_MAP.get(token, [])
        if not synonyms:
            result_tokens.append(token)
            continue

        diff_current = abs(current_score - target_formality)
        best_syn = token
        best_diff = diff_current
        for syn in synonyms:
            if len(syn) == 1 and syn in AMBIGUOUS_SINGLE_CHARS:
                continue
            if len(syn) < len(token):
                continue
            syn_score = FORMALITY_MAP.get(syn, 3)
            syn_diff = abs(syn_score - target_formality)
            if syn_diff < best_diff:
                best_diff = syn_diff
                best_syn = syn
        result_tokens.append(best_syn)
    return ''.join(result_tokens)


def _adjust_voice_ratio(text, target_passive_ratio):
    sentences = split_sentences(text)
    if not sentences:
        return text
    total = len(sentences)
    passive_indicators = ['被', '受', '遭']
    passive_count = sum(1 for s in sentences if any(ind in s for ind in passive_indicators))
    current_ratio = passive_count / total if total > 0 else 0

    if current_ratio < target_passive_ratio:
        active_sentences = [i for i, s in enumerate(sentences) if '把' in s and '被' not in s]
        need_convert = min(len(active_sentences), int((target_passive_ratio - current_ratio) * total))
        for idx in active_sentences[:need_convert]:
            s = sentences[idx]
            new_s = s.replace('把', '被')
            if new_s != s:
                sentences[idx] = new_s
    elif current_ratio > target_passive_ratio:
        passive_sentences = [i for i, s in enumerate(sentences) if '被' in s and '把' not in s]
        need_convert = min(len(passive_sentences), int((current_ratio - target_passive_ratio) * total))
        for idx in passive_sentences[:need_convert]:
            s = sentences[idx]
            parts = s.split('被')
            if len(parts) == 2:
                sentences[idx] = parts[1].replace(parts[0], '') + '把' + parts[0] + s[s.index('被')+1:]

    return '。'.join(sentences)


def _add_colloquial_markers(text):
    markers = ['啊', '呢', '嘛', '吧', '呀', '哈', '哦', '哎']
    sentences = split_sentences(text)
    result = []
    for s in sentences:
        if s and s[-1] not in markers and s[-1] not in '。！？；':
            s += random.choice(markers)
        result.append(s)
    return '。'.join(result)


def _remove_colloquial_markers(text):
    colloquial_endings = ['啊', '呢', '嘛', '吧', '呀', '哈', '哦', '哎', '哟', '哇', '嘿']
    sentences = split_sentences(text)
    result = []
    for s in sentences:
        while s and s[-1] in colloquial_endings:
            s = s[:-1]
        result.append(s)
    return '。'.join(result)


def migrate_rule_based(source_text, target_style_key, style_features):
    features = style_features
    target_formality = features.get("avg_formality", 3.0)
    target_long_ratio = features.get("long_sentence_ratio", 0.3)
    target_passive_ratio = features.get("passive_voice_ratio", 0.1)
    target_colloquial = features.get("colloquial_ratio", 0.1)
    is_structured = features.get("paragraph_structured", True)

    text = source_text

    text = _replace_by_formality(text, target_formality)

    sentences = split_sentences(text)
    avg_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0

    if target_long_ratio < 0.3 and avg_len > 20:
        text = _split_long_sentence(text, max_len=20)
    elif target_long_ratio > 0.5 and avg_len < 25:
        text = _merge_short_sentences(text, min_len=25)

    text = _adjust_voice_ratio(text, target_passive_ratio)

    if target_colloquial > 0.3:
        text = _add_colloquial_markers(text)
    elif target_colloquial < 0.1:
        text = _remove_colloquial_markers(text)

    if is_structured and '\n\n' not in text:
        sents = split_sentences(text)
        chunk_size = max(2, len(sents) // 3)
        paragraphs = []
        for i in range(0, len(sents), chunk_size):
            paragraphs.append('。'.join(sents[i:i+chunk_size]))
        text = '\n\n'.join(paragraphs)

    if text and text[-1] not in '。！？；':
        text += '。'

    return text
