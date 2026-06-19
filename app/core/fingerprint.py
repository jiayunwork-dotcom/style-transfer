import re
import math
from collections import Counter


def tokenize(text):
    import jieba
    return list(jieba.cut(text))


def split_sentences(text):
    parts = re.split(r'[。！？；\n]+', text)
    return [s.strip() for s in parts if s.strip()]


def extract_fingerprint(texts):
    if not texts:
        return {}
    all_sentences = []
    all_tokens = []
    all_token_count = 0
    total_chars = 0
    for text in texts:
        sents = split_sentences(text)
        all_sentences.extend(sents)
        tokens = tokenize(text)
        all_tokens.extend(tokens)
        all_token_count += len(tokens)
        total_chars += len(text.replace(" ", ""))

    if all_token_count == 0:
        return {}

    sentence_lengths = [len(s) for s in all_sentences if len(s) > 0]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    long_sentences = [l for l in sentence_lengths if l > 30]
    long_sentence_ratio = len(long_sentences) / len(sentence_lengths) if sentence_lengths else 0

    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    vocabulary_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0

    content_words = [t for t in all_tokens if len(t) > 1 and re.match(r'[\u4e00-\u9fff]+', t)]
    content_word_counts = Counter(content_words)
    top_words = content_word_counts.most_common(20)
    word_freq_dist = {w: c / len(content_words) for w, c in top_words} if content_words else {}

    punctuation_pattern = r'[，。！？；：""''【】《》、—…·]'
    punct_count = len(re.findall(punctuation_pattern, "".join(texts)))
    punctuation_density = punct_count / total_chars if total_chars > 0 else 0

    paragraph_lengths = []
    for text in texts:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paras:
            paragraph_lengths.extend([len(p) for p in paras])
    avg_paragraph_length = sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0

    from app.core.synonym_dict import FORMALITY_MAP
    formal_scores = [FORMALITY_MAP.get(w, 3) for w in content_words]
    avg_formality = sum(formal_scores) / len(formal_scores) if formal_scores else 3.0

    passive_indicators = ['被', '受', '遭', '由', '为...所']
    passive_count = sum(1 for t in all_tokens if any(ind in t for ind in passive_indicators))
    passive_voice_ratio = passive_count / len(all_sentences) if all_sentences else 0

    colloquial_words = [w for w in all_tokens if FORMALITY_MAP.get(w, 3) <= 2]
    colloquial_ratio = len(colloquial_words) / len(content_words) if content_words else 0

    return {
        "avg_sentence_length": round(avg_sentence_length, 2),
        "long_sentence_ratio": round(long_sentence_ratio, 4),
        "vocabulary_diversity": round(vocabulary_diversity, 4),
        "punctuation_density": round(punctuation_density, 4),
        "avg_paragraph_length": round(avg_paragraph_length, 2),
        "avg_formality": round(avg_formality, 2),
        "passive_voice_ratio": round(passive_voice_ratio, 4),
        "colloquial_ratio": round(colloquial_ratio, 4),
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "top_word_freq": {k: round(v, 4) for k, v in list(word_freq_dist.items())[:10]},
    }
