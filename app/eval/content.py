import re
import math
import jieba
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.fingerprint import tokenize, split_sentences
from app.core.synonym_dict import FORMALITY_MAP


def _compute_semantic_similarity(source_text, result_text):
    try:
        vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
        tfidf_matrix = vectorizer.fit_transform([source_text, result_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(sim)
    except Exception:
        source_tokens = set(tokenize(source_text))
        result_tokens = set(tokenize(result_text))
        if not source_tokens or not result_tokens:
            return 0.0
        intersection = source_tokens & result_tokens
        union = source_tokens | result_tokens
        return len(intersection) / len(union) if union else 0.0


def _extract_named_entities(text):
    entities = set()
    patterns = [
        r'[\u4e00-\u9fff]{2,4}(?:省|市|区|县|镇|村|州|盟|旗)',
        r'[\u4e00-\u9fff]{2,4}(?:公司|集团|协会|研究院|大学|学院|银行|基金会|组织|机构|部门|委员会)',
        r'(?:张|王|李|赵|刘|陈|杨|黄|周|吴|徐|孙|马|朱|胡|郭|何|林|罗|高|梁|郑|谢|宋|唐|韩|冯|邓|曹|彭|曾|萧|田|董|袁|潘|于|蒋|蔡|余|杜|叶|程|苏|魏|吕|丁|任|沈|姚|卢|傅|钟|姜|崔|谭|廖|范|汪|陆|金石|戴|贾|夏|魏|薛|闫|段|雷|侯|龙|史|陶|黎|贺|顾|毛|郝|龚|邵|万|钱|严|覃|武|洪|赖|莫|秦|尹|江|白|文|管|殷|施|陶|翟|安|颜|倪|严|牛|温|芦|季|俞|章|鲁|葛|伍|韦|申|尤|毕|聂|丛|焦|向|邢|路|岳|齐|沿|梅|莫|尚|庄|辛|管|祝|桂|漆|司马|欧阳|上官|诸葛|司徒|端木|百里|轩辕|皇甫|令狐|宇文)[\u4e00-\u9fff]{1,3}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        entities.update(matches)

    known_organizations = ['中国', '美国', '北京', '上海', '广州', '深圳', '腾讯', '阿里巴巴', '百度', '华为', '小米', '京东', '字节跳动', '微软', '苹果', '谷歌', '亚马逊']
    for org in known_organizations:
        if org in text:
            entities.add(org)

    return entities


def _compute_entity_preservation(source_text, result_text):
    source_entities = _extract_named_entities(source_text)
    if not source_entities:
        return 1.0
    result_entities = _extract_named_entities(result_text)
    preserved = source_entities & result_entities
    return len(preserved) / len(source_entities) if source_entities else 1.0


def _extract_spo_triples(text):
    triples = []
    sentences = split_sentences(text)
    for sent in sentences:
        tokens = tokenize(sent)
        if len(tokens) < 3:
            continue
        subject = None
        predicate = None
        obj = None
        for i, token in enumerate(tokens):
            if token in ['是', '有', '在', '会', '能', '要', '可以', '应该', '需要', '必须']:
                predicate = token
                if i > 0:
                    subject = tokens[i-1]
                if i < len(tokens) - 1:
                    obj = tokens[i+1]
                break
        if subject and predicate and obj:
            triples.append((subject, predicate, obj))
    return triples


def _compute_triple_preservation(source_text, result_text):
    source_triples = _extract_spo_triples(source_text)
    if not source_triples:
        return 1.0
    result_triples = _extract_spo_triples(result_text)
    if not result_triples:
        return 0.0
    preserved = 0
    for s_triple in source_triples:
        for r_triple in result_triples:
            if s_triple[0] in r_triple[2] or s_triple[2] in r_triple[2]:
                preserved += 1
                break
    return preserved / len(source_triples)


def compute_content_preservation(source_text, result_text):
    semantic_sim = _compute_semantic_similarity(source_text, result_text)
    entity_pres = _compute_entity_preservation(source_text, result_text)
    triple_pres = _compute_triple_preservation(source_text, result_text)
    score = 0.5 * semantic_sim + 0.3 * entity_pres + 0.2 * triple_pres
    return round(max(0.0, min(1.0, score)), 4)
