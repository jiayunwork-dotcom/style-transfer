import re
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from app.core.fingerprint import tokenize, split_sentences
from app.core.synonym_dict import FORMALITY_MAP
from app.core.config import PRESET_STYLES


_style_classifier = None
_style_keys = None
_feature_scaler = None
_style_feature_centroids = {}

COLLOQUIAL_MARKERS = ['啊', '呢', '嘛', '吧', '呀', '哈', '哦', '哎', '哟', '哇', '嘿', '啦', '喽', '呗', '噻', '咯', '啵', '哪', '呐']
HUMOR_MARKERS = ['哈哈', '哈哈哈', '笑死', '逗', '搞笑', '段子', '吐槽', '黑', '666', '绝了', '牛', '离谱', '绝绝子']
FORMAL_MARKERS = ['根据', '鉴于', '经', '现', '予以', '特此', '为', '对于', '关于', '截至', '兹', '本', '贵', '该', '此', '均', '皆', '须', '应', '需', '宜']
ACADEMIC_MARKERS = ['研究', '分析', '表明', '基于', '综上所述', '结论', '假设', '验证', '显著', '统计', '数据', '实验', '模型', '方法', '理论', '文献', '综述']


def _extract_style_features(text):
    sentences = split_sentences(text)
    tokens = tokenize(text)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return [0] * 12

    sentence_lengths = [len(s) for s in sentences if s.strip()]
    avg_sent_len = np.mean(sentence_lengths) if sentence_lengths else 0
    long_sent_ratio = sum(1 for l in sentence_lengths if l > 30) / len(sentence_lengths) if sentence_lengths else 0
    short_sent_ratio = sum(1 for l in sentence_lengths if l < 10) / len(sentence_lengths) if sentence_lengths else 0

    content_words = [t for t in tokens if len(t) > 1 and re.match(r'[\u4e00-\u9fff]+', t)]
    formal_scores = [FORMALITY_MAP.get(w, 3) for w in content_words]
    avg_formality = np.mean(formal_scores) if formal_scores else 3.0

    colloquial_words = [w for w in content_words if FORMALITY_MAP.get(w, 3) <= 2]
    colloquial_ratio = len(colloquial_words) / len(content_words) if content_words else 0

    punct_pattern = r'[，。！？；：""''【】《》、—…·]'
    punct_count = len(re.findall(punct_pattern, text))
    total_chars = len(text.replace(" ", ""))
    punct_density = punct_count / total_chars if total_chars > 0 else 0

    passive_indicators = ['被', '受', '遭', '由']
    passive_count = sum(1 for t in tokens if any(ind in t for ind in passive_indicators))
    passive_ratio = passive_count / len(sentences) if sentences else 0

    unique_tokens = len(set(tokens))
    vocab_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0

    exclamation_count = text.count('！') + text.count('!')
    question_count = text.count('？') + text.count('?')
    emotional_punct_ratio = (exclamation_count + question_count) / total_chars if total_chars > 0 else 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    avg_para_len = np.mean([len(p) for p in paragraphs]) if paragraphs else total_chars

    coll_marker_count = sum(text.count(m) for m in COLLOQUIAL_MARKERS)
    coll_marker_density = coll_marker_count / total_chars if total_chars > 0 else 0

    humor_marker_count = sum(text.count(m) for m in HUMOR_MARKERS)
    humor_marker_density = humor_marker_count / total_chars if total_chars > 0 else 0

    return [
        avg_sent_len,
        long_sent_ratio,
        short_sent_ratio,
        avg_formality,
        colloquial_ratio,
        punct_density,
        passive_ratio,
        vocab_diversity,
        emotional_punct_ratio,
        avg_para_len,
        coll_marker_density,
        humor_marker_density,
    ]


def _generate_training_data():
    templates = {
        "formal_business": [
            "根据相关法律法规的规定，公司应当于年度结束后三十日内完成审计工作并提交审计报告。",
            "鉴于当前市场环境的复杂性，本企业需要重新评估投资策略并优化资源配置方案。",
            "经董事会研究决定，自即日起对公司组织架构进行调整，以适应战略发展需要。",
            "各部门应严格按照质量管理体系要求执行相关工作，确保产品质量符合标准。",
            "本项目实施方案已经专家评审组审核通过，现正式批准执行。",
            "为提升企业核心竞争力，公司拟加大研发投入，推进技术创新与产业升级。",
            "根据合同约定，甲方应于收到乙方发票后十五个工作日内支付相应款项。",
            "公司年度经营业绩报告显示，营业收入同比增长百分之十五，利润率稳步提升。",
            "本通知自发布之日起生效，请各部门认真遵照执行并及时反馈实施情况。",
            "经多方协商，双方就合作事宜达成一致意见，并签署了战略合作框架协议。",
            "特此公告：公司拟于下月召开年度股东大会，审议相关事项。",
            "对于本次重大资产重组事项，公司将严格按照监管要求履行信息披露义务。",
            "截至报告期末，本集团总资产规模较上年同期增长约百分之二十。",
            "兹任命张某为公司总经理，全面负责日常经营管理工作。",
            "该方案经多方论证，具备可行性与可操作性，建议予以采纳。",
        ],
        "colloquial": [
            "今天天儿真好啊，咱出去溜达溜达吧！",
            "这事儿你别急，慢慢来，急也急不来嘛。",
            "哎呀，我又忘带钥匙了，真是个马大哈！",
            "那家新开的火锅店可好吃了，改天咱去尝尝呗。",
            "你这话说得也太逗了吧，笑死我了哈哈！",
            "哥几个好久没聚了，这周末约一下呗？",
            "这个APP也太好用了吧，安利给你们！",
            "等下我出去买个饭，你要带啥不？",
            "你说这事儿搁谁身上不闹心啊？",
            "今儿加班到这么晚，回家得好好歇歇了。",
            "哇塞，你这新衣服也太好看了吧！哪里买的呀？",
            "啊这，我也不知道咋回事儿啊，莫名其妙的。",
            "害，这事儿没啥大不了的，别往心里去哈。",
            "咱就是说，这波操作真的是绝了呀！",
            "嘿嘿，我偷偷告诉你哦，你可别跟别人说。",
        ],
        "academic": [
            "本研究旨在探讨该领域的关键问题，通过系统性分析揭示其内在规律与机制。",
            "实验结果表明，该方法在各项指标上均取得了显著的改善，验证了假设的有效性。",
            "基于文献综述与实证分析，本文提出了一个全新的理论框架用于解释该现象。",
            "数据采集采用分层随机抽样方法，样本量为五百份，置信水平为百分之九十五。",
            "综上所述，本研究对该领域的理论发展与实践应用均具有重要的参考价值。",
            "在方法论层面，本研究采用了定量与定性相结合的混合研究方法。",
            "统计检验结果显示，自变量对因变量具有显著的正向影响效应。",
            "本研究存在一定局限性，未来研究可进一步扩大样本规模并引入更多控制变量。",
            "从历史演进的视角来看，该概念经历了从狭义到广义的理论拓展过程。",
            "研究假设已通过实证数据得到验证，证明了该模型在实践中的适用性。",
            "本文通过构建多维度评价体系，对研究对象进行了全面的量化分析。",
            "相关领域已有研究主要集中于宏观层面，而微观机制仍有待深入探讨。",
            "本研究采用控制变量法，排除了可能影响实验结果的干扰因素。",
            "回归分析结果表明，核心解释变量的系数在统计上显著为正。",
            "根据上述分析，可以得出以下结论：该理论具有较强的解释力与预测能力。",
        ],
        "humorous": [
            "上班的意义就是赚钱，但赚的钱又不够花，所以上班的意义是什么？我也想知道！",
            "人生就像一盒巧克力，你永远不知道下一颗是不是也是难吃的。",
            "我不是在摸鱼，我是在进行创造性的休息，这叫战略性充电！",
            "减肥？那是不可能的，这辈子都不可能的，肉这么可爱为什么要减它。",
            "别人的周末：逛街旅游看电影。我的周末：床和手机二选一，我全都要！",
            "领导说加班使我快乐，我心想那您倒是给我加班费让我更快乐啊！",
            "问：如何快速成为百万富翁？答：先成为千万富翁，然后创业。",
            "每天叫醒我的不是梦想，是闹钟和贫穷。",
            "今天的我你爱理不理，明天的我你还爱理不理，但我还是会来的。",
            "程序员最怕什么？不是bug，是产品经理说就改一点点。",
            "我的钱包就像洋葱，每次打开都让我泪流满面，真的太心酸了！",
            "这事儿要是成了，我当场表演倒立洗头！说到做到哈哈！",
            "我这人没啥优点，就是擅长在该努力的时候选择躺平。",
            "有人说我脸皮厚，我当场就笑了，我这么帅怎么可能脸皮厚。",
            "今天的风好大，差点把我吹走，还好我胖，稳如泰山。",
        ],
    }

    X = []
    y = []
    for style_key, texts in templates.items():
        for text in texts:
            features = _extract_style_features(text)
            X.append(features)
            y.append(style_key)
    return np.array(X), np.array(y)


def _ensure_classifier():
    global _style_classifier, _style_keys, _feature_scaler, _style_feature_centroids
    if _style_classifier is not None:
        return
    X, y = _generate_training_data()
    _style_keys = sorted(list(set(y)))

    _feature_scaler = StandardScaler()
    X_scaled = _feature_scaler.fit_transform(X)

    y_encoded = np.array([_style_keys.index(label) for label in y])
    _style_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_leaf=2)
    _style_classifier.fit(X_scaled, y_encoded)

    for style_key in _style_keys:
        mask = y == style_key
        style_features = X[mask]
        _style_feature_centroids[style_key] = np.mean(style_features, axis=0)


def _cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _rule_based_style_score(text, target_style_key):
    text_lower = text
    scores = {}

    formal_marker_count = sum(1 for m in FORMAL_MARKERS if m in text)
    coll_marker_count = sum(1 for m in COLLOQUIAL_MARKERS if text.endswith(m) or m + '。' in text or m + '！' in text or m + '？' in text)
    humor_marker_count = sum(text.count(m) for m in HUMOR_MARKERS)
    academic_marker_count = sum(1 for m in ACADEMIC_MARKERS if m in text)

    sentences = split_sentences(text)
    avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0
    total_chars = len(text)

    coll_density = coll_marker_count / max(1, len(sentences))
    formal_density = formal_marker_count / max(1, total_chars / 30)
    humor_density = humor_marker_count / max(1, total_chars / 40)
    academic_density = academic_marker_count / max(1, total_chars / 30)
    sent_len_score = min(1.0, avg_sent_len / 40)

    scores['formal_business'] = min(1.0, formal_density * 0.5 + sent_len_score * 0.3 + (1.0 - min(1.0, coll_density * 2)) * 0.2)
    scores['colloquial'] = min(1.0, coll_density * 0.45 + (1.0 - formal_density) * 0.25 + (1.0 - sent_len_score) * 0.3)
    scores['academic'] = min(1.0, academic_density * 0.4 + sent_len_score * 0.35 + formal_density * 0.25)
    scores['humorous'] = min(1.0, humor_density * 0.5 + coll_density * 0.25 + (1.0 - formal_density) * 0.25)

    return scores.get(target_style_key, 0.3)


def compute_style_intensity(text, target_style_key):
    _ensure_classifier()
    features = _extract_style_features(text)
    features_array = np.array([features])

    features_scaled = _feature_scaler.transform(features_array)
    proba = _style_classifier.predict_proba(features_scaled)[0]

    classifier_score = 0.0
    if target_style_key in _style_keys:
        target_idx = _style_keys.index(target_style_key)
        classifier_score = float(proba[target_idx])

    centroid_score = 0.0
    if target_style_key in _style_feature_centroids:
        centroid = _style_feature_centroids[target_style_key]
        similarity = _cosine_similarity(np.array(features), centroid)
        centroid_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

    rule_score = _rule_based_style_score(text, target_style_key)

    if target_style_key in _style_keys:
        final_score = 0.4 * classifier_score + 0.3 * centroid_score + 0.3 * rule_score
    else:
        fingerprint_features = _extract_style_features(text)
        custom_sim = _compute_custom_style_similarity(fingerprint_features, target_style_key)
        final_score = 0.3 * centroid_score + 0.4 * rule_score + 0.3 * custom_sim

    return round(max(0.05, min(0.98, final_score)), 4)


def _compute_custom_style_similarity(features, style_key):
    from app.core.database import get_db
    from app.models.models import Style
    try:
        db = next(get_db())
        style = db.query(Style).filter(Style.key == style_key).first()
        if style and style.features:
            import json
            target_feats_dict = json.loads(style.features)
            feature_names = [
                "avg_sent_len", "long_sentence_ratio", "short_sentence_ratio",
                "avg_formality", "colloquial_ratio", "punct_density",
                "passive_voice_ratio", "vocab_diversity", "emotional_punct_ratio",
                "avg_para_len"
            ]
            target_feats = []
            for fn in feature_names:
                if fn in target_feats_dict:
                    target_feats.append(float(target_feats_dict[fn]))
                else:
                    target_feats.append(features[feature_names.index(fn)] if feature_names.index(fn) < len(features) else 0)
            if len(target_feats) >= 8:
                v1 = np.array(features[:len(target_feats)])
                v2 = np.array(target_feats)
                sim = _cosine_similarity(v1, v2)
                return max(0.2, min(0.95, (sim + 1.0) / 2.0))
    except Exception:
        pass
    return 0.5
