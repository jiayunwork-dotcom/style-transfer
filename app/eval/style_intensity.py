import re
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from app.core.fingerprint import tokenize, split_sentences
from app.core.synonym_dict import FORMALITY_MAP
from app.core.config import PRESET_STYLES


_style_classifier = None
_style_keys = None


def _extract_style_features(text):
    sentences = split_sentences(text)
    tokens = tokenize(text)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return [0] * 10

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
        ],
        "literary": [
            "落日余晖洒落在古老的城墙上，如同岁月的手轻轻抚摸着历史的皱纹。",
            "那些远去的时光，像一片片飘落的梧桐叶，在记忆的河流中缓缓流淌。",
            "月光如水般倾泻而下，将整个世界都笼罩在一片银白色的柔光之中。",
            "山间的溪流蜿蜒而下，在石缝间低声吟唱着亘古不变的歌谣。",
            "春风拂过田野，麦浪翻涌如金色的海洋，远处的村庄炊烟袅袅。",
            "那些曾经刻骨铭心的往事，如今都化作了指尖流逝的细沙。",
            "清晨的露珠挂在草叶上，映照出整个世界的倒影，美得不真实。",
            "黄昏时分，夕阳将天边染成了一片绚烂的橘红，如同一幅泼墨山水画。",
            "雨后的空气清新而湿润，泥土的芬芳弥漫在每一条幽静的小巷。",
            "岁月静好，不过是因为有人在替你负重前行，而你却浑然不觉。",
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
        ],
        "concise": [
            "审计已完成。报告已提交。",
            "市场复杂。需重评投资策略。",
            "架构调整，即日执行。",
            "严格质量管理。确保达标。",
            "方案获批。正式执行。",
            "加大研发。推进创新。",
            "合同约定：十五日内付款。",
            "营收增百分之十五。利润稳升。",
            "通知即日生效。遵照执行。",
            "双方达成一致。已签协议。",
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
    global _style_classifier, _style_keys
    if _style_classifier is not None:
        return
    X, y = _generate_training_data()
    _style_keys = sorted(list(set(y)))
    y_encoded = np.array([_style_keys.index(label) for label in y])
    _style_classifier = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    _style_classifier.fit(X, y_encoded)


def compute_style_intensity(text, target_style_key):
    _ensure_classifier()
    features = _extract_style_features(text)
    features_array = np.array([features])
    proba = _style_classifier.predict_proba(features_array)[0]
    if target_style_key in _style_keys:
        target_idx = _style_keys.index(target_style_key)
        score = float(proba[target_idx])
    else:
        fingerprint_features = _extract_style_features(text)
        score = _compute_custom_style_similarity(fingerprint_features, target_style_key)
    return round(max(0.0, min(1.0, score)), 4)


def _compute_custom_style_similarity(features, style_key):
    return 0.5
