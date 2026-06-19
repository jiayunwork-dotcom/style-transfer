import re
import math
from collections import Counter
from app.core.fingerprint import tokenize, split_sentences


class NgramLanguageModel:
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self._trained = False

    def train(self, texts):
        for text in texts:
            tokens = ['<s>'] + tokenize(text) + ['</s>']
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = tuple(tokens[i:i+self.n-1])
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
        self._trained = True

    def perplexity(self, text):
        if not self._trained:
            return 50.0
        tokens = ['<s>'] + tokenize(text) + ['</s>']
        log_prob = 0.0
        count = 0
        vocab_size = max(len(self.vocab), 1)
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            context = tuple(tokens[i:i+self.n-1])
            ngram_count = self.ngram_counts.get(ngram, 0)
            context_count = self.context_counts.get(context, 0)
            prob = (ngram_count + 1) / (context_count + vocab_size)
            log_prob += math.log(prob)
            count += 1
        if count == 0:
            return 50.0
        avg_log_prob = log_prob / count
        perplexity = math.exp(-avg_log_prob)
        return perplexity


_default_lm = None


def _get_default_lm():
    global _default_lm
    if _default_lm is not None:
        return _default_lm
    _default_lm = NgramLanguageModel(n=2)
    training_texts = [
        "根据相关法律法规的规定，公司应当于年度结束后三十日内完成审计工作并提交审计报告。",
        "鉴于当前市场环境的复杂性，本企业需要重新评估投资策略并优化资源配置方案。",
        "经董事会研究决定，自即日起对公司组织架构进行调整，以适应战略发展需要。",
        "本研究旨在探讨该领域的关键问题，通过系统性分析揭示其内在规律与机制。",
        "实验结果表明，该方法在各项指标上均取得了显著的改善，验证了假设的有效性。",
        "今天天儿真好啊，咱出去溜达溜达吧！",
        "这事儿你别急，慢慢来，急也急不来嘛。",
        "落日余晖洒落在古老的城墙上，如同岁月的手轻轻抚摸着历史的皱纹。",
        "那些远去的时光，像一片片飘落的梧桐叶，在记忆的河流中缓缓流淌。",
        "上班的意义就是赚钱，但赚的钱又不够花，所以上班的意义是什么？",
        "人生就像一盒巧克力，你永远不知道下一颗是不是也是难吃的。",
        "人工智能技术在近年来取得了快速发展，深度学习模型已经在多个领域展现出卓越性能。",
        "数据驱动的决策方法正在改变传统行业的运营模式，提升了效率与准确性。",
        "春暖花开的时节，万物复苏，一片生机盎然的景象展现在眼前。",
        "科技创新是推动社会进步的重要力量，我们应当积极拥抱技术变革。",
        "教育是民族振兴的基石，培养创新人才是国家发展的战略需求。",
        "城市管理需要科学规划和精细运营，确保市民生活质量不断提升。",
        "环境保护与经济发展应当协调推进，实现可持续发展的长远目标。",
        "文化传承需要与时俱进，在保留精髓的同时融入现代元素。",
        "健康管理已成为现代生活的重要组成部分，良好的生活方式至关重要。",
    ]
    _default_lm.train(training_texts)
    return _default_lm


def _check_grammar_errors(text):
    errors = 0
    sentences = split_sentences(text)
    for sent in sentences:
        if not sent.strip():
            continue

        tokens = tokenize(sent)
        if not tokens:
            errors += 1
            continue

        has_verb = any(t in ['是', '有', '在', '会', '能', '要', '可以', '应该', '需要', '必须',
                              '做', '看', '说', '走', '跑', '吃', '喝', '写', '读', '学',
                              '工作', '学习', '研究', '分析', '发展', '建设', '管理', '优化',
                              '提升', '改进', '完善', '推动', '促进', '加强', '落实', '执行'] for t in tokens)
        has_noun = any(len(t) > 1 and re.match(r'[\u4e00-\u9fff]+', t) for t in tokens)
        if not has_verb and not has_noun and len(tokens) > 2:
            errors += 1

        consecutive_punct = re.findall(r'[，。！？；：]{2,}', sent)
        errors += len(consecutive_punct)

        if sent and sent[0] in '，。！？；：、':
            errors += 1

        if len(sent) > 100 and '，' not in sent and '、' not in sent:
            errors += 1

        unmatched_quotes = 0
        for quote_pair in [('"', '"'), ("'", "'")]:
            unmatched_quotes += abs(sent.count(quote_pair[0]) - sent.count(quote_pair[1]))
        errors += unmatched_quotes // 2

    return errors


def compute_fluency(text):
    lm = _get_default_lm()
    ppl = lm.perplexity(text)
    max_ppl = 200.0
    ppl_score = max(0.0, 1.0 - ppl / max_ppl)

    grammar_errors = _check_grammar_errors(text)
    sentences = split_sentences(text)
    num_sentences = max(len(sentences), 1)
    error_rate = grammar_errors / num_sentences
    grammar_score = max(0.0, 1.0 - error_rate)

    fluency_score = 0.6 * ppl_score + 0.4 * grammar_score
    return round(max(0.0, min(1.0, fluency_score)), 4)
