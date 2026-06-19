import json
from sqlalchemy.orm import Session
from app.models.models import MixedStyle, Style
from app.core.config import PRESET_STYLES
from app.engine.rule_engine import migrate_rule_based
from app.eval.evaluator import evaluate


def _get_style_category(db, style_key):
    if style_key in PRESET_STYLES:
        return "preset"
    custom = db.query(Style).filter(Style.key == style_key).first()
    if custom:
        return "custom"
    mixed = db.query(MixedStyle).filter(MixedStyle.key == style_key).first()
    if mixed:
        return "mixed"
    return None


def validate_mix_ratio(db, style_a_key, style_b_key, ratio_a):
    cat_a = _get_style_category(db, style_a_key)
    cat_b = _get_style_category(db, style_b_key)
    same_category = False
    if cat_a is not None and cat_a == cat_b:
        same_category = True

    ratio_a_pct = round(ratio_a * 100)
    diff_pct = abs(ratio_a_pct - (100 - ratio_a_pct))
    if same_category and diff_pct < 20:
        corrected = 0.6
        return False, corrected, same_category, "同类风格混合需明确主次，比例差值至少为20个百分点"
    return True, ratio_a, same_category, None

INTERPOLATED_FEATURE_KEYS = [
    "avg_sentence_length",
    "long_sentence_ratio",
    "passive_voice_ratio",
    "colloquial_ratio",
    "terminology_density",
    "avg_formality",
]

INTERPOLATED_FINGERPRINT_KEYS = [
    "avg_sentence_length",
    "long_sentence_ratio",
    "vocabulary_diversity",
    "punctuation_density",
    "avg_paragraph_length",
    "avg_formality",
    "passive_voice_ratio",
    "colloquial_ratio",
]


def _get_style_features_and_fingerprint(db, style_key):
    features = {}
    fingerprint = {}
    style_info = PRESET_STYLES.get(style_key, {})
    if style_info:
        features = style_info.get("features", {})
    style_obj = db.query(Style).filter(Style.key == style_key).first()
    if style_obj:
        features = style_obj.get_features()
        fingerprint = style_obj.get_fingerprint()
    mixed_obj = db.query(MixedStyle).filter(MixedStyle.key == style_key).first()
    if mixed_obj:
        features = mixed_obj.get_features()
        fingerprint = mixed_obj.get_fingerprint()
    return features, fingerprint


def _interpolate_dicts(dict_a, dict_b, ratio_a, keys):
    result = {}
    for k in keys:
        val_a = dict_a.get(k, 0)
        val_b = dict_b.get(k, 0)
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            result[k] = round(ratio_a * val_a + (1 - ratio_a) * val_b, 4)
        elif isinstance(val_a, (int, float)):
            result[k] = round(ratio_a * val_a, 4)
        elif isinstance(val_b, (int, float)):
            result[k] = round((1 - ratio_a) * val_b, 4)
    return result


def compute_mixed_features(db, style_a_key, style_b_key, ratio_a):
    ratio_a = max(0.0, min(1.0, ratio_a))
    features_a, _ = _get_style_features_and_fingerprint(db, style_a_key)
    features_b, _ = _get_style_features_and_fingerprint(db, style_b_key)

    numeric_features = _interpolate_dicts(features_a, features_b, ratio_a, INTERPOLATED_FEATURE_KEYS)

    bool_a = features_a.get("paragraph_structured", True)
    bool_b = features_b.get("paragraph_structured", True)
    numeric_features["paragraph_structured"] = bool_a if ratio_a >= 0.5 else bool_b

    return numeric_features


def compute_mixed_fingerprint(db, style_a_key, style_b_key, ratio_a):
    ratio_a = max(0.0, min(1.0, ratio_a))
    _, fp_a = _get_style_features_and_fingerprint(db, style_a_key)
    _, fp_b = _get_style_features_and_fingerprint(db, style_b_key)

    numeric_fp = _interpolate_dicts(fp_a, fp_b, ratio_a, INTERPOLATED_FINGERPRINT_KEYS)

    total_tokens_a = fp_a.get("total_tokens", 0)
    total_tokens_b = fp_b.get("total_tokens", 0)
    numeric_fp["total_tokens"] = int(ratio_a * total_tokens_a + (1 - ratio_a) * total_tokens_b)
    unique_tokens_a = fp_a.get("unique_tokens", 0)
    unique_tokens_b = fp_b.get("unique_tokens", 0)
    numeric_fp["unique_tokens"] = int(ratio_a * unique_tokens_a + (1 - ratio_a) * unique_tokens_b)

    return numeric_fp


def create_mixed_style(db: Session, key, name, description, source_a_key, source_b_key, ratio_a):
    ratio_a = max(0.0, min(1.0, ratio_a))
    ratio_b = round(1.0 - ratio_a, 4)

    existing = db.query(MixedStyle).filter(MixedStyle.key == key).first()
    if existing:
        return None, "key already exists"
    existing_style = db.query(Style).filter(Style.key == key).first()
    if existing_style:
        return None, "key already exists in styles"

    features = compute_mixed_features(db, source_a_key, source_b_key, ratio_a)
    fingerprint = compute_mixed_fingerprint(db, source_a_key, source_b_key, ratio_a)

    desc = description
    if not desc:
        info_a = PRESET_STYLES.get(source_a_key, {})
        info_b = PRESET_STYLES.get(source_b_key, {})
        name_a = info_a.get("name", source_a_key)
        name_b = info_b.get("name", source_b_key)
        style_a_obj = db.query(Style).filter(Style.key == source_a_key).first()
        style_b_obj = db.query(Style).filter(Style.key == source_b_key).first()
        if style_a_obj:
            name_a = style_a_obj.name
        if style_b_obj:
            name_b = style_b_obj.name
        desc = f"{name_a}({ratio_a:.0%}) + {name_b}({ratio_b:.0%})的混合风格"

    mixed = MixedStyle(
        key=key,
        name=name,
        description=desc,
        source_style_a_key=source_a_key,
        source_style_b_key=source_b_key,
        mix_ratio_a=ratio_a,
        mix_ratio_b=ratio_b,
        style_type="mixed",
    )
    mixed.set_features(features)
    mixed.set_fingerprint(fingerprint)
    db.add(mixed)
    db.commit()
    db.refresh(mixed)
    return mixed, None


def get_mixed_style(db: Session, key):
    return db.query(MixedStyle).filter(MixedStyle.key == key).first()


def list_mixed_styles(db: Session):
    return db.query(MixedStyle).order_by(MixedStyle.created_at.desc()).all()


def delete_mixed_style(db: Session, key):
    mixed = db.query(MixedStyle).filter(MixedStyle.key == key).first()
    if not mixed:
        return False, "mixed style not found"
    db.delete(mixed)
    db.commit()
    return True, None


def migrate_with_mixed_style(db: Session, source_text, mixed_style_key):
    mixed = db.query(MixedStyle).filter(MixedStyle.key == mixed_style_key).first()
    if not mixed:
        return None, "mixed style not found"

    features = mixed.get_features()
    result_text = migrate_rule_based(source_text, mixed_style_key, features)
    scores = evaluate(source_text, result_text, mixed_style_key)
    return {
        "result_text": result_text,
        "scores": scores,
        "mixed_style_key": mixed.key,
        "mixed_style_name": mixed.name,
        "source_a": mixed.source_style_a_key,
        "source_b": mixed.source_style_b_key,
        "ratio_a": mixed.mix_ratio_a,
        "ratio_b": mixed.mix_ratio_b,
    }, None


def get_mixed_style_depth(db, style_key):
    if style_key in PRESET_STYLES:
        return 0
    custom = db.query(Style).filter(Style.key == style_key).first()
    if custom:
        return 0
    mixed = db.query(MixedStyle).filter(MixedStyle.key == style_key).first()
    if not mixed:
        return 0
    depth_a = get_mixed_style_depth(db, mixed.source_style_a_key)
    depth_b = get_mixed_style_depth(db, mixed.source_style_b_key)
    return 1 + max(depth_a, depth_b)


CONTRADICTION_RULES = [
    {
        "name": "formal_colloquial_conflict",
        "check": lambda f: f.get("avg_formality", 3) >= 4.0 and f.get("colloquial_ratio", 0) >= 0.3,
        "indicators": ["avg_formality", "colloquial_ratio"],
        "message": "正式度很高但口语化比例也很高，存在矛盾",
        "suggestion": "建议降低风格A占比或选择差异更大的源风格",
    },
    {
        "name": "passive_short_sentence_conflict",
        "check": lambda f: f.get("passive_voice_ratio", 0) >= 0.35 and f.get("avg_sentence_length", 15) <= 12,
        "indicators": ["passive_voice_ratio", "avg_sentence_length"],
        "message": "被动语态比例很高同时平均句长很短，不合理",
        "suggestion": "建议降低风格A占比或选择差异更大的源风格",
    },
    {
        "name": "terminology_colloquial_conflict",
        "check": lambda f: f.get("terminology_density", 0) >= 0.3 and f.get("colloquial_ratio", 0) >= 0.3,
        "indicators": ["terminology_density", "colloquial_ratio"],
        "message": "术语密度很高同时口语化比例也很高，存在矛盾",
        "suggestion": "建议调整混合比例，使术语密度与口语化程度更协调",
    },
    {
        "name": "long_sentence_unstructured_conflict",
        "check": lambda f: f.get("long_sentence_ratio", 0) >= 0.5 and not f.get("paragraph_structured", True),
        "indicators": ["long_sentence_ratio", "paragraph_structured"],
        "message": "长句比例很高但段落结构不分明，阅读体验差",
        "suggestion": "建议增加段落结构化程度或降低长句比例",
    },
]


def detect_feature_anomalies(features):
    anomalies = []
    for rule in CONTRADICTION_RULES:
        if rule["check"](features):
            anomalies.append({
                "rule_name": rule["name"],
                "indicators": rule["indicators"],
                "message": rule["message"],
                "suggestion": rule["suggestion"],
            })
    return anomalies
