from app.eval.content import compute_content_preservation
from app.eval.style_intensity import compute_style_intensity
from app.eval.fluency import compute_fluency
from app.core.config import DEFAULT_WEIGHTS


def evaluate(source_text, result_text, target_style_key, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    content_score = compute_content_preservation(source_text, result_text)
    style_score = compute_style_intensity(result_text, target_style_key)
    fluency_score = compute_fluency(result_text)

    overall = (
        weights.get("content_preservation", 0.4) * content_score
        + weights.get("style_intensity", 0.35) * style_score
        + weights.get("fluency", 0.25) * fluency_score
    )
    overall = round(max(0.0, min(1.0, overall)), 4)

    return {
        "content_preservation": content_score,
        "style_intensity": style_score,
        "fluency": fluency_score,
        "overall": overall,
    }
