import json
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.models import ABTask, ABPreference, MigrationResult
from app.engine.migrator import migrate
from app.eval.evaluator import evaluate
from app.core.config import PRESET_STYLES
from app.core.fingerprint import extract_fingerprint


def create_ab_task(db: Session, name, method_a, method_b, target_style_key, texts):
    task = ABTask(
        name=name,
        method_a=method_a,
        method_b=method_b,
        target_style_key=target_style_key,
        status="pending",
    )
    task.texts_json = json.dumps(texts, ensure_ascii=False)
    db.add(task)
    db.commit()
    db.refresh(task)

    style_info = PRESET_STYLES.get(target_style_key, {})
    style_features = style_info.get("features", {})
    custom_desc = style_info.get("description", "")

    style_obj = db.query(_import_style_model()).filter_by(key=target_style_key).first()
    if style_obj:
        style_features = style_obj.get_features()
        custom_desc = style_obj.description

    for text in texts:
        result_a_text = migrate(text, target_style_key, method_a, style_features, style_name=style_info.get("name"), custom_description=custom_desc)
        scores_a = evaluate(text, result_a_text, target_style_key)
        result_a = MigrationResult(
            source_text=text,
            target_style_key=target_style_key,
            migration_method=method_a,
            result_text=result_a_text,
            content_score=scores_a["content_preservation"],
            style_score=scores_a["style_intensity"],
            fluency_score=scores_a["fluency"],
            overall_score=scores_a["overall"],
            ab_task_id=task.id,
        )
        db.add(result_a)

        result_b_text = migrate(text, target_style_key, method_b, style_features, style_name=style_info.get("name"), custom_description=custom_desc)
        scores_b = evaluate(text, result_b_text, target_style_key)
        result_b = MigrationResult(
            source_text=text,
            target_style_key=target_style_key,
            migration_method=method_b,
            result_text=result_b_text,
            content_score=scores_b["content_preservation"],
            style_score=scores_b["style_intensity"],
            fluency_score=scores_b["fluency"],
            overall_score=scores_b["overall"],
            ab_task_id=task.id,
        )
        db.add(result_b)

    task.status = "completed"
    db.commit()
    db.refresh(task)
    return task


def _import_style_model():
    from app.models.models import Style
    return Style


def get_ab_task(db: Session, task_id):
    return db.query(ABTask).filter(ABTask.id == task_id).first()


def list_ab_tasks(db: Session):
    return db.query(ABTask).order_by(ABTask.created_at.desc()).all()


def get_ab_results(db: Session, task_id):
    task = get_ab_task(db, task_id)
    if not task:
        return None
    results = db.query(MigrationResult).filter(MigrationResult.ab_task_id == task_id).all()
    method_a_results = [r for r in results if r.migration_method == task.method_a]
    method_b_results = [r for r in results if r.migration_method == task.method_b]

    def avg_scores(results_list):
        if not results_list:
            return {"content_preservation": 0, "style_intensity": 0, "fluency": 0, "overall": 0}
        n = len(results_list)
        return {
            "content_preservation": round(sum(r.content_score for r in results_list) / n, 4),
            "style_intensity": round(sum(r.style_score for r in results_list) / n, 4),
            "fluency": round(sum(r.fluency_score for r in results_list) / n, 4),
            "overall": round(sum(r.overall_score for r in results_list) / n, 4),
        }

    avg_a = avg_scores(method_a_results)
    avg_b = avg_scores(method_b_results)
    diff = {k: round(avg_a[k] - avg_b[k], 4) for k in avg_a}

    pairs = []
    texts = json.loads(task.texts_json) if task.texts_json else []
    for i, text in enumerate(texts):
        a_result = method_a_results[i] if i < len(method_a_results) else None
        b_result = method_b_results[i] if i < len(method_b_results) else None
        pairs.append({
            "source_text": text,
            "result_a": a_result.result_text if a_result else "",
            "result_b": b_result.result_text if b_result else "",
            "scores_a": {
                "content_preservation": a_result.content_score if a_result else 0,
                "style_intensity": a_result.style_score if a_result else 0,
                "fluency": a_result.fluency_score if a_result else 0,
                "overall": a_result.overall_score if a_result else 0,
            } if a_result else {},
            "scores_b": {
                "content_preservation": b_result.content_score if b_result else 0,
                "style_intensity": b_result.style_score if b_result else 0,
                "fluency": b_result.fluency_score if b_result else 0,
                "overall": b_result.overall_score if b_result else 0,
            } if b_result else {},
        })

    return {
        "task_id": task_id,
        "method_a": task.method_a,
        "method_b": task.method_b,
        "avg_scores_a": avg_a,
        "avg_scores_b": avg_b,
        "score_diff": diff,
        "pairs": pairs,
    }


def add_preference(db: Session, task_id, annotator, source_text, preferred_method):
    pref = ABPreference(
        ab_task_id=task_id,
        annotator=annotator,
        source_text=source_text,
        preferred_method=preferred_method,
    )
    db.add(pref)
    db.commit()
    db.refresh(pref)
    return pref


def get_preferences(db: Session, task_id):
    return db.query(ABPreference).filter(ABPreference.ab_task_id == task_id).all()


def get_preference_stats(db: Session, task_id):
    task = get_ab_task(db, task_id)
    if not task:
        return None
    prefs = get_preferences(db, task_id)
    total = len(prefs)
    if total == 0:
        return {"method_a": task.method_a, "method_b": task.method_b, "total": 0, "a_count": 0, "b_count": 0, "a_rate": 0, "b_rate": 0}
    a_count = sum(1 for p in prefs if p.preferred_method == task.method_a)
    b_count = sum(1 for p in prefs if p.preferred_method == task.method_b)
    return {
        "method_a": task.method_a,
        "method_b": task.method_b,
        "total": total,
        "a_count": a_count,
        "b_count": b_count,
        "a_rate": round(a_count / total, 4),
        "b_rate": round(b_count / total, 4),
    }
