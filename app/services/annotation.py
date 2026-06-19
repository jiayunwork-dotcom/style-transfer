import csv
import io
import json
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.models import AnnotationTask, Annotation, MigrationResult


def create_annotation_task(db: Session, name, description, result_ids, assignees):
    task = AnnotationTask(
        name=name,
        description=description,
        status="pending",
    )
    task.set_result_ids(result_ids)
    task.set_assignees(assignees)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_annotation_task(db: Session, task_id):
    return db.query(AnnotationTask).filter(AnnotationTask.id == task_id).first()


def list_annotation_tasks(db: Session):
    return db.query(AnnotationTask).order_by(AnnotationTask.created_at.desc()).all()


def add_annotation(db: Session, task_id, migration_result_id, annotator, content_score, style_score, fluency_score, note=""):
    existing = db.query(Annotation).filter(
        Annotation.annotation_task_id == task_id,
        Annotation.migration_result_id == migration_result_id,
        Annotation.annotator == annotator,
    ).first()
    if existing:
        existing.content_score = content_score
        existing.style_score = style_score
        existing.fluency_score = fluency_score
        existing.note = note
        existing.created_at = datetime.utcnow()
        db.commit()
        return existing

    annotation = Annotation(
        annotation_task_id=task_id,
        migration_result_id=migration_result_id,
        annotator=annotator,
        content_score=content_score,
        style_score=style_score,
        fluency_score=fluency_score,
        note=note,
    )
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation


def get_task_annotations(db: Session, task_id):
    return db.query(Annotation).filter(Annotation.annotation_task_id == task_id).all()


def get_task_progress(db: Session, task_id):
    task = get_annotation_task(db, task_id)
    if not task:
        return None
    result_ids = task.get_result_ids()
    assignees = task.get_assignees()
    total = len(result_ids) * len(assignees)
    annotations = db.query(Annotation).filter(Annotation.annotation_task_id == task_id).all()
    completed = len(annotations)
    return {
        "task_id": task_id,
        "total": total,
        "completed": completed,
        "uncompleted": total - completed,
        "progress": round(completed / total, 4) if total > 0 else 0.0,
    }


def compute_cohen_kappa(db: Session, task_id, result_id, annotator_a, annotator_b):
    ann_a = db.query(Annotation).filter(
        Annotation.annotation_task_id == task_id,
        Annotation.migration_result_id == result_id,
        Annotation.annotator == annotator_a,
    ).first()
    ann_b = db.query(Annotation).filter(
        Annotation.annotation_task_id == task_id,
        Annotation.migration_result_id == result_id,
        Annotation.annotator == annotator_b,
    ).first()
    if not ann_a or not ann_b:
        return None

    scores_a = [ann_a.content_score, ann_a.style_score, ann_a.fluency_score]
    scores_b = [ann_b.content_score, ann_b.style_score, ann_b.fluency_score]
    return _kappa_for_scores(scores_a, scores_b)


def compute_cohen_kappa_task(db: Session, task_id, annotator_a, annotator_b):
    annotations_a = db.query(Annotation).filter(
        Annotation.annotation_task_id == task_id,
        Annotation.annotator == annotator_a,
    ).all()
    annotations_b = db.query(Annotation).filter(
        Annotation.annotation_task_id == task_id,
        Annotation.annotator == annotator_b,
    ).all()

    map_b = {ann.migration_result_id: ann for ann in annotations_b}
    all_scores_a = []
    all_scores_b = []
    for ann_a in annotations_a:
        ann_b = map_b.get(ann_a.migration_result_id)
        if ann_b is None:
            continue
        all_scores_a.extend([ann_a.content_score, ann_a.style_score, ann_a.fluency_score])
        all_scores_b.extend([ann_b.content_score, ann_b.style_score, ann_b.fluency_score])

    if len(all_scores_a) < 2:
        return None
    return _kappa_for_scores(all_scores_a, all_scores_b)


def _kappa_for_scores(scores_a, scores_b):
    n = len(scores_a)
    if n == 0:
        return None

    categories = sorted(set(scores_a + scores_b))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    observed = [[0] * k for _ in range(k)]
    for a, b in zip(scores_a, scores_b):
        observed[cat_to_idx[a]][cat_to_idx[b]] += 1

    total = sum(sum(row) for row in observed)
    if total == 0:
        return None

    p_observed = sum(observed[i][i] for i in range(k)) / total

    row_sums = [sum(observed[i]) for i in range(k)]
    col_sums = [sum(observed[j][i] for j in range(k)) for i in range(k)]
    p_expected = sum((row_sums[i] * col_sums[i]) / (total * total) for i in range(k))

    if p_expected == 1.0:
        return 1.0
    kappa = (p_observed - p_expected) / (1 - p_expected)
    return round(kappa, 4)


def export_annotations_csv(db: Session, task_id):
    task = get_annotation_task(db, task_id)
    if not task:
        return ""
    annotations = get_task_annotations(db, task_id)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["annotation_id", "result_id", "annotator", "content_score", "style_score", "fluency_score", "note", "created_at"])
    for ann in annotations:
        writer.writerow([
            ann.id, ann.migration_result_id, ann.annotator,
            ann.content_score, ann.style_score, ann.fluency_score,
            ann.note or "", ann.created_at.isoformat() if ann.created_at else ""
        ])
    return output.getvalue()
