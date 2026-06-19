import json
import time
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import PRESET_STYLES, MAX_BATCH_SIZE
from app.core.fingerprint import extract_fingerprint
from app.models.models import Style, MigrationResult, BatchTask
from app.engine.migrator import migrate
from app.eval.evaluator import evaluate
from app.services.batch_queue import batch_queue
from app.services.annotation import (
    create_annotation_task as _create_annotation_task,
    get_annotation_task as _get_annotation_task,
    list_annotation_tasks as _list_annotation_tasks,
    add_annotation as _add_annotation,
    get_task_annotations as _get_task_annotations,
    get_task_progress as _get_task_progress,
    compute_cohen_kappa as _compute_cohen_kappa,
    compute_cohen_kappa_task as _compute_cohen_kappa_task,
    export_annotations_csv as _export_annotations_csv,
)
from app.services.ab_comparison import (
    create_ab_task as _create_ab_task,
    get_ab_task as _get_ab_task,
    list_ab_tasks as _list_ab_tasks,
    get_ab_results as _get_ab_results,
    add_preference as _add_preference,
    get_preference_stats as _get_preference_stats,
)

router = APIRouter()


class MigrateRequest(BaseModel):
    text: str
    target_style: str
    method: str = "rule"


class MigrateResponse(BaseModel):
    id: int
    source_text: str
    result_text: str
    target_style: str
    method: str
    scores: dict


class BatchSubmitRequest(BaseModel):
    texts: List[str]
    target_style: str
    method: str = "rule"


class BatchSubmitResponse(BaseModel):
    task_id: int
    total: int
    message: str


class BatchProgressResponse(BaseModel):
    task_id: int
    status: str
    total: int
    completed: int
    progress: float
    estimated_remaining_seconds: Optional[float] = None


class CreateStyleRequest(BaseModel):
    key: str
    name: str
    description: str = ""
    example_texts: List[str]


class AnnotationTaskCreateRequest(BaseModel):
    name: str
    description: str = ""
    result_ids: List[int]
    assignees: List[str]


class AnnotationCreateRequest(BaseModel):
    task_id: int
    migration_result_id: int
    annotator: str
    content_score: int
    style_score: int
    fluency_score: int
    note: str = ""


class KappaRequest(BaseModel):
    task_id: int
    result_id: Optional[int] = None
    annotator_a: str
    annotator_b: str


class ABTaskCreateRequest(BaseModel):
    name: str
    method_a: str
    method_b: str
    target_style: str
    texts: List[str]


class ABPreferenceRequest(BaseModel):
    task_id: int
    annotator: str
    source_text: str
    preferred_method: str


def _get_style_features(db, target_style_key):
    style_info = PRESET_STYLES.get(target_style_key, {})
    style_features = style_info.get("features", {})
    style_name = style_info.get("name", "")
    custom_desc = style_info.get("description", "")

    style_obj = db.query(Style).filter(Style.key == target_style_key).first()
    if style_obj:
        style_features = style_obj.get_features()
        style_name = style_obj.name
        custom_desc = style_obj.description
    return style_features, style_name, custom_desc


@router.post("/migrate", response_model=MigrateResponse)
def migrate_text(req: MigrateRequest, db: Session = Depends(get_db)):
    if req.method not in ("rule", "prompt", "hybrid"):
        raise HTTPException(status_code=400, detail="method must be one of: rule, prompt, hybrid")

    style_features, style_name, custom_desc = _get_style_features(db, req.target_style)
    if not style_features and not style_name:
        style_obj = db.query(Style).filter(Style.key == req.target_style).first()
        if not style_obj:
            raise HTTPException(status_code=404, detail=f"Style '{req.target_style}' not found")

    result_text = migrate(req.text, req.target_style, req.method, style_features, style_name, custom_desc)
    scores = evaluate(req.text, result_text, req.target_style)

    result = MigrationResult(
        source_text=req.text,
        target_style_key=req.target_style,
        migration_method=req.method,
        result_text=result_text,
        content_score=scores["content_preservation"],
        style_score=scores["style_intensity"],
        fluency_score=scores["fluency"],
        overall_score=scores["overall"],
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    return MigrateResponse(
        id=result.id,
        source_text=result.source_text,
        result_text=result.result_text,
        target_style=result.target_style_key,
        method=result.migration_method,
        scores=scores,
    )


@router.post("/batch/submit", response_model=BatchSubmitResponse)
def batch_submit(req: BatchSubmitRequest, db: Session = Depends(get_db)):
    if len(req.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Max {MAX_BATCH_SIZE} texts per batch")
    if req.method not in ("rule", "prompt", "hybrid"):
        raise HTTPException(status_code=400, detail="method must be one of: rule, prompt, hybrid")

    task = BatchTask(
        status="pending",
        total_count=len(req.texts),
        target_style_key=req.target_style,
        migration_method=req.method,
    )
    task.texts_json = json.dumps(req.texts, ensure_ascii=False)
    db.add(task)
    db.commit()
    db.refresh(task)

    def process_batch(task_id, is_cancelled):
        db_session = next(get_db())
        try:
            batch_task = db_session.query(BatchTask).filter(BatchTask.id == task_id).first()
            if not batch_task:
                return
            batch_task.status = "running"
            db_session.commit()

            texts = json.loads(batch_task.texts_json) if batch_task.texts_json else []
            style_features, style_name, custom_desc = _get_style_features(db_session, batch_task.target_style_key)

            start_time = time.time()
            for i, text in enumerate(texts):
                if is_cancelled(task_id):
                    batch_task.status = "cancelled"
                    db_session.commit()
                    return

                result_text = migrate(text, batch_task.target_style_key, batch_task.migration_method, style_features, style_name, custom_desc)
                scores = evaluate(text, result_text, batch_task.target_style_key)
                result = MigrationResult(
                    source_text=text,
                    target_style_key=batch_task.target_style_key,
                    migration_method=batch_task.migration_method,
                    result_text=result_text,
                    content_score=scores["content_preservation"],
                    style_score=scores["style_intensity"],
                    fluency_score=scores["fluency"],
                    overall_score=scores["overall"],
                    batch_task_id=task_id,
                )
                db_session.add(result)

                batch_task.completed_count = i + 1
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (len(texts) - i - 1)
                else:
                    remaining = 0
                db_session.commit()

            batch_task.status = "completed"
            db_session.commit()
        except Exception as e:
            try:
                batch_task = db_session.query(BatchTask).filter(BatchTask.id == task_id).first()
                if batch_task:
                    batch_task.status = "failed"
                    db_session.commit()
            except Exception:
                pass
        finally:
            db_session.close()

    batch_queue.submit(task.id, process_batch)

    return BatchSubmitResponse(
        task_id=task.id,
        total=task.total_count,
        message="Batch task submitted successfully",
    )


@router.get("/batch/progress/{task_id}", response_model=BatchProgressResponse)
def batch_progress(task_id: int, db: Session = Depends(get_db)):
    task = db.query(BatchTask).filter(BatchTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Batch task not found")

    progress = task.completed_count / task.total_count if task.total_count > 0 else 0.0
    remaining = None
    if task.status == "running" and task.completed_count > 0:
        elapsed = (datetime.utcnow() - task.created_at).total_seconds()
        avg_time = elapsed / task.completed_count
        remaining = avg_time * (task.total_count - task.completed_count)

    return BatchProgressResponse(
        task_id=task.id,
        status=task.status,
        total=task.total_count,
        completed=task.completed_count,
        progress=round(progress, 4),
        estimated_remaining_seconds=round(remaining, 1) if remaining else None,
    )


@router.post("/batch/cancel/{task_id}")
def batch_cancel(task_id: int, db: Session = Depends(get_db)):
    task = db.query(BatchTask).filter(BatchTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Batch task not found")
    if task.status in ("completed", "cancelled", "failed"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel task with status '{task.status}'")

    batch_queue.cancel(task_id)
    task.status = "cancelled"
    db.commit()
    return {"message": "Batch task cancelled", "task_id": task_id}


@router.get("/styles")
def list_styles(db: Session = Depends(get_db)):
    styles = []
    for key, info in PRESET_STYLES.items():
        style_obj = db.query(Style).filter(Style.key == key).first()
        if style_obj:
            styles.append({
                "key": key,
                "name": style_obj.name,
                "description": style_obj.description,
                "is_preset": True,
                "features": style_obj.get_features(),
                "fingerprint": style_obj.get_fingerprint(),
            })
        else:
            styles.append({
                "key": key,
                "name": info["name"],
                "description": info["description"],
                "is_preset": True,
                "features": info["features"],
                "fingerprint": {},
            })

    custom_styles = db.query(Style).filter(Style.is_preset == False).all()
    for s in custom_styles:
        styles.append({
            "key": s.key,
            "name": s.name,
            "description": s.description,
            "is_preset": False,
            "features": s.get_features(),
            "fingerprint": s.get_fingerprint(),
        })
    return styles


@router.post("/styles/create")
def create_style(req: CreateStyleRequest, db: Session = Depends(get_db)):
    if len(req.example_texts) < 3:
        raise HTTPException(status_code=400, detail="At least 3 example texts required")

    existing = db.query(Style).filter(Style.key == req.key).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Style key '{req.key}' already exists")

    fingerprint = extract_fingerprint(req.example_texts)

    style = Style(
        key=req.key,
        name=req.name,
        description=req.description,
        is_preset=False,
    )
    style.set_example_texts(req.example_texts)
    style.set_fingerprint(fingerprint)
    features = {
        "avg_sentence_length": fingerprint.get("avg_sentence_length", 15),
        "long_sentence_ratio": fingerprint.get("long_sentence_ratio", 0.3),
        "passive_voice_ratio": fingerprint.get("passive_voice_ratio", 0.1),
        "terminology_density": 0.1,
        "colloquial_ratio": fingerprint.get("colloquial_ratio", 0.1),
        "paragraph_structured": fingerprint.get("avg_paragraph_length", 50) > 30,
        "avg_formality": fingerprint.get("avg_formality", 3.0),
    }
    style.set_features(features)
    db.add(style)
    db.commit()
    db.refresh(style)

    return {
        "key": style.key,
        "name": style.name,
        "fingerprint": fingerprint,
        "features": features,
    }


@router.get("/history")
def get_history(limit: int = Query(20, le=100), db: Session = Depends(get_db)):
    results = db.query(MigrationResult).order_by(MigrationResult.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "source_text": r.source_text[:100] + "..." if len(r.source_text) > 100 else r.source_text,
            "result_text": r.result_text[:100] + "..." if len(r.result_text) > 100 else r.result_text,
            "target_style": r.target_style_key,
            "method": r.migration_method,
            "scores": {
                "content_preservation": r.content_score,
                "style_intensity": r.style_score,
                "fluency": r.fluency_score,
                "overall": r.overall_score,
            },
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in results
    ]


@router.get("/result/{result_id}")
def get_result(result_id: int, db: Session = Depends(get_db)):
    result = db.query(MigrationResult).filter(MigrationResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return {
        "id": result.id,
        "source_text": result.source_text,
        "result_text": result.result_text,
        "target_style": result.target_style_key,
        "method": result.migration_method,
        "scores": {
            "content_preservation": result.content_score,
            "style_intensity": result.style_score,
            "fluency": result.fluency_score,
            "overall": result.overall_score,
        },
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }


@router.post("/annotation/task")
def create_annotation_task(req: AnnotationTaskCreateRequest, db: Session = Depends(get_db)):
    for rid in req.result_ids:
        result = db.query(MigrationResult).filter(MigrationResult.id == rid).first()
        if not result:
            raise HTTPException(status_code=400, detail=f"Migration result {rid} not found")

    task = _create_annotation_task(db, req.name, req.description, req.result_ids, req.assignees)
    return {"task_id": task.id, "name": task.name, "status": task.status}


@router.get("/annotation/tasks")
def list_annotation_tasks_api(db: Session = Depends(get_db)):
    tasks = _list_annotation_tasks(db)
    result = []
    for t in tasks:
        progress = _get_task_progress(db, t.id)
        result.append({
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "status": t.status,
            "progress": progress,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        })
    return result


@router.get("/annotation/task/{task_id}")
def get_annotation_task_api(task_id: int, db: Session = Depends(get_db)):
    task = _get_annotation_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Annotation task not found")
    progress = _get_task_progress(db, task_id)
    result_ids = task.get_result_ids()
    results = []
    for rid in result_ids:
        mr = db.query(MigrationResult).filter(MigrationResult.id == rid).first()
        if mr:
            annotations = db.query(_import_annotation_model()).filter_by(annotation_task_id=task_id, migration_result_id=rid).all()
            results.append({
                "result_id": rid,
                "source_text": mr.source_text,
                "result_text": mr.result_text,
                "target_style": mr.target_style_key,
                "method": mr.migration_method,
                "annotations": [
                    {
                        "annotator": a.annotator,
                        "content_score": a.content_score,
                        "style_score": a.style_score,
                        "fluency_score": a.fluency_score,
                        "note": a.note,
                    }
                    for a in annotations
                ],
            })
    return {
        "id": task.id,
        "name": task.name,
        "description": task.description,
        "status": task.status,
        "progress": progress,
        "assignees": task.get_assignees(),
        "results": results,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }


def _import_annotation_model():
    from app.models.models import Annotation
    return Annotation


@router.post("/annotation/add")
def add_annotation_api(req: AnnotationCreateRequest, db: Session = Depends(get_db)):
    if not (1 <= req.content_score <= 5 and 1 <= req.style_score <= 5 and 1 <= req.fluency_score <= 5):
        raise HTTPException(status_code=400, detail="Scores must be between 1 and 5")
    annotation = _add_annotation(db, req.task_id, req.migration_result_id, req.annotator,
                                  req.content_score, req.style_score, req.fluency_score, req.note)
    return {"annotation_id": annotation.id, "message": "Annotation added"}


@router.post("/annotation/kappa")
def compute_kappa(req: KappaRequest, db: Session = Depends(get_db)):
    if req.result_id:
        kappa = _compute_cohen_kappa(db, req.task_id, req.result_id, req.annotator_a, req.annotator_b)
    else:
        kappa = _compute_cohen_kappa_task(db, req.task_id, req.annotator_a, req.annotator_b)
    if kappa is None:
        raise HTTPException(status_code=400, detail="Cannot compute Kappa: insufficient annotations")
    return {"kappa": kappa}


@router.get("/annotation/export/{task_id}")
def export_annotations(task_id: int, db: Session = Depends(get_db)):
    csv_content = _export_annotations_csv(db, task_id)
    if not csv_content:
        raise HTTPException(status_code=404, detail="Annotation task not found")
    return {"csv": csv_content}


@router.post("/ab/create")
def create_ab_task_api(req: ABTaskCreateRequest, db: Session = Depends(get_db)):
    if req.method_a == req.method_b:
        raise HTTPException(status_code=400, detail="method_a and method_b must be different")
    if req.method_a not in ("rule", "prompt", "hybrid") or req.method_b not in ("rule", "prompt", "hybrid"):
        raise HTTPException(status_code=400, detail="Methods must be one of: rule, prompt, hybrid")
    task = _create_ab_task(db, req.name, req.method_a, req.method_b, req.target_style, req.texts)
    return {"task_id": task.id, "name": task.name, "status": task.status}


@router.get("/ab/tasks")
def list_ab_tasks_api(db: Session = Depends(get_db)):
    tasks = _list_ab_tasks(db)
    return [
        {"id": t.id, "name": t.name, "method_a": t.method_a, "method_b": t.method_b,
         "target_style": t.target_style_key, "status": t.status,
         "created_at": t.created_at.isoformat() if t.created_at else None}
        for t in tasks
    ]


@router.get("/ab/results/{task_id}")
def get_ab_results_api(task_id: int, db: Session = Depends(get_db)):
    results = _get_ab_results(db, task_id)
    if not results:
        raise HTTPException(status_code=404, detail="A/B task not found")
    return results


@router.post("/ab/preference")
def add_ab_preference(req: ABPreferenceRequest, db: Session = Depends(get_db)):
    task = _get_ab_task(db, req.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="A/B task not found")
    pref = _add_preference(db, req.task_id, req.annotator, req.source_text, req.preferred_method)
    return {"preference_id": pref.id, "message": "Preference recorded"}


@router.get("/ab/preferences/{task_id}")
def get_ab_preferences(task_id: int, db: Session = Depends(get_db)):
    stats = _get_preference_stats(db, task_id)
    if not stats:
        raise HTTPException(status_code=404, detail="A/B task not found")
    return stats
