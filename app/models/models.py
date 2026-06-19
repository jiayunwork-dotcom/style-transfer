import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base


class Style(Base):
    __tablename__ = "styles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), unique=True, nullable=False)
    name = Column(String(64), nullable=False)
    description = Column(Text, default="")
    is_preset = Column(Boolean, default=False)
    features_json = Column(Text, default="{}")
    fingerprint_json = Column(Text, default="{}")
    example_texts_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)

    def get_features(self):
        return json.loads(self.features_json) if self.features_json else {}

    def set_features(self, val):
        self.features_json = json.dumps(val, ensure_ascii=False)

    def get_fingerprint(self):
        return json.loads(self.fingerprint_json) if self.fingerprint_json else {}

    def set_fingerprint(self, val):
        self.fingerprint_json = json.dumps(val, ensure_ascii=False)

    def get_example_texts(self):
        return json.loads(self.example_texts_json) if self.example_texts_json else []

    def set_example_texts(self, val):
        self.example_texts_json = json.dumps(val, ensure_ascii=False)


class MigrationResult(Base):
    __tablename__ = "migration_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_text = Column(Text, nullable=False)
    target_style_key = Column(String(64), nullable=False)
    migration_method = Column(String(32), nullable=False)
    result_text = Column(Text, default="")
    content_score = Column(Float, default=0.0)
    style_score = Column(Float, default=0.0)
    fluency_score = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)
    batch_task_id = Column(Integer, ForeignKey("batch_tasks.id"), nullable=True)
    ab_task_id = Column(Integer, ForeignKey("ab_tasks.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    annotations = relationship("Annotation", back_populates="migration_result")
    batch_task = relationship("BatchTask", back_populates="results")
    ab_task = relationship("ABTask", back_populates="results")


class BatchTask(Base):
    __tablename__ = "batch_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String(32), default="pending")
    total_count = Column(Integer, default=0)
    completed_count = Column(Integer, default=0)
    target_style_key = Column(String(64), nullable=False)
    migration_method = Column(String(32), nullable=False)
    texts_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)

    results = relationship("MigrationResult", back_populates="batch_task")


class AnnotationTask(Base):
    __tablename__ = "annotation_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, default="")
    status = Column(String(32), default="pending")
    result_ids_json = Column(Text, default="[]")
    assignees_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)

    annotations = relationship("Annotation", back_populates="annotation_task")

    def get_result_ids(self):
        return json.loads(self.result_ids_json) if self.result_ids_json else []

    def set_result_ids(self, val):
        self.result_ids_json = json.dumps(val, ensure_ascii=False)

    def get_assignees(self):
        return json.loads(self.assignees_json) if self.assignees_json else []

    def set_assignees(self, val):
        self.assignees_json = json.dumps(val, ensure_ascii=False)


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    annotation_task_id = Column(Integer, ForeignKey("annotation_tasks.id"), nullable=False)
    migration_result_id = Column(Integer, ForeignKey("migration_results.id"), nullable=False)
    annotator = Column(String(64), nullable=False)
    content_score = Column(Integer, default=0)
    style_score = Column(Integer, default=0)
    fluency_score = Column(Integer, default=0)
    note = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    annotation_task = relationship("AnnotationTask", back_populates="annotations")
    migration_result = relationship("MigrationResult", back_populates="annotations")


class MixedStyle(Base):
    __tablename__ = "mixed_styles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), unique=True, nullable=False)
    name = Column(String(64), nullable=False)
    description = Column(Text, default="")
    source_style_a_key = Column(String(64), nullable=False)
    source_style_b_key = Column(String(64), nullable=False)
    mix_ratio_a = Column(Float, nullable=False)
    mix_ratio_b = Column(Float, nullable=False)
    features_json = Column(Text, default="{}")
    fingerprint_json = Column(Text, default="{}")
    style_type = Column(String(16), default="mixed")
    created_at = Column(DateTime, default=datetime.utcnow)

    def get_features(self):
        return json.loads(self.features_json) if self.features_json else {}

    def set_features(self, val):
        self.features_json = json.dumps(val, ensure_ascii=False)

    def get_fingerprint(self):
        return json.loads(self.fingerprint_json) if self.fingerprint_json else {}

    def set_fingerprint(self, val):
        self.fingerprint_json = json.dumps(val, ensure_ascii=False)


class ABTask(Base):
    __tablename__ = "ab_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    method_a = Column(String(32), nullable=False)
    method_b = Column(String(32), nullable=False)
    target_style_key = Column(String(64), nullable=False)
    texts_json = Column(Text, default="[]")
    status = Column(String(32), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    results = relationship("MigrationResult", back_populates="ab_task")
    preferences = relationship("ABPreference", back_populates="ab_task")


class ABPreference(Base):
    __tablename__ = "ab_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ab_task_id = Column(Integer, ForeignKey("ab_tasks.id"), nullable=False)
    annotator = Column(String(64), nullable=False)
    source_text = Column(Text, default="")
    preferred_method = Column(String(32), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    ab_task = relationship("ABTask", back_populates="preferences")
