from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import init_db
from app.core.config import PRESET_STYLES
from app.models.models import Style
from app.core.database import SessionLocal
from app.api.routes import router


app = FastAPI(title="文本风格迁移与改写质量评估系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


def _init_preset_styles():
    db = SessionLocal()
    try:
        for key, info in PRESET_STYLES.items():
            existing = db.query(Style).filter(Style.key == key).first()
            if not existing:
                style = Style(
                    key=key,
                    name=info["name"],
                    description=info["description"],
                    is_preset=True,
                )
                style.set_features(info["features"])
                db.add(style)
        db.commit()
    finally:
        db.close()


@app.on_event("startup")
def startup():
    init_db()
    _init_preset_styles()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
