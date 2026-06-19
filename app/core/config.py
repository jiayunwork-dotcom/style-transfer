import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "style_transfer.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

DEFAULT_WEIGHTS = {
    "content_preservation": 0.4,
    "style_intensity": 0.35,
    "fluency": 0.25,
}

MAX_BATCH_SIZE = 100
MAX_CONCURRENT_BATCH = 3

PRESET_STYLES = {
    "formal_business": {
        "name": "正式商务",
        "description": "正式商务风格，适用于商务信函、报告等场景",
        "features": {
            "long_sentence_ratio": 0.6,
            "passive_voice_ratio": 0.4,
            "terminology_density": 0.3,
            "colloquial_ratio": 0.0,
            "paragraph_structured": True,
        },
    },
    "colloquial": {
        "name": "口语化",
        "description": "口语化风格，适用于日常交流、社交媒体等场景",
        "features": {
            "long_sentence_ratio": 0.2,
            "passive_voice_ratio": 0.05,
            "terminology_density": 0.05,
            "colloquial_ratio": 0.5,
            "paragraph_structured": False,
        },
    },
    "academic": {
        "name": "学术论文",
        "description": "学术论文风格，适用于论文、研究报告等场景",
        "features": {
            "long_sentence_ratio": 0.7,
            "passive_voice_ratio": 0.5,
            "terminology_density": 0.4,
            "colloquial_ratio": 0.0,
            "paragraph_structured": True,
        },
    },
    "humorous": {
        "name": "幽默诙谐",
        "description": "幽默诙谐风格，适用于轻松娱乐、段子等场景",
        "features": {
            "long_sentence_ratio": 0.25,
            "passive_voice_ratio": 0.05,
            "terminology_density": 0.05,
            "colloquial_ratio": 0.6,
            "paragraph_structured": False,
        },
    },
}
