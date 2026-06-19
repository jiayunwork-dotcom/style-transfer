from app.engine.rule_engine import migrate_rule_based
from app.engine.prompt_engine import migrate_prompt_based


def migrate_hybrid(source_text, target_style_key, style_features, style_name=None, custom_description=None):
    intermediate = migrate_rule_based(source_text, target_style_key, style_features)
    result = migrate_prompt_based(intermediate, target_style_key, style_name, custom_description)
    return result


MIGRATION_METHODS = {
    "rule": migrate_rule_based,
    "prompt": None,
    "hybrid": None,
}


def migrate(source_text, target_style_key, method, style_features, style_name=None, custom_description=None):
    if method == "rule":
        return migrate_rule_based(source_text, target_style_key, style_features)
    elif method == "prompt":
        return migrate_prompt_based(source_text, target_style_key, style_name, custom_description)
    elif method == "hybrid":
        return migrate_hybrid(source_text, target_style_key, style_features, style_name, custom_description)
    else:
        raise ValueError(f"Unknown migration method: {method}")
