from typing import Optional

from transformers import pipeline

from config import DEFAULT_CONFIG, resolve_device

_translation_pipeline = None


def load_translation_model(device: Optional[str] = None):
    global _translation_pipeline
    if _translation_pipeline is None:
        if device is None:
            device = resolve_device(DEFAULT_CONFIG.device)
        _translation_pipeline = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-fr",
            device=0 if device == "cuda" else -1,
        )
    return _translation_pipeline


def translate_and_enrich_caption(raw_caption: str) -> str:
    if not raw_caption:
        return ""

    try:
        translator = load_translation_model()
        translated_result = translator(raw_caption)
        translated_text = (
            translated_result[0]["translation_text"]
            if isinstance(translated_result, list)
            else translated_result.get("translation_text", raw_caption)
        )
    except Exception as e:
        print(f"  ⚠ Erreur de traduction, utilisation de la légende brute: {e}")
        translated_text = raw_caption

    caption_lower = raw_caption.lower()

    description_parts = []

    has_leaves = any(w in caption_lower for w in ["leaf", "leaves", "feuille"])
    has_spots = any(w in caption_lower for w in ["spot", "spots", "tache", "taches"])
    has_yellow = any(w in caption_lower for w in ["yellow", "jaune"])
    has_white = any(w in caption_lower for w in ["white", "blanc"])
    has_green = any(w in caption_lower for w in ["green", "vert"])
    has_damage = any(w in caption_lower for w in ["damage", "damaged", "dommage"])
    has_grapes = any(w in caption_lower for w in ["grape", "grapes", "raisin"])

    if has_leaves:
        leaf_desc = "Les feuilles présentent"
        leaf_details = []

        if has_spots:
            if has_yellow:
                leaf_details.append("des taches jaunâtres")
            if has_white:
                leaf_details.append("des plages blanchâtres")
            if not (has_yellow or has_white):
                leaf_details.append("des taches de coloration anormale")

        if has_green and not has_spots:
            leaf_details.append("une coloration verte globalement homogène")

        if has_damage:
            leaf_details.append("des signes de dégradation du limbe foliaire")

        if leaf_details:
            description_parts.append(f"{leaf_desc} {', '.join(leaf_details)}")
        else:
            description_parts.append(
                f"{leaf_desc} des altérations visuelles non spécifiques"
            )

    if has_grapes:
        if has_spots:
            description_parts.append(
                "Les grappes et baies montrent des altérations de surface (taches ou différences de coloration)."
            )
        else:
            description_parts.append(
                "Les grappes et baies sont visibles mais sans anomalies clairement décrites dans la légende."
            )

    if description_parts:
        core_desc = " ".join(description_parts)
        return (
            f"{core_desc} La description est déduite de l'observation suivante de la scène : "
            f"« {translated_text} ». Une observation directe sur le terrain serait nécessaire "
            f"pour confirmer et préciser la nature exacte des altérations."
        )
    else:
        return (
            "Les symptômes visuels ne sont pas clairement détaillés dans la légende brute. "
            f"Observation textuelle disponible : « {translated_text} ». "
            "Une analyse directe des plants de vigne serait nécessaire pour caractériser les éventuelles anomalies."
        )


def refine_caption_to_symptoms(
    raw_caption: str,
    rewriter=None,
    max_new_tokens: int = 256,
) -> str:
    if not raw_caption:
        return ""

    return translate_and_enrich_caption(raw_caption)

