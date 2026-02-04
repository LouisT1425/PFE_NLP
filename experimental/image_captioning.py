from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import pipeline

from config import DEFAULT_CONFIG, resolve_device

_translation_pipeline = None


def load_image_captioner(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
):
    if model_name is None:
        model_name = DEFAULT_CONFIG.image_caption_model
    if device is None:
        device = resolve_device(DEFAULT_CONFIG.device)

    captioner = pipeline(
        task="image-to-text",
        model=model_name,
        device=0 if device == "cuda" else -1,
    )
    return captioner


def _load_translation_model(device: Optional[str] = None):
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


def _enrich_caption_to_symptoms(raw_caption: str, device: Optional[str] = None) -> str:
    if not raw_caption:
        return ""
    
    caption_lower = raw_caption.lower()
    
    try:
        translator = _load_translation_model(device)
        translated_result = translator(raw_caption)
        translated_text = (
            translated_result[0]['translation_text'] 
            if isinstance(translated_result, list) 
            else translated_result.get('translation_text', raw_caption)
        )
    except Exception as e:
        print(f"  ⚠ Erreur de traduction, utilisation d'une traduction basique: {e}")
        translated_text = raw_caption
    
    description_parts = []
    
    has_leaves = any(word in caption_lower for word in ["leaf", "leaves", "feuille"])
    has_spots = any(word in caption_lower for word in ["spot", "tache", "spots", "patches"])
    has_yellow = any(word in caption_lower for word in ["yellow", "jaune", "yellowish"])
    has_white = any(word in caption_lower for word in ["white", "blanc", "whitish", "pale"])
    has_brown = any(word in caption_lower for word in ["brown", "brun", "browned"])
    has_green = any(word in caption_lower for word in ["green", "vert"])
    has_damage = any(word in caption_lower for word in ["damage", "dommage", "showing", "lesion"])
    has_grapes = any(word in caption_lower for word in ["grape", "grapes", "raisin", "berry", "berries"])
    has_necrosis = any(word in caption_lower for word in ["necrosis", "nécrose", "dead", "mort"])
    has_curl = any(word in caption_lower for word in ["curl", "curling", "enroulement", "rolled"])
    has_mold = any(word in caption_lower for word in ["mold", "mildew", "mildiou", "moisissure", "fungus"])
    
    if has_leaves:
        leaf_desc = "Les feuilles présentent"
        leaf_details = []
        
        if has_spots:
            if has_yellow:
                leaf_details.append("des taches jaunâtres")
            if has_white:
                leaf_details.append("des zones blanchâtres")
            if has_brown:
                leaf_details.append("des taches brunâtres")
            if not (has_yellow or has_white or has_brown):
                leaf_details.append("des taches de coloration anormale")
        
        if has_necrosis:
            leaf_details.append("des signes de nécrose")
        
        if has_curl:
            leaf_details.append("un enroulement")
        
        if has_damage and not has_necrosis:
            leaf_details.append("des signes de dégradation du limbe foliaire")
        
        if has_mold:
            leaf_details.append("des dépôts poudreux ou feutrage visible")
        
        if leaf_details:
            description_parts.append(f"{leaf_desc} {', '.join(leaf_details)}")
        elif has_green:
            description_parts.append(f"{leaf_desc} une coloration verte globalement homogène")
        else:
            description_parts.append(f"{leaf_desc} des altérations visibles")
    
    if has_grapes:
        grape_desc = "Les grappes et baies sont observables"
        grape_details = []
        if has_spots:
            grape_details.append("avec des altérations de surface visibles")
        if has_damage:
            grape_details.append("présentant des signes de dégradation")
        if grape_details:
            description_parts.append(f"{grape_desc}, {', '.join(grape_details)}")
        else:
            description_parts.append(grape_desc)
    
    if description_parts:
        core_desc = ". ".join(description_parts)
        return (
            f"{core_desc}. "
            f"Cette description est déduite de l'observation suivante de la scène : « {translated_text} ». "
            f"Une observation directe sur le terrain serait nécessaire pour confirmer et préciser la nature exacte des altérations."
        )
    else:
        return (
            f"Description visuelle basée sur l'observation suivante de la scène : « {translated_text} ». "
            f"Une analyse directe des plants de vigne serait nécessaire pour caractériser les éventuelles anomalies."
        )


def generate_symptom_description(
    image_path: Path,
    captioner=None,
    max_new_tokens: int = 64,
    device: Optional[str] = None,
) -> Optional[str]:
    if captioner is None:
        captioner = load_image_captioner(device=device)
    
    if device is None:
        device = resolve_device(DEFAULT_CONFIG.device)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"✗ Erreur lors du chargement de l'image {image_path}: {e}")
        return None

    try:
        outputs = captioner(image, max_new_tokens=max_new_tokens)
        if not outputs:
            return None
        text = outputs[0].get("generated_text") or outputs[0].get("caption")
        if text is None:
            return None
        raw_caption = text.strip()
        
        return _enrich_caption_to_symptoms(raw_caption, device=device)
        
    except Exception as e:
        print(f"✗ Erreur lors du captioning de {image_path.name}: {e}")
        return None

