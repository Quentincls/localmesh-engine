"""Display labels for the engine's public progress stages."""

_ENGLISH_STAGES = {
    "encodage de la forme": "encoding the shape",
    "échantillonnage de la texture": "sampling the texture",
    "cuisson de l'atlas": "baking the atlas",
    "projection de la photo": "projecting the photo",
}


def english_stage(stage: str) -> str:
    """Render a stage in English without changing the callback's stage value."""
    return _ENGLISH_STAGES.get(stage, stage)
