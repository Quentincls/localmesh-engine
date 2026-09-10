"""Restore deterministic, non-state-dict RoPE tensors after meta construction."""
import torch

def materialiser_tampons_de_position(model):
    """Recalculer les tampons de position qu'aucun fichier de poids ne porte.

    `SparseStructureFlowModel` enregistre `rope_phases` (ou `pos_emb`) dans son
    constructeur, calcule depuis sa seule configuration : resolution, nombre de
    canaux, nombre de tetes. Ce ne sont donc PAS des poids appris, et le
    fichier de poids ne les contient pas — construit sur `meta`, le modele les
    garde vides et la premiere passe echoue.

    On les refait ici, avec la classe du modele lui-meme plutot qu'un chemin
    d'import ecrit en dur : c'est ce qui evite que ce code mente le jour ou le
    paquet vendorise bouge.

    Rend le nombre de tampons refaits.
    """
    import importlib
    import torch

    module = importlib.import_module(type(model).__module__)
    refaits = 0
    resolution = getattr(model, "resolution", None)
    canaux = getattr(model, "model_channels", None)
    if resolution is None or canaux is None:
        return 0

    for nom, classe, taille in (
            ("rope_phases", "RotaryPositionEmbedder",
             (canaux // model.num_heads) if getattr(model, "num_heads", 0) else None),
            ("pos_emb", "AbsolutePositionEmbedder", canaux)):
        tampon = getattr(model, nom, None)
        if not torch.is_tensor(tampon) or not tampon.is_meta or taille is None:
            continue
        fabrique = getattr(module, classe, None)
        if fabrique is None:
            continue
        axe = torch.arange(resolution)
        coords = torch.stack(torch.meshgrid(*[axe] * 3, indexing="ij"),
                             dim=-1).reshape(-1, 3)
        setattr(model, nom, fabrique(taille, 3)(coords))
        refaits += 1
    return refaits


def materialize_rotary_frequencies(model):
    repaired = 0
    for module in model.modules():
        if type(module).__name__ == 'SparseRotaryPositionEmbedder' and module.freqs.is_meta:
            frequencies = torch.arange(module.freq_dim,dtype=torch.float32,device='cpu')/module.freq_dim
            module.freqs = module.rope_freq[0]/(module.rope_freq[1]**frequencies)
            repaired += 1
    repaired += materialiser_tampons_de_position(model)
    leftovers = [(type(module).__name__,name) for module in model.modules()
                 for name,value in vars(module).items() if isinstance(value,torch.Tensor) and value.is_meta]
    if leftovers:
        raise RuntimeError(f'Unmaterialized non-parameter tensors: {leftovers}')
    return repaired
