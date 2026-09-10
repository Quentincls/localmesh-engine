import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae',
    
    # vggt
    'ModulatedMultiViewCond': 'sparse_structure_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # MODIFICATION LOCALMESH : construire SANS tirer les poids au hasard.
    #
    # Le constructeur allouait de vrais tenseurs et les remplissait deux fois
    # au hasard — le kaiming de `nn.Linear.__init__`, puis le `normal_` de
    # `initialize_weights()` — avant que la ligne suivante ne les ecrase
    # integralement avec le fichier de poids. MESURE sur le python du produit :
    # 8,7 s par transformeur de 1,3 milliard de parametres, 37 s par travail au
    # palier Standard, soit un dixieme a un tiers d'une generation. Et chaque
    # transformeur etait materialise en fp32 AVANT sa conversion en fp8, ce qui
    # faisait passer la memoire vive de 0,4 a 4,0 Go pour un seul modele — sur
    # une machine ou la famine finale est justement en memoire vive.
    #
    # Le remede etait deja ecrit ailleurs dans ce moteur (multivue/structure.py
    # et multivue/forme.py) : construire sur le peripherique `meta`, qui
    # n'alloue rien, puis poser les tenseurs du fichier par `assign=True`.
    #
    # LE REPLI EST VOLONTAIRE ET IL COMPTE. `strict=False` laisse passer les
    # tampons qui ne sont pas dans le fichier ; s'il en reste un seul sur
    # `meta`, on refait le chemin d'origine plutot que de rendre un modele
    # troue. Le pire cas est donc l'etat d'avant, jamais une panne.
    import torch as _torch

    try:
        with _torch.device('meta'):
            model = __getattr__(config['name'])(**config['args'], **kwargs)

        # LE TYPE DES TENSEURS FAIT PARTIE DU MODELE, ET `assign` NE LE SAIT PAS.
        #
        # Le chemin d'origine COPIE les poids dans des tenseurs deja crees par
        # le constructeur, donc il les convertit au passage : un fichier en fp8
        # donnait un modele en fp32, converti en fp8 plus tard par la chaine.
        # `assign=True` fait l'inverse : il POSE les tenseurs du fichier, et le
        # modele se retrouve en fp8 la ou le code attend du fp32. La generation
        # meurt alors au premier produit matriciel, pas au chargement — un
        # garde-fou qui ne regarde que les tenseurs vides ne voit rien.
        #
        # On releve donc les types que le constructeur voulait — un tenseur
        # `meta` porte son type sans rien allouer — et on les remet apres.
        types_voulus = {nom: t.dtype for nom, t in
                        list(model.named_parameters()) + list(model.named_buffers())}

        model.load_state_dict(load_file(model_file), strict=False, assign=True)
        from ....multivue.poids_meta import materialize_rotary_frequencies
        materialize_rotary_frequencies(model)

        restants = [nom for nom, t in list(model.named_parameters())
                    + list(model.named_buffers()) if t.is_meta]
        if restants:
            raise RuntimeError('tenseurs encore vides : %s' % restants[:4])

        for nom, voulu in types_voulus.items():
            actuel = model.get_parameter(nom) if nom in dict(model.named_parameters()) \
                else model.get_buffer(nom)
            if actuel.dtype == voulu:
                continue
            converti = actuel.detach().to(voulu)
            cible, _, feuille = nom.rpartition('.')
            module = model.get_submodule(cible) if cible else model
            if isinstance(actuel, _torch.nn.Parameter):
                setattr(module, feuille,
                        _torch.nn.Parameter(converti, requires_grad=actuel.requires_grad))
            else:
                module.register_buffer(feuille, converti, persistent=False)

        faux = [nom for nom, voulu in types_voulus.items()
                for t in (dict(list(model.named_parameters())
                               + list(model.named_buffers())).get(nom),)
                if t is not None and t.dtype != voulu]
        if faux:
            raise RuntimeError('types encore faux : %s' % faux[:4])
        return model
    except Exception:                                         # noqa: BLE001
        import logging
        logging.getLogger('localmesh_engine.modeles').warning(
            'construction sur meta impossible pour %s, chemin d origine',
            config['name'], exc_info=True)

    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
