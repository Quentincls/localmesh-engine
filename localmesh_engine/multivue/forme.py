# -*- coding: utf-8 -*-
"""La forme, en deux descentes : 512 puis 1024.

Le volume de cellules donne le squelette ; ces deux passes lui donnent sa
surface. La première travaille sur la grille de 32 héritée de la structure,
subdivise, et rend une grille plus fine ; la seconde travaille dessus et rend
le maillage.

Chaque passe conditionne le modèle avec DEUX moitiés par vue : la carte de
jetons, seize fois plus grossière que la photo, et le champ appris qui
rattrape le détail entre deux jetons. Sans la seconde, les petites pièces
fondent.

Le budget de la seconde passe est un refus, pas un repli : au-delà de son
plafond de cellules, on s'arrête et on le dit. Baisser la résolution en
silence rendrait un objet moins fin sous le même nom.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

from .cameras import nominal_queries
from .naf_echantillon import sample_naf_features
from .poids_meta import materialize_rotary_frequencies
from .repere import decoded_to_local
from .structure import ROLES, _role_projete

#: Le plafond de cellules de la passe fine. C'est le garde-fou de mémoire de
#: la carte : au-delà, la passe ne tient pas sur 8 Go.
PLAFOND_CELLULES = 24000

#: LES PAS SUIVENT LE PALIER, comme sur la voie a une photo.
#:
#: Ils etaient figes a six par passe -- douze au total, le compte du palier
#: Standard. Un client qui payait Detaille recevait donc les pas de Standard,
#: et Brouillon payait ceux de Standard sans les avoir demandes.
#:
#: La regle est celle qui etait deja ecrite ici : les deux etages se PARTAGENT
#: le budget de pas du palier, comme les douze de la voie a une photo servent
#: sa cascade unique.
#:
#:     Brouillon  8 pas   ->  4 + 4
#:     Standard  12 pas   ->  6 + 6      (ce qui tournait, inchange)
#:     Detaille  16 pas   ->  8 + 8
#:
#: MESURE A CONNAITRE AVANT D'Y TOUCHER : doubler les pas de six a douze n'a
#: RIEN change au banc a verite terrain du 9 septembre (moyenne 4,172 contre
#: 4,171 sur sept sujets). Monter les pas n'achete donc pas de fidelite ; ce
#: qu'on corrige ici est une INCOHERENCE d'etiquette, pas un defaut de forme.
import os as _os
PAS = int(_os.environ.get("LUMENGEN_MV_PAS", "6"))


def _conditionner(traits, coords, parite, taille, geometrie, azimuts=None,
                  force=None):
    """Les deux moitiés de chaque vue, projetées et mélangées.

    C'EST ICI QUE LE NEZ EST SCULPTÉ, et c'est ici que la pondération compte le
    plus. L'étage de structure ne pose que des cellules grossières ; les traits
    fins — un nez, une lèvre, une paupière — sortent de cette cascade. Tant
    qu'elle moyennait les quatre vues à plat, une cellule du visage recevait
    aussi ce qu'en disait la vue de dos, qui ne voit rien de ce visage. Sur des
    photos qui ne s'accordent pas sur les proportions, elle sculptait les deux
    réponses : DEUX NEZ EMPILÉS, vus à l'œil par Quentin.

    La pondération est la même qu'à l'étage de structure, appliquée à la grille
    de cet étage-ci. À force nulle, elle rend la moyenne plate au bit près.
    """
    import torch
    import torch.nn.functional as F

    import os

    from .structure import (FORCE_VISIBILITE, poids_des_vues,
                            poids_par_normale, poids_par_visibilite)

    # L'ANCIENNE REGLE RESTE ATTEIGNABLE, POUR LE BANC SEULEMENT. Comparer
    # deux regles a graine differente ne prouve rien ; il faut pouvoir les
    # lancer sur la meme. `LUMENGEN_MV_REGLE=position` rend le proxy
    # positionnel. Rien d'autre ne lit cette variable, et le produit ne la
    # pose jamais.
    # LA REGLE RETENUE EST `poids_des_vues`, rendue anisotrope le 9 septembre
    # 2026. Les deux autres ont ete ecrites, mesurees et ECARTEES le meme
    # jour ; elles restent atteignables pour le banc, avec leurs chiffres :
    #
    #   dos du feu, ressemblance a la photo, meme graine
    #     moyenne plate                          0,476
    #     position ANISOTROPE  (retenue)         0,429
    #     visibilite par profondeur, force 6     0,456   mais samourai 0,711 -> 0,627
    #     position isotrope    (ancienne)        0,377
    #     normale estimee, force 3               0,298
    #
    # La normale est la bonne grandeur en theorie et la pire en pratique :
    # validee sur un maillage final propre, elle tourne en realite sur
    # l'occupation grossiere de l'etage precedent, ou elle n'est que du bruit.
    # Ne pas la reprendre sans une estimation de normale qui tienne a cette
    # resolution-la.
    _regle = {"visibilite": poids_par_visibilite,
              "normale": poids_par_normale,
              }.get(os.environ.get("LUMENGEN_MV_REGLE"), poids_des_vues)

    cote = taille // 16
    # LE VRAI TEST DE VISIBILITÉ, ET SEULEMENT ICI : voir sa docstring pour la
    # mesure. Cet étage-ci travaille sur une occupation réelle ; l'étage de
    # structure, lui, part d'une grille pleine et garde le proxy positionnel.
    poids_vue = _regle(coords, parite,
                       FORCE_VISIBILITE if force is None else force, cote,
                       azimuts=azimuts)
    projete = globaux = None
    for indice, role in enumerate(ROLES):
        jetons = torch.from_numpy(traits[role]["jetons"]).cuda()
        carte = jetons[:, 5:].reshape(1, cote, cote, 1024).permute(0, 3, 1, 2).contiguous()
        grille = nominal_queries(
            coords, role if azimuts else _role_projete(role, parite),
            grid_resolution=cote, image_resolution=taille,
            image_transform=(geometrie or {}).get(role),
            azimut=(azimuts or {}).get(role))
        requetes = torch.from_numpy(traits[role]["requetes"]).cuda()
        with torch.inference_mode():
            basse = F.grid_sample(carte, grille[None, :, None], align_corners=False,
                                  padding_mode="border")[0, :, :, 0].T
            haute = sample_naf_features(requetes, carte, grille)
            vue = torch.cat((basse, haute), 1)
        # Le poids porte sur les traits PROJETÉS, pas sur les jetons globaux :
        # ces cinq-là disent ce qu'EST l'objet, pas où sont ses morceaux.
        vue = vue * poids_vue[indice].unsqueeze(1)
        projete = vue.clone() if projete is None else projete + vue
        globaux = jetons[:, :5].clone() if globaux is None else globaux + jetons[:, :5]
        del jetons, carte, grille, requetes, basse, haute, vue
        gc.collect()
        torch.cuda.empty_cache()
    # Les poids somment déjà à 1 sur les vues : la somme pondérée EST la
    # moyenne. Les jetons globaux, eux, restent une moyenne ordinaire.
    globaux.div_(len(ROLES))
    del poids_vue
    return globaux, projete


def maillage(config, traits_par_taille: dict, cellules, poids, graine: int,
             parite: dict, recette: dict, geometrie: dict | None = None,
             azimuts: dict | None = None, force: float | None = None,
             pas: int | None = None):
    """Descendre la forme et rendre le maillage brut.

    `traits_par_taille` porte les traits encodés pour 512 et pour 1024.
    `cellules` est le volume rendu par l'étage de structure.

    Rend le maillage tel que le décodeur le produit, celui que le nettoyage
    de la voie normale sait prendre.
    """
    import numpy as np
    import torch
    from safetensors.torch import load_file

    models = config.trellis("models")
    creux = config.trellis("modules.sparse")
    samplers = config.trellis("pipelines.samplers.flow_euler")

    coords = torch.from_numpy(cellules).cuda()
    if coords.shape[1] != 4:
        raise ValueError("Cells must carry a batch index")

    sortie = None
    for taille, fichier in ((512, poids.forme_512), (1024, poids.forme_1024)):
        globaux, projete = _conditionner(
            traits_par_taille[taille], coords, parite, taille, geometrie,
            azimuts, force)

        params = json.loads(Path(fichier).with_suffix(".json").read_text())["args"]
        with torch.device("meta"):
            modele = models.SLatFlowModel(**params)
        modele.load_state_dict(load_file(str(fichier)), strict=True, assign=True)
        materialize_rotary_frequencies(modele)
        modele = modele.eval().cuda()

        positif = {"global": globaux,
                   "proj": creux.SparseTensor(feats=projete, coords=coords)}
        negatif = {"global": torch.zeros_like(globaux),
                   "proj": creux.SparseTensor(feats=torch.zeros_like(projete),
                                              coords=coords)}
        torch.manual_seed(graine)
        bruit = creux.SparseTensor(feats=torch.randn(len(coords), 32).cuda(),
                                   coords=coords)
        reglages = dict(recette["forme"], steps=PAS if pas is None else pas)
        with torch.inference_mode():
            latent = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5).sample(
                modele, bruit, cond=positif, neg_cond=negatif, **reglages,
                dino_lock=0, verbose=False).samples
            norme = recette["normalisation_forme"]
            latent = (latent
                      * torch.tensor(norme["std"], device="cuda")[None]
                      + torch.tensor(norme["mean"], device="cuda")[None])
        torch.cuda.synchronize()

        del modele, positif, negatif, globaux, projete, bruit
        gc.collect()
        torch.cuda.empty_cache()

        decodeur = models.from_pretrained(str(
            config.MODELS_ROOT / "TRELLIS.2-4B" / "ckpts_fp8"
            / "shape_dec_next_dc_f16c32_fp8")).eval().cuda()
        # SANS CES DEUX LIGNES, LE DECODEUR REND UN MAILLAGE VIDE. Il porte
        # une resolution interne qui doit s'accorder a celle du latent ; sans
        # elle, rien n'echoue franchement, on recoit zero face et l'erreur
        # tombe beaucoup plus loin, dans le nettoyage.
        decodeur.low_vram = True
        decodeur.set_resolution(taille)
        with torch.inference_mode():
            if taille == 512:
                surface = decodeur.upsample(latent, upsample_times=4)
                coords = torch.cat(
                    (surface[:, :1], ((surface[:, 1:] + .5) / 512 * 63).round().int()),
                    1).unique(dim=0).contiguous()
                if len(coords) > PLAFOND_CELLULES:
                    raise RuntimeError(
                        "The fine pass would need %d cells, over the %d ceiling. "
                        "Nothing is silently downgraded: try one photo, or a "
                        "lighter tier." % (len(coords), PLAFOND_CELLULES))
                del surface
            else:
                maillages = decodeur(latent, useTiled=True)
                brut = maillages[0]
                brut.vertices = decoded_to_local(brut.vertices, grid_resolution=64,
                                                 decoder_grid=64)
                torch.cuda.synchronize()
                # ON REND L'OBJET, PAS DEUX TABLEAUX. Le nettoyage appelle
                # `fill_holes` dessus : lui passer des sommets et des faces
                # nus obligerait a reconstruire ce que le decodeur vient de
                # produire, et a le recopier une fois de plus en memoire.
                sortie = brut
                del maillages
        del decodeur, latent
        gc.collect()
        torch.cuda.empty_cache()

    if sortie is None:
        raise RuntimeError("The fine pass produced no mesh")
    return sortie
