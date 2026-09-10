# -*- coding: utf-8 -*-
"""De quel côté les deux profils ont-ils été pris ?

POURQUOI CETTE ÉTAPE EXISTE. Le multivue projette chaque photo sur une grille
commune par sa caméra. Les deux profils sont interchangeables du point de vue
d'une étiquette : l'emplacement où l'utilisateur a rangé l'image dit où elle a
été CLASSÉE, pas d'où elle a été PRISE. Se tromper de côté ne dégrade pas un
peu le résultat, ça le retourne : le samouraï du banc sortait avec un torse
frontal, le masque et deux lames dégainées À L'ARRIÈRE. Vérifié en argile,
sans texture : c'était sculpté.

Une convention ne suffit pas. Sur les six sujets mesurés, cinq la suivent et un
non ; celui-là sort avec deux visages et rien ne prévient. La convention reste
le repli quand la mesure est impossible ou hésitante, et le résultat le dit.

CE QU'ON MESURE, ET CE QU'ON NE MESURE PAS. On ne calibre pas les caméras. On
répond à UNE question binaire : les deux profils sont-ils à leur place, ou
échangés ? Le modèle rend un angle par vue depuis deux têtes de pose ; on ne
retient la réponse que si les deux têtes s'accordent, si les angles tombent
dans le secteur d'un profil, et si les deux profils sortent de côtés opposés.
Sinon on répond « je ne sais pas », qui est une réponse honnête.

La face et le dos restent des ancres : ils ne sont jamais réétiquetés.

COÛT. Le modèle pèse 517 Mo, son code 1,1 Mo, ses dépendances 1,1 Mo. Il tourne
UNE fois, AVANT que quoi que ce soit d'autre ne soit résident, en un peu moins
de deux secondes pour un pic mesuré à 2,6 Go, puis il rend la carte. Il est
sous licence Apache 2.0, code et poids.
"""
from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

#: Les quatre rôles, dans l'ordre où le modèle reçoit les images.
ROLES = ("front", "right", "back", "left")

#: La règle de décision, mesurée sur six sujets. Un profil doit tomber dans ce
#: secteur pour compter : au-delà, la photo n'est pas un profil et l'angle ne
#: départage rien.
SECTEUR = (45.0, 135.0)
#: Une photo prise en plongée ou en contre-plongée forte ne dit plus de quel
#: côté elle est.
ELEVATION_MAX = 30.0
#: Les deux têtes partagent un tronc commun : leur accord n'est pas une
#: confiance calibrée, c'est un garde-fou contre une sortie aberrante.
DESACCORD_MAX = 15.0

#: La résolution d'entrée du modèle, fixée par son entraînement.
_RESOLUTION = 504


def dossier(config) -> Path:
    return Path(config.MODELS_ROOT) / "multivue" / "cameras"


def _code_range(d: Path) -> bool:
    """Le code du mesureur est-il rangé à côté de ses poids ?"""
    return (d / "source" / "src").is_dir() and (d / "deps").is_dir()


def _code_installe() -> bool:
    """Ou bien `depth_anything_3` est-il simplement installé ?"""
    import importlib.util
    try:
        return importlib.util.find_spec("depth_anything_3") is not None
    except (ImportError, ValueError):
        return False


def manquants(config) -> list[str]:
    """Ce qui manque pour mesurer, en clair. Vide = on peut mesurer.

    DEUX FAÇONS D'AVOIR LE CODE, ET LES DEUX COMPTENT. L'application le range
    à côté des poids pour n'entrer dans l'environnement de personne ; qui
    clone le dépôt public installe le paquet publié. Exiger la première
    fermait la porte à la seconde, et rien ne le disait.
    """
    d = dossier(config)
    absents = []
    for nom, chemin in (("poids du mesureur", d / "model.safetensors"),
                        ("description du mesureur", d / "config.json")):
        if not chemin.exists():
            absents.append(nom)
    if not _code_range(d) and not _code_installe():
        absents.append("code du mesureur (depth_anything_3)")
    return absents


def disponible(config) -> bool:
    return not manquants(config)


def decision(camera: dict, rayon: dict) -> dict:
    """Trancher le côté des deux profils, et rien d'autre.

    `camera` et `rayon` portent, par rôle, `lacet` et `elevation` en degrés,
    mesurés relativement à la vue de face.

    Rend `nominal`, `swap_sides` ou `ambiguous`. La règle est celle d'Astra,
    reprise telle quelle parce qu'elle est mesurée : convention de signe,
    profil droit NÉGATIF et profil gauche positif quand tout est à sa place.
    """
    rapport = {
        "decision": "ambiguous",
        "regle": {"secteur_deg": list(SECTEUR),
                  "elevation_max_deg": ELEVATION_MAX,
                  "desaccord_max_deg": DESACCORD_MAX},
        "ancres": {"front": 0, "back": 180},
        "controles": {},
        "limites": ("Deux têtes partagent un tronc : leur accord n'est pas une "
                    "confiance calibrée. Seul le côté des profils est déduit, "
                    "jamais une calibration de caméra ni un réétiquetage du dos."),
    }
    signes = []
    for role in ("right", "left"):
        a, b = camera[role], rayon[role]
        ya, yb = a["lacet"], b["lacet"]
        ecart = abs((ya - yb + 180) % 360 - 180)
        valide = (
            all(math.isfinite(v) for v in (ya, yb, a["elevation"], b["elevation"]))
            and SECTEUR[0] <= abs(ya) <= SECTEUR[1]
            and SECTEUR[0] <= abs(yb) <= SECTEUR[1]
            and abs(a["elevation"]) <= ELEVATION_MAX
            and abs(b["elevation"]) <= ELEVATION_MAX
            and ecart <= DESACCORD_MAX
            and ya * yb > 0)
        rapport["controles"][role] = {
            "dans_le_secteur": valide, "lacet_camera": ya, "lacet_rayon": yb,
            "desaccord_deg": ecart}
        signes.append((1 if ya > 0 else -1) if valide else 0)
    if signes == [-1, 1]:
        rapport["decision"] = "nominal"
    if signes == [1, -1]:
        rapport["decision"] = "swap_sides"
    # Un dos qui n'est pas au dos veut dire que les photos ne forment pas le
    # tour qu'on croit. On ne s'en sert pas pour decider, on le signale.
    rapport["le_dos_contredit_son_etiquette"] = any(
        abs(abs(m["back"]["lacet"]) - 180) > 45 for m in (camera, rayon))
    return rapport


def _paire(modele, images):
    """Les deux têtes de pose, sur un seul passage du tronc commun.

    Les deux têtes partent APRÈS le même calcul commun et depuis le même état
    aléatoire : c'est ce qui rend leurs deux réponses comparables. Chacune
    reçoit sa propre copie, sinon la première abîme l'entrée de la seconde.
    """
    import torch

    def copier(v):
        if isinstance(v, torch.Tensor):
            return v.clone()
        if isinstance(v, dict):
            return type(v)({k: copier(x) for k, x in v.items()})
        if isinstance(v, list):
            return [copier(x) for x in v]
        if isinstance(v, tuple):
            return tuple(copier(x) for x in v)
        return v

    if modele.training:
        raise ValueError("La mesure demande le mode evaluation")
    if images.ndim != 5 or tuple(images.shape[:3]) != (1, 4, 3):
        raise ValueError("Quatre vues RVB attendues")

    traits, aux = modele.backbone(
        images, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
    h, w = images.shape[-2:]
    with torch.autocast(device_type=images.device.type, enabled=False):
        commun = modele._process_depth_head(traits, h, w)
        etat_cpu = torch.get_rng_state()
        etat_gpu = (torch.cuda.get_rng_state(images.device)
                    if images.is_cuda else None)
        camera = modele._process_camera_estimation(traits, h, w, copier(commun))
    camera = modele._process_mono_sky_estimation(camera)
    # La tête « camera » consomme du hasard ; sans cette remise a zero, la tête
    # « rayon » ne tire plus les memes echantillons et les deux mesures ne sont
    # plus comparables.
    torch.set_rng_state(etat_cpu)
    if etat_gpu is not None:
        torch.cuda.set_rng_state(etat_gpu, images.device)
    with torch.autocast(device_type=images.device.type, enabled=False):
        rayon = modele._process_ray_pose_estimation(copier(commun), h, w)
    rayon = modele._process_mono_sky_estimation(rayon)
    return {"camera": camera, "rayon": rayon}


def _charger(config):
    """Monter le modèle, en rétablissant ses poids partagés.

    Le fichier de poids ne stocke qu'une fois les tenseurs que le modèle
    partage entre plusieurs endroits. Les recoller par IDENTITÉ plutôt que par
    nom est ce qui permet un chargement strict : on refuse de deviner.
    """
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object

    d = dossier(config)
    conf = json.loads((d / "config.json").read_text(encoding="utf-8"))["config"]
    modele = create_object(OmegaConf.create(conf))
    poids = load_file(str(d / "model.safetensors"))
    if not all(k.startswith("model.") for k in poids):
        raise RuntimeError("Poids du mesureur inattendus")
    poids = {k[6:]: v for k, v in poids.items()}

    nommes = dict(modele.named_parameters(remove_duplicate=False))
    nommes.update(dict(modele.named_buffers(remove_duplicate=False)))
    par_identite = {id(nommes[k]): k for k in poids if k in nommes}
    for k in set(modele.state_dict()) - set(poids):
        alias = par_identite.get(id(nommes[k]))
        if alias is None:
            raise RuntimeError("Poids absent et non partagé : %s" % k)
        poids[k] = poids[alias]
    modele.load_state_dict(poids, strict=True)
    del poids
    return modele.eval().cuda()


def _angles(resultat, roles) -> dict:
    """Le lacet et l'élévation de chaque vue, relativement à la face."""
    import numpy as np

    extrinseques = resultat["extrinsics"].detach().float().cpu().numpy()[0]
    E = np.tile(np.eye(4), (len(roles), 1, 1))
    E[:, :3, :] = extrinseques[:, :3, :]
    relatif = E[0] @ np.linalg.inv(E)
    avant = relatif[:, :3, 2]
    lacet = np.degrees(np.arctan2(avant[:, 0], avant[:, 2]))
    elevation = np.degrees(np.arcsin(np.clip(-avant[:, 1], -1, 1)))
    return {r: {"lacet": float(lacet[i]), "elevation": float(elevation[i])}
            for i, r in enumerate(roles)}


#: Deux vues qui tombent l'une sur l'autre ne fusionnent plus, elles se
#: recouvrent. En deçà de cet écart entre deux angles mesurés, on refuse la
#: mesure et on reprend les angles ronds : quatre vues mal réparties valent
#: encore mieux que deux vues empilées.
ECART_MINI_ENTRE_VUES = 20.0


def azimuts_mesures(rapport: dict) -> dict | None:
    """Les angles de prise de vue, dans la convention de la projection.

    Rend `{role: azimut_en_degres}` quand la mesure est fiable, `None` sinon.
    L'appelant retombe alors sur les angles ronds.

    LA CONVENTION DE SIGNE EST INVERSE DE CELLE DE LA MESURE, et ce n'est pas
    un détail : `decision` lit un profil droit NÉGATIF quand tout est à sa
    place, tandis que `cameras.AZIMUTHS` donne +90 au rôle « right ». L'azimut
    de projection vaut donc MOINS le lacet mesuré. Contrôle : un tour parfait
    mesure (0, -90, 180, +90) et rend (0, 90, 180, 270), c'est-à-dire
    exactement `AZIMUTHS`.

    CE QUI FAIT REFUSER LA MESURE. Les mêmes contrôles que la décision de
    latéralité — les deux têtes doivent s'accorder, l'élévation rester faible,
    les nombres être finis — plus un contrôle propre à cet usage : deux vues
    ne doivent pas tomber au même endroit.
    """
    angles = (rapport or {}).get("angles") or {}
    cam, ray = angles.get("camera"), angles.get("rayon")
    if not cam or not ray:
        return None

    # DEUX REFUS QUI ONT ETE PAYES LE 9 SEPTEMBRE 2026, ET QUI NE SONT PAS DES
    # PRECAUTIONS DE PRINCIPE.
    #
    # Depth Anything 3 s'egare sur les sujets pauvres en relief. Mesure sur
    # trois sujets, photos verifiees a l'oeil une par une :
    #
    #   samourai   diorama, socle, arbre, lanterne     mesure JUSTE
    #   feu        panneau large et plat, symetrique   le vrai DOS mesure a 79 deg
    #   buste      crane lisse, quasi symetrique       le vrai DOS mesure a 72 deg
    #
    # Sur les deux derniers, croire la mesure aurait projete la vue de dos sur
    # un flanc -- bien pire que l'angle rond qu'on remplace. Le premier jet de
    # cette fonction les acceptait tous les deux.
    #
    # CE QUI LES SEPARE EXISTAIT DEJA DANS LE RAPPORT :
    #
    # 1. `decision` ne tranche que si les deux profils tombent dans le secteur
    #    45-135 degres avec les deux tetes d'accord. Le feu et le buste rendent
    #    « ambiguous » ; le samourai rend « swap_sides ». Une pose trop pauvre
    #    pour dire de quel cote est un profil est trop pauvre pour donner un
    #    angle.
    #
    # 2. Le DOS est la seule vue que l'utilisateur ancre lui-meme en la rangeant.
    #    S'il ne se mesure pas vers 180 degres, la pose contredit la seule chose
    #    qu'on savait, et c'est elle qu'on croit.
    if rapport.get("decision") not in ("nominal", "swap_sides"):
        return None
    if rapport.get("le_dos_contredit_son_etiquette"):
        return None
    sortie = {}
    for role in ROLES:
        a, b = cam.get(role), ray.get(role)
        if not a or not b:
            return None
        ya, yb = a["lacet"], b["lacet"]
        if not all(math.isfinite(v) for v in (ya, yb, a["elevation"], b["elevation"])):
            return None
        if abs(a["elevation"]) > ELEVATION_MAX or abs(b["elevation"]) > ELEVATION_MAX:
            return None
        if abs((ya - yb + 180) % 360 - 180) > DESACCORD_MAX:
            return None
        sortie[role] = -((ya + yb) / 2) % 360
    vus = list(sortie.values())
    for i in range(len(vus)):
        for j in range(i + 1, len(vus)):
            if abs((vus[i] - vus[j] + 180) % 360 - 180) < ECART_MINI_ENTRE_VUES:
                return None
    return sortie


def mesurer(config, vues: dict) -> dict:
    """Mesurer le côté des deux profils sur les quatre vues.

    `vues` porte, par rôle, l'image DÉJÀ DÉTOURÉE — celle que la chaîne vient
    de préparer, pas la photo d'origine. C'est sur celles-là que la règle de
    décision a été mesurée, et c'est aussi ce qui évite de relire les fichiers
    et de refaire le détourage une seconde fois.

    Rend le rapport de `decision`, augmenté des angles mesurés. Ne lève pas si
    la mesure échoue : elle rend `ambiguous` avec la raison, parce qu'un
    travail rendu vaut mieux qu'une erreur pour un choix binaire que
    l'utilisateur peut corriger d'un mot.
    """
    import torch

    d = dossier(config)
    # Le code rangé à côté des poids passe DEVANT l'environnement : c'est la
    # version contre laquelle le moteur a été mesuré. S'il n'est pas là, on
    # laisse l'import se résoudre normalement, sur le paquet installé.
    if _code_range(d):
        for chemin in (d / "source" / "src", d / "deps"):
            if str(chemin) not in sys.path:
                sys.path.insert(0, str(chemin))

    modele = images = sortie = None
    try:
        from depth_anything_3.utils.io.input_processor import InputProcessor

        # Le lecteur accepte des images en memoire ; on ne repasse pas par
        # des fichiers pour des pixels qu'on tient deja.
        images, _, _ = InputProcessor()(
            [vues[r] for r in ROLES], process_res=_RESOLUTION,
            process_res_method="upper_bound_resize", num_workers=1,
            sequential=True)
        # Selon sa version il rend (N,3,H,W) ou (1,N,3,H,W). On accepte les
        # deux plutot que d'epingler une version par un plantage.
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5 or tuple(images.shape[:3]) != (1, 4, 3):
            raise RuntimeError("Le lecteur d'images a rendu %s"
                               % (tuple(images.shape),))
        modele = _charger(config)
        images = images.cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sortie = _paire(modele, images)
        torch.cuda.synchronize()
        mesures = {tete: _angles(r, ROLES) for tete, r in sortie.items()}
        rapport = decision(mesures["camera"], mesures["rayon"])
        rapport["angles"] = mesures
        return rapport
    except Exception as exc:                                  # noqa: BLE001
        return {"decision": "ambiguous", "erreur": repr(exc)}
    finally:
        # LA CARTE EST RENDUE QUOI QU'IL ARRIVE. Cette mesure passe avant les
        # quatre encodages et les deux cascades ; un demi-giga oublié ici se
        # paie tout le reste de la generation.
        del modele, images, sortie
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:                                     # noqa: BLE001
            pass
