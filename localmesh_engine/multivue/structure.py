# -*- coding: utf-8 -*-
"""Quatre vues, un volume de cellules.

C'est l'étape qui distingue ce chemin de l'ancien. L'ancien mélangeur prenait
une prédiction par vue et les moyennait en pondérant par la position supposée
de chaque caméra ; dès que le modèle plaçait l'objet autrement qu'attendu, les
poids tombaient au mauvais endroit et l'objet héritait de morceaux en trop.

Ici, rien n'est pondéré par position. Les traits de chaque vue sont PROJETÉS
sur la grille par sa caméra, et c'est un modèle entraîné pour cette fusion qui
décide. La pose de l'objet cesse d'être une supposition de notre part.

La sortie est une liste de cellules occupées sur une grille de 32, celle que
la cascade de forme prend en entrée.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from .cameras import nominal_queries
from .poids_meta import materialize_rotary_frequencies

#: Les rôles, dans l'ordre où on les additionne. L'ordre ne change pas le
#: résultat (c'est une moyenne) mais le fixer rend deux exécutions
#: comparables.
ROLES = ("front", "right", "back", "left")

#: COMBIEN CHAQUE VUE PÈSE, SELON CE QU'ELLE VOIT.
#:
#: La fusion officielle est une MOYENNE PLATE : dans chaque cellule, les quatre
#: vues comptent pareil, y compris celle qui regarde l'objet par derrière et ne
#: peut rien en dire. Tant que les quatre photos s'accordent, ça marche : elles
#: disent la même chose, la moyenne ne perd rien. Dès qu'elles se contredisent,
#: la cellule reçoit deux réponses et le modèle sculpte les deux. MESURÉ sur le
#: buste de pierre : un nez, des lèvres et des orbites en double.
#:
#: L'ancien mélangeur, lui, pondère par la position de chaque caméra, et c'est
#: exactement pour ça qu'il encaisse mieux le désaccord.
#:
#: Le poids est une exponentielle du produit scalaire entre la cellule et la
#: direction de la caméra : une cellule du devant compte surtout pour la vue de
#: face. À zéro, les quatre poids sont égaux et on retrouve la moyenne plate au
#: bit près — c'est ce qui rend la comparaison honnête.
#:
#: LA VALEUR PAR DÉFAUT EST 6, ET ELLE EST MESURÉE. Sur le buste de pierre,
#: en argile, nez et bouche de près :
#:
#:     moyenne plate    DEUX nez empilés, deux lèvres
#:     force 6          un nez, des narines, une lèvre
#:     force 14         un nez aussi, mais le visage se facette
#:
#: ELLE DOIT S'APPLIQUER AUX DEUX ÉTAGES. Posée sur la seule structure, elle ne
#: changeait rien au nez : la structure ne pose que des cellules grossières,
#: c'est la cascade de forme qui sculpte les traits fins. On l'a cru corrigé une
#: fois pour cette raison, à tort.
#:
#: LA FORCE EST NORMALISÉE PAR LA TAILLE DU SUJET, ET LE CHIFFRE A CHANGÉ AVEC.
#: Sans normalisation, une force de 6 donnait 87 % de contraste sur un sujet
#: large et 44 % sur une silhouette élancée : le réglage ne voulait pas dire la
#: même chose d'un objet à l'autre, et les sujets les plus exposés recevaient la
#: correction la plus molle. Normalisée, la force 3 donne 88 % PARTOUT — le même
#: régime que celui où le buste a été validé, mais tenu sur tous les sujets.
FORCE_VISIBILITE = float(os.environ.get("LUMENGEN_MV_VISIBILITE", "3"))


def _role_projete(role: str, parite: dict) -> str:
    """Le rôle sous lequel cette vue est projetée.

    Les deux profils échangent leur place quand la latéralité est inversée ;
    la face et le dos sont des ancres et ne bougent jamais.
    """
    if not parite.get("swap_sides"):
        return role
    return {"right": "left", "left": "right"}.get(role, role)


#: AU-DELA DE QUEL DESACCORD LA PONDERATION SERT-ELLE ?
#:
#: Elle est un ARBITRE, et un arbitre ne sert que s'il y a litige. Mesure du
#: 9 septembre 2026 : sur les rendus du banc a verite terrain -- des vues
#: parfaitement coherentes -- la ponderation est INERTE (2,905 / 1,865 / 1,364
#: contre 2,898 / 1,868 / 1,372, soit moins de 0,01 %). Sur de vraies photos
#: elle DECIDE, et pas toujours dans le bon sens :
#:
#:     buste     elle sauve      un nez au lieu de deux
#:     samourai  elle tient      accord 0,708 contre 0,721 a plat
#:     feu       elle ABIME      dos 0,429 avec, 0,476 sans
#:
#: CE QUI SEPARE CES SUJETS SE MESURE AVANT DE GENERER, et le moteur le mesure
#: deja : `_accord_des_vues` compare la hauteur relative du sujet dans les
#: quatre cadres.
#:
#:     feu       0,2 %     photos d'un vrai tour, elles s'accordent
#:     pot       0,3 %     (rendus)
#:     lampion   1,0 %     (rendus)
#:     ---------------------- le fosse -------------------------
#:     samourai  5,0 %     prises a la main, distances differentes
#:     buste     8,7 %
#:
#: CE SEUIL N'ARBITRE PLUS RIEN, et il reste pour ce qu'il apprend. Il a
#: pilote la force pendant une journee ; la mesure du 9 septembre 2026 a montre
#: que la force elle-meme devait tomber a zero (voir `force_selon_desaccord`),
#: donc plus rien ne le lit. L'ecart de hauteur entre photos reste mesure et
#: sert AILLEURS : au-dela de `ECART_DE_VUES_MAX`, `pipeline.py` en avertit
#: l'utilisateur, parce que des photos prises a des distances trop differentes
#: sont un vrai probleme -- simplement pas celui-la.
SEUIL_DESACCORD = float(os.environ.get("LUMENGEN_MV_SEUIL_DESACCORD", "3.0"))


def force_selon_desaccord(accord: dict | None) -> float:
    """La force a employer. ZERO, et c'est mesure.

    ELLE A ETE ADAPTATIVE PENDANT UNE JOURNEE, ET C'ETAIT UNE ERREUR. La regle
    disait : les photos s'accordent -> force 0 ; elles se contredisent -> force
    3, pour que chaque vue commande la zone qu'elle voit. Le seuil etait pose a
    3 % d'ecart de hauteur du sujet.

    CE QUE CETTE REGLE A COUTE, VU A L'ECRAN LE 9 SEPTEMBRE 2026. Un buste de
    pierre fissuree est sorti meconnaissable : crane gonfle en oeuf, visage
    ecrase, peau bardee de plaques separees par des ravins. Ses quatre photos
    variaient de 8,7 % en hauteur -- un simple ecart de cadrage -- donc la
    regle a mis la force a fond sur le sujet qui la supportait le moins.

    LE MECANISME. A force elevee, chaque photo commande sa zone ; sur un sujet
    dont toute la peau est un motif, chaque vue GRAVE SON MOTIF dans sa zone,
    et les frontieres entre zones deviennent des aretes vives. Le modele de
    forme ne distingue pas une fissure peinte d'une fissure creusee : la
    ponderation lui demande de trancher quatre fois, il sculpte quatre fois.

    LA MESURE, onze generations, meme graine, une variable a la fois. La part
    d'aretes qui plient de plus de 30 degres, et le nombre de morceaux :

        sujet                    palier      force 3            force 0
        buste fissure (Quentin)  Detaille    17,2 %  3 882      5,95 %    394
        buste fissure (banc)     Detaille    14,7 %  1 748      9,76 %  1 551
        buste fissure (banc)     Standard    25,3 %  4 611      4,41 %    281
        samourai (Quentin)       Detaille    29,7 %  6 053      21,5 %  1 321

    Force 0 gagne sur les quatre, et de loin. Le samourai garde ses deux
    sabres, sa cape et son socle garni ; il est seulement plus propre.

    ET LE DEFAUT POUR LEQUEL LA FORCE EXISTAIT NE REVIENT PAS. Elle avait ete
    posee parce que la fusion a plat donnait DEUX BOUCHES au buste. Rejoue a
    force 0, sous les quatre angles, en Standard comme en Detaille : une seule
    bouche, un seul visage, un crane propre. Entre-temps deux vrais correctifs
    sont arrives -- les angles de prise de vue mesures, et la normalisation par
    axe. Ils traitent la cause. La force etait un pansement sur une plaie
    refermee.

    CE QUI RESTE VRAI DE L'ANCIENNE NOTE : le dos d'un feu perdait a recevoir
    la ponderation (0,476 -> 0,429). Ce chiffre allait deja dans le meme sens ;
    il n'avait pas suffi a le voir.

    `LUMENGEN_MV_VISIBILITE` et le champ `force_mv` de la demande imposent une
    autre valeur : c'est ce qui permet de refaire cette mesure sans redemarrer
    le moteur, et de la contredire si un sujet le demande un jour.
    """
    if os.environ.get("LUMENGEN_MV_VISIBILITE") is not None:
        return FORCE_VISIBILITE
    return 0.0


def poids_par_normale(coords, parite, force: float, resolution: int,
                      azimuts: dict | None = None):
    """Ce que chaque vue pèse, selon l'ORIENTATION de la surface.

    LES DEUX REGLES QUI ONT PRECEDE, ET CE QUI LEUR MANQUAIT.

    `poids_des_vues` note une vue sur la POSITION de la cellule : une cellule
    vers l'arrière compte pour la vue de dos. Sur une tête, position et
    orientation se confondent — une tête est ronde — et ça marche : le buste
    de pierre sort avec un seul nez là où la moyenne plate en donne deux.

    Sur un objet LARGE ET PLAT, elles n'ont plus rien à voir. Mesure du
    9 septembre 2026 sur le feu tricolore, panneau large photographié aux
    quatre angles ronds, donc des données parfaites :

        cellule du panneau arrière, au centre       vue de dos  52 %
        cellule du panneau arrière, près du bord    vue de dos  18 %
                                                ... vue de DROITE  78 %

    La vue de droite rase ce panneau et n'en sait rien. Comme un panneau large
    a l'essentiel de son aire près des bords, LA MAJORITE DU DOS ETAIT DECIDEE
    PAR DES VUES QUI NE LE VOIENT PAS : le dos s'enfonce, un éventail de
    triangles le capote, et la cuisson ne voit jamais ces faces — elles
    sortent noires. Quentin : « il est magnifique de face, et de dos, gros
    trou ».

    `poids_par_visibilite` (test de profondeur) corrige le feu mais affame les
    profils : à force 6, l'accord du samouraï tombe de 0,711 à 0,627, ses deux
    profils de 0,65 à 0,44. Une cellule à peine masquée perdait toute voix.

    CE QU'ON FAIT ICI. On estime la NORMALE de chaque cellule depuis
    l'occupation — la direction du vide autour d'elle — et on note chaque vue
    par `n · u`. C'est la grandeur juste : sur une tête, la normale est
    radiale et on retrouve exactement l'ancienne règle ; sur un panneau plat,
    elle vaut -z partout et la vue de dos gagne sur TOUT le panneau, bord
    compris. Le score reste dans [-1, 1] comme celui de la position, donc la
    force garde son ordre de grandeur.

    POURQUOI PAS A L'ETAGE DE STRUCTURE : là, la grille est PLEINE, il n'y a
    pas de vide autour d'une cellule et la normale n'existe pas. Cet étage
    garde le proxy positionnel.

    À force nulle, rend exactement 1/4 partout.
    """
    import math
    import torch

    from .cameras import AZIMUTHS

    n = coords.shape[0]
    if force == 0.0:
        return torch.full((len(ROLES), n), 1.0 / len(ROLES), device=coords.device)

    c = coords[:, 1:].long()
    occupe = torch.zeros((resolution,) * 3, dtype=torch.bool, device=coords.device)
    occupe[c[:, 0], c[:, 1], c[:, 2]] = True

    # LA NORMALE, C'EST LA DIRECTION DU VIDE. On additionne les 26 voisins
    # VIDES : une cellule au fond d'un creux voit le vide d'un seul côté, une
    # cellule perdue au milieu de la matière n'en voit aucun et garde une
    # normale nulle — elle retombe alors sur la moyenne plate, ce qui est le
    # bon aveu d'ignorance.
    normale = torch.zeros((n, 3), dtype=torch.float32, device=coords.device)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                v = c + torch.tensor([dx, dy, dz], device=coords.device)
                dehors = ((v < 0) | (v >= resolution)).any(1)
                vc = v.clamp(0, resolution - 1)
                vide = ~occupe[vc[:, 0], vc[:, 1], vc[:, 2]] | dehors
                d = torch.tensor([dx, dy, dz], dtype=torch.float32,
                                 device=coords.device)
                normale += vide.float()[:, None] * (d / d.norm())
    longueur = normale.norm(dim=1, keepdim=True)
    normale = normale / longueur.clamp_min(1e-6)

    scores = []
    for role in ROLES:
        vu = (azimuts or {}).get(role)
        lacet = math.radians(AZIMUTHS[_role_projete(role, parite)]
                             if vu is None else vu)
        vers = torch.tensor([math.sin(lacet), 0.0, math.cos(lacet)],
                            device=coords.device, dtype=normale.dtype)
        scores.append((normale * vers).sum(-1))
    return torch.softmax(force * torch.stack(scores), dim=0)


def poids_par_visibilite(coords, parite, force: float, resolution: int,
                         azimuts: dict | None = None):
    """Ce que chaque vue pèse, selon qu'elle VOIT la cellule ou non.

    LE DÉFAUT QUE CECI CORRIGE, ET SA MESURE. `poids_des_vues` note une vue
    sur la POSITION de la cellule : une cellule vers l'arrière compte pour la
    vue de dos. Sur une boule, la position et l'orientation de la surface se
    confondent et ça marche. Sur un objet LARGE ET PLAT, elles n'ont plus rien
    à voir.

    Mesure du 9 septembre 2026 sur le feu tricolore de Quentin — un panneau
    large, plat, photographié aux quatre angles ronds, donc des données
    parfaites :

        cellule du panneau arrière, au centre (x=0)     vue de dos  52 %
        cellule du panneau arrière, près du bord (x=0,2) vue de dos  18 %
                                                     ... vue de DROITE  78 %

    La vue de droite ne voit rien de ce panneau : elle le rase. Et comme un
    panneau large a l'essentiel de son aire près des bords, LA MAJORITÉ DU DOS
    ÉTAIT DÉCIDÉE PAR DES VUES QUI NE LE VOIENT PAS. Résultat à l'écran : le
    dos s'enfonce, un éventail de triangles le capote, et la cuisson de texture
    ne voit jamais ces faces — elles sortent noires. Quentin : « il est
    magnifique de face, et de dos, gros trou ».

    CE QU'ON FAIT À LA PLACE. On dispose ici d'une occupation réelle — les
    cellules que l'étage précédent a posées — donc on peut faire ce que le nom
    « pondération par visibilité » promettait : un test de profondeur. Pour
    chaque vue, on projette les cellules, on les range par pixel, et on garde
    le retard de chacune sur la plus proche de son pixel. Une cellule en
    première ligne pour une vue a un retard nul ; une cellule cachée derrière
    la moitié de l'objet a un grand retard, et cette vue cesse de décider pour
    elle.

    POURQUOI PAS À L'ÉTAGE DE STRUCTURE : là, la grille est PLEINE — les 16³
    cellules sont candidates — donc « la plus proche » ne désigne que la
    surface du cube et ne veut rien dire. Cet étage garde le proxy positionnel.

    À force nulle, rend exactement 1/4 partout, comme `poids_des_vues` : on ne
    veut pas « presque » la moyenne plate quand on demande la moyenne plate.
    """
    import math
    import torch

    from .cameras import AZIMUTHS, DISTANCE

    n = coords.shape[0]
    if force == 0.0:
        return torch.full((len(ROLES), n), 1.0 / len(ROLES), device=coords.device)

    axe = torch.linspace(-1, 1, resolution, device=coords.device) / 2
    p = axe[coords[:, 1:].long()]

    retards = []
    for role in ROLES:
        vu = (azimuts or {}).get(role)
        lacet = math.radians(AZIMUTHS[_role_projete(role, parite)]
                             if vu is None else vu)
        sin, cos = math.sin(lacet), math.cos(lacet)
        # Le même repère que `nominal_queries` : profondeur croissante en
        # s'éloignant de la caméra, et les deux axes de l'image.
        profondeur = DISTANCE - sin * p[:, 0] - cos * p[:, 2]
        u = cos * p[:, 0] - sin * p[:, 2]
        v = p[:, 1]
        # L'EPAISSEUR DE L'OBJET VU DE CETTE VUE-LA. Sans elle, un retard se
        # mesure en unites de grille : une plaque mince separe dix fois moins
        # qu'une boule, et la force ne veut plus rien dire d'un sujet a
        # l'autre. C'est exactement l'erreur deja payee sur le proxy
        # positionnel, ou il a fallu diviser par le rayon du sujet.
        epaisseur = (profondeur.max() - profondeur.min()).clamp_min(1e-6)

        # Un pixel par cellule de la grille : plus fin ne sert à rien, plus
        # grossier mélangerait des colonnes voisines.
        iu = ((u + 0.5) * resolution).clamp(0, resolution - 1).long()
        iv = ((v + 0.5) * resolution).clamp(0, resolution - 1).long()
        seau = iu * resolution + iv

        # La plus proche de chaque pixel, par un minimum dispersé.
        proche = torch.full((resolution * resolution,), float("inf"),
                            device=coords.device, dtype=profondeur.dtype)
        proche.scatter_reduce_(0, seau, profondeur, reduce="amin",
                               include_self=True)
        retards.append((profondeur - proche[seau]) / epaisseur)

    # Le retard vaut donc 0 en premiere ligne et 1 pour une cellule cachee
    # derriere TOUTE l'epaisseur de l'objet, quelle que soit sa taille.
    return torch.softmax(-force * torch.stack(retards), dim=0)


def poids_des_vues(coords, parite, force: float, resolution: int = 16,
                   azimuts: dict | None = None):
    """Ce que chaque vue pèse dans chaque cellule.

    Rend un tenseur (4, N) qui somme à 1 sur les vues, dans l'ordre de ROLES.
    À force nulle il vaut exactement 1/4 partout : on ne veut pas « presque »
    la moyenne plate quand on demande la moyenne plate, on veut la même.

    `resolution` est le côté de la grille où vivent les coordonnées : 16 pour
    la structure, 32 puis 64 pour les deux étages de la cascade de forme. Le
    même calcul sert aux trois — c'est la POSITION dans l'objet qui décide,
    pas la finesse à laquelle on la regarde.
    """
    import math
    import torch

    from .cameras import AZIMUTHS

    n = coords.shape[0]
    if force == 0.0:
        return torch.full((len(ROLES), n), 1.0 / len(ROLES),
                          device=coords.device)

    # Les cellules, ramenées au centre de la grille, dans les mêmes unités que
    # `nominal_queries` : c'est la position de la cellule qui décide.
    axe = torch.linspace(-1, 1, resolution, device=coords.device) / 2
    p = axe[coords[:, 1:].long()]

    # LA FORCE DOIT VOULOIR DIRE LA MÊME CHOSE SUR TOUS LES SUJETS.
    #
    # Le score est un produit scalaire, donc il grandit avec l'objet. Sans
    # cette normalisation, une force de 6 donnait 91 % de contraste sur un
    # sujet large et environ la moitié sur une silhouette élancée — un buste,
    # un personnage debout. Autrement dit : les sujets les plus exposés aux
    # traits en double recevaient la correction la plus molle, et le réglage
    # ne voulait rien dire d'un objet à l'autre.
    #
    # On divise par le rayon horizontal du sujet, mesuré sur les cellules qu'on
    # a. La hauteur ne compte pas : les quatre caméras tournent autour de l'axe
    # vertical, donc seul l'éloignement horizontal décide de qui voit quoi.
    rayon = p[:, [0, 2]].norm(dim=-1).max()
    if not torch.isfinite(rayon) or rayon <= 0:
        rayon = torch.ones((), device=p.device, dtype=p.dtype)

    # UN RAYON UNIQUE SUPPOSE L'OBJET AUSSI EPAIS QUE LARGE, ET C'EST FAUX.
    #
    # Mesure du 9 septembre 2026 sur le feu tricolore -- un panneau large et
    # plat, photographie aux quatre angles ronds, donc des donnees parfaites.
    # Une cellule du panneau ARRIERE a un |z| petit (l'objet est mince) mais
    # un |x| qui peut aller jusqu'au bord (l'objet est large). Divises par le
    # meme rayon, le score de la vue de dos reste minuscule et celui d'un
    # profil devient enorme :
    #
    #     au centre du panneau (x=0)        vue de dos      52 %
    #     pres du bord (x=0,2)              vue de dos      18 %
    #                                       vue de DROITE   78 %
    #
    # La vue de droite rase ce panneau et n'en sait rien. Comme un panneau
    # large a l'essentiel de son aire pres des bords, la majorite du dos etait
    # decidee par des vues qui ne le voient pas : le dos s'enfonce, un
    # eventail le capote, la cuisson ne voit jamais ces faces -- elles sortent
    # noires. Quentin : « magnifique de face, et de dos, gros trou ».
    #
    # ON RAMENE DONC CHAQUE AXE A SA PROPRE ETENDUE avant le produit scalaire.
    # Une plaque se comporte alors comme une boule : la direction compte, pas
    # l'aplatissement. Sur un sujet deja rond les deux etendues sont egales et
    # rien ne change -- c'est ce qui protege le buste de pierre, seul sujet ou
    # cette ponderation a fait ses preuves.
    demi = 0.5 * (p.max(dim=0).values - p.min(dim=0).values)
    echelle = torch.tensor(
        [max(float(demi[0]), 1e-6), 1.0, max(float(demi[2]), 1e-6)],
        device=p.device, dtype=p.dtype)
    echelle = echelle / echelle[[0, 2]].max()
    p = p / echelle

    scores = []
    for role in ROLES:
        # Le poids suit la caméra : si on a mesuré d'où la photo vient, c'est
        # cet angle-là qui dit ce qu'elle voit.
        vu = (azimuts or {}).get(role)
        lacet = math.radians(AZIMUTHS[_role_projete(role, parite)]
                             if vu is None else vu)
        # La direction de la caméra vue du centre. Une cellule devant elle
        # donne un produit scalaire positif ; une cellule cachée derrière
        # l'objet, négatif.
        vers = torch.tensor([math.sin(lacet), 0.0, math.cos(lacet)],
                            device=coords.device, dtype=p.dtype)
        scores.append((p * vers).sum(-1) / rayon)
    return torch.softmax(force * torch.stack(scores), dim=0)


def volume(config, traits: dict, poids, graine: int, parite: dict,
           recette: dict, geometrie: dict | None = None,
           azimuts: dict | None = None, force: float | None = None,
           pas: int | None = None):
    """Fabriquer le volume de cellules à partir des traits des quatre vues.

    `traits` porte, par rôle, le tableau des jetons d'image en 512.
    Rend les coordonnées occupées sur la grille de 32, en tableau numpy.

    La carte est rendue entre les deux modèles : le modèle de flux d'abord, le
    décodeur ensuite. C'est ce qui permet à tout ceci de tenir sur 8 Go.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    models = config.trellis("models")
    samplers = config.trellis("pipelines.samplers.flow_euler")
    geometrie = geometrie or {}

    # La grille de départ : 16 au cube, en coordonnées entières, avec un
    # indice de lot en tête comme le veut le format creux.
    axe = torch.arange(16)
    xyz = torch.stack(torch.meshgrid(axe, axe, axe, indexing="ij"), -1).reshape(-1, 3)
    coords = torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int64), xyz), 1).int().cuda()

    f = FORCE_VISIBILITE if force is None else force
    poids_vue = poids_des_vues(coords, parite, f, 16, azimuts=azimuts)
    if f:
        notes = "pondération par visibilité, force %.2f" % f
    else:
        notes = "moyenne plate"

    projete = globaux = None
    for indice, role in enumerate(ROLES):
        jetons = torch.from_numpy(traits[role]).cuda()
        if jetons.shape != (1, 1029, 1024):
            raise ValueError(
                "View %s: expected 1029 tokens of 1024 channels, got %s"
                % (role, tuple(jetons.shape)))
        # AVEC LES ANGLES MESURES, LA PARITE NE SERT PLUS : un azimut de 306
        # degrés dit déjà que cette photo est du côté gauche. La bascule
        # gauche/droite n'est que la version binaire de la même information,
        # et l'appliquer en plus la compterait deux fois.
        grille = nominal_queries(
            coords, role if azimuts else _role_projete(role, parite),
            grid_resolution=16, image_resolution=512,
            image_transform=geometrie.get(role),
            azimut=(azimuts or {}).get(role))
        # Les cinq premiers jetons sont globaux ; les 1024 suivants forment
        # la carte 32 par 32 qu'on échantillonne.
        carte = jetons[:, 5:].reshape(1, 32, 32, 1024).permute(0, 3, 1, 2).contiguous()
        vue = F.grid_sample(carte, grille[None, :, None], align_corners=False,
                            padding_mode="border")[0, :, :, 0].T[None]
        # LE POIDS S'APPLIQUE AUX TRAITS PROJETÉS, PAS AUX JETONS GLOBAUX. Les
        # cinq jetons globaux disent CE QU'EST l'objet, pas où sont ses
        # morceaux : les pondérer par position n'aurait aucun sens, et les
        # quatre vues sont censées voir le même objet.
        vue = vue * poids_vue[indice].view(1, -1, 1)
        projete = vue.clone() if projete is None else projete + vue
        globaux = jetons[:, :5].clone() if globaux is None else globaux + jetons[:, :5]
        del jetons, grille, carte, vue

    # Les poids somment déjà à 1 sur les vues : la somme pondérée EST la
    # moyenne, il n'y a rien à diviser. Les jetons globaux, eux, restent une
    # moyenne ordinaire.
    globaux.div_(len(ROLES))
    del coords, poids_vue
    gc.collect()
    torch.cuda.empty_cache()

    # Le modèle de flux, construit sur `meta` puis rempli : c'est ce qui évite
    # d'allouer deux fois ses poids au chargement.
    params = json.loads(Path(poids.structure).with_suffix(".json").read_text())["args"]
    with torch.device("meta"):
        modele = models.SparseStructureFlowModel(**params)
    modele.load_state_dict(load_file(str(poids.structure)), strict=True, assign=True)
    materialize_rotary_frequencies(modele)
    if modele.resolution != 16:
        raise RuntimeError("Structure model resolution is %s, expected 16"
                           % modele.resolution)
    modele = modele.eval().cuda()

    torch.manual_seed(graine)
    bruit = torch.randn(1, 8, 16, 16, 16).cuda()
    positif = {"global": globaux, "proj": projete}
    negatif = {k: torch.zeros_like(v) for k, v in positif.items()}
    with torch.inference_mode():
        # `.samples` : l'echantillonneur rend un objet qui porte aussi ses
        # etats intermediaires. C'est le champ qu'on veut, pas l'objet.
        latent = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5).sample(
            modele, bruit, cond=positif, neg_cond=negatif,
            **(recette["structure"] if pas is None
               else dict(recette["structure"], steps=pas)),
            dino_lock=0, verbose=False).samples
    torch.cuda.synchronize()

    del modele, positif, negatif, globaux, projete, bruit
    gc.collect()
    torch.cuda.empty_cache()

    # Le décodeur rend une occupation à 64 ; on la ramène à 32 par un maximum
    # local, qui est la résolution que la cascade de forme attend.
    decodeur = models.from_pretrained(str(
        config.MODELS_ROOT / "microsoft" / "TRELLIS-image-large" / "ckpts"
        / "ss_dec_conv3d_16l8_fp16")).eval().cuda()
    with torch.inference_mode():
        occupe = decodeur(latent) > 0
        if tuple(occupe.shape) != (1, 1, 64, 64, 64):
            raise RuntimeError("Structure decoder returned %s" % (tuple(occupe.shape),))
        reduit = F.max_pool3d(occupe.float(), 2, 2) > 0.5
        cellules = torch.argwhere(reduit)[:, [0, 2, 3, 4]].int().contiguous()
    torch.cuda.synchronize()

    sortie = cellules.cpu().numpy()
    if not len(sortie):
        raise RuntimeError(
            "The four views produced an empty volume. Nothing is salvageable "
            "downstream: check that the photos show the same object.")
    if sortie[:, 1:].max() >= 32:
        raise RuntimeError("Cell outside the 32 grid")

    del decodeur, latent, occupe, reduit, cellules
    gc.collect()
    torch.cuda.empty_cache()
    return sortie
