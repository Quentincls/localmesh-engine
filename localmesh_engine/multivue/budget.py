# -*- coding: utf-8 -*-
"""Ramener une forme trop lourde sous le plafond, sans changer de palier.

La chaîne à une photo se replie d'un palier quand la forme brute dépasse ce
que la carte porte : la même demande passe en plus léger, et la note le dit.
C'est le bon geste là-bas, parce que la forme vient d'une seule image et que
tout le travail est à refaire de toute façon.

Ici il est mauvais. Le multivue a déjà encodé quatre vues, fusionné un volume
et descendu deux cascades ; se replier jetterait tout cela pour recommencer
plus grossièrement, alors que le dépassement est souvent de quelques pour
cent. Sur le feu tricolore, la forme sort à 8 234 430 faces contre un plafond
de 7 995 605 : trois pour cent de trop, et un palier perdu.

On réduit donc la forme au lieu de la refaire, et on le dit dans les notes.
La réduction est un pis-aller assumé, pas une amélioration : elle enlève des
triangles là où la surface en supporte le moins, et le maillage qui en sort
n'est pas celui qu'une carte plus grande aurait rendu.
"""
from __future__ import annotations


#: On vise un peu sous le plafond plutôt que dessus : la décimation atteint
#: rarement sa cible au triangle près, et repasser au-dessus relancerait le
#: repli qu'on cherche justement à éviter.
MARGE = 0.97


def reduire_si_trop_lourd(mesh, plafond: int, notes: list | None = None):
    """Décimer le maillage brut s'il dépasse, et rendre le compte.

    Rend `(mesh, reduit)`. Sans dépassement, rien n'est touché et `reduit`
    vaut faux : c'est le cas ordinaire, et il ne coûte rien.
    """
    import torch

    depart = int(mesh.faces.shape[0])
    if depart <= plafond:
        return mesh, False

    # La même décimation que le nettoyage utilise ensuite, appelée plus tôt :
    # une seule implémentation, un seul comportement à connaître.
    cible = int(plafond * MARGE)
    with torch.inference_mode():
        mesh.simplify_with_cumesh(target=cible, verbose=False)
    arrivee = int(mesh.faces.shape[0])

    if notes is not None:
        notes.append(
            "Forme réduite de %s à %s faces pour tenir sur la carte, sans "
            "changer de palier." % (f"{depart:n}", f"{arrivee:n}"))
    return mesh, True
