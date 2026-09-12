# -*- coding: utf-8 -*-
"""Le banc à VÉRITÉ TERRAIN : comparer la forme rendue au modèle d'origine.

POURQUOI CELUI-CI EN PLUS DE `banc.py`. `banc.py` compte ce qui se recompte
dans le fichier livré — faces, bords ouverts, morceaux. C'est utile et ça ne
suffit pas : il ne sait pas dire si la FORME est juste. Le 9 septembre 2026,
deux indicateurs indirects m'ont fait conclure de travers dans la même
journée :

  - le compte de faces retournées donnait un excellent score (0,43 %) à un dos
    de feu tricolore parfaitement lisse, vide, sans une seule trappe ;
  - une pondération jugée la meilleure de toutes sur une mesure hors ligne
    s'est révélée la PIRE au banc (0,298 contre 0,377 pour celle qu'elle
    devait remplacer).

Ici il n'y a plus de proxy. On part d'un maillage CONNU, on prend les quatre
rendus à 0, 90, 180 et 270 degrés que la planche de démonstration embarque
déjà, on les repasse au moteur, et on compare le résultat AU MAILLAGE
D'ORIGINE, en 3D. Les quatre vues étant parfaitement cohérentes entre elles,
tout écart est de notre fait : ni photo bancale, ni angle approximatif, ni
désaccord de proportions entre deux clichés.

    py -3 -m localmesh_engine.banc.banc3d resultats
    py -3 -m localmesh_engine.banc.banc3d resultats volet lampion

    BANC3D_PALIER=standard   le palier (défaut : draft, ~95 s par sujet)
    BANC3D_MONO=1            le contrôle à une seule photo

CE QUE LA MESURE VAUT, ET CE QU'ELLE NE VAUT PAS. La distance de Chamfer est
donnée en pour-cent de la diagonale. Deux repères mesurés le 9 septembre :

    un maillage contre LUI-MÊME          0,27 %   (le bruit d'échantillonnage)
    deux objets sans rapport             3,8 à 7,3 %

Entre les deux, le chiffre veut dire quelque chose. MAIS il compare des boîtes
englobantes normalisées : sur `lanterne`, dont la scène contient une lanterne
ET une jardinière, le moteur ne rend que la lanterne et la grossit pour
remplir le cadre. L'écart de 10 % qu'on lit alors dit surtout ça. À sujet
FIXE, en revanche, comparer deux réglages reste parfaitement valide : même
vérité, mêmes vues, une seule chose change.

CE BANC NE PEUT PAS DÉPARTAGER UNE PHOTO DE QUATRE, ET C'EST SA LIMITE LA PLUS
IMPORTANTE. Ses maillages de vérité portent la signature de LocalMesh — atlas
2048/4096, `doubleSided` faux, 57 896 / 59 756 / 229 150 / 248 506 faces, soit
exactement les budgets des paliers. Ce sont des objets QUE LE MOTEUR A
LUI-MÊME FABRIQUÉS, par la voie à une photo. Lui redemander de les reproduire
depuis leurs propres rendus, c'est l'interroger sur sa propre production : il
part gagnant, et le multivue — dont les modèles de structure et de forme sont
d'une autre famille — part perdant.

La mesure du 9 septembre le montre bien : moyenne 3,85 % à une photo contre
4,17 % à quatre, la voie à une photo gagnant sur cinq sujets sur sept. LA
CONCLUSION EST FAUSSE, et elle a failli passer. Sur de VRAIES photos d'objets
réels, mesurées le même jour, les quatre vues gagnent partout :

    feu tricolore, dos      0,352 à une photo   contre 0,476 à quatre
    samouraï, silhouettes   0,689               contre 0,711
    buste de pierre         deux bouches        contre une

CE QUE CE BANC MESURE VRAIMENT, ET TRÈS BIEN : deux RÉGLAGES à sujet fixe. Là
il n'y a aucune circularité — même vérité, mêmes vues, une seule chose change.
C'est ainsi qu'ont été écartés en une journée la pondération des vues, le
transport de cadre des caméras et le doublement des pas d'échantillonnage :
tous les trois inertes, à moins de 0,01 % de la référence.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

#: Les sujets de la planche de démonstration qui embarquent À LA FOIS les
#: quatre rendus et leur maillage. Ils vivent dans le dépôt : ce banc-ci n'a
#: donc pas besoin qu'on lui apporte des photos, contrairement à `banc.py`.
DEMO = (Path(__file__).resolve().parents[3]
        / "app" / "public" / "planche-demo" / "sujets")

#: La graine est fixée par sujet : deux mesures qui ne partagent pas leur
#: graine ne se comparent pas.
SUJETS = {
    "volet":     ("cremieux_volet", 11),      # plat et mince
    "porte":     ("cremieux_porte26", 12),    # plat et épais
    "pot":       ("cremieux_pot", 13),        # rond, le cas facile
    "lanterne":  ("cremieux_lanterne", 14),   # ajouré, DEUX objets dans la scène
    "lanterne2": ("japon_lanterne2", 15),     # ajouré
    "enseigne":  ("japon_enseigne", 16),      # panneau et potence
    "lampion":   ("japon_lampion", 17),       # rond et côtelé
}

NB_POINTS = 60000


def _points(chemin, graine=0):
    """Points de surface et normales, ramenés à une boîte unité centrée."""
    import trimesh

    scene = trimesh.load(chemin, process=False)
    m = (trimesh.util.concatenate(list(scene.geometry.values()))
         if hasattr(scene, "geometry") else scene)
    v = np.asarray(m.vertices, np.float64)
    centre = (v.max(0) + v.min(0)) / 2
    echelle = float((v.max(0) - v.min(0)).max())
    m.vertices = (v - centre) / max(echelle, 1e-9)
    p, idx = trimesh.sample.sample_surface(m, NB_POINTS, seed=graine)
    return np.asarray(p), np.asarray(m.face_normals[idx])


def _quart_de_tour(p, n, k):
    for _ in range(k % 4):
        p = np.stack([p[:, 2], p[:, 1], -p[:, 0]], 1)
        n = np.stack([n[:, 2], n[:, 1], -n[:, 0]], 1)
    return p, n


def comparer(rendu, verite) -> dict:
    """Chamfer symétrique (% de la diagonale) et accord des normales.

    LE MEILLEUR DES QUATRE QUARTS DE TOUR EST RETENU, et l'écart au quart de
    tour zéro est rendu à côté : le moteur peut sortir l'objet tourné sans que
    ce soit un défaut de forme, mais si les deux chiffres divergent, c'est une
    information et non un détail à cacher.

    L'accord des normales ne se calcule que sur les points BIEN APPARIÉS. Il
    dit si la surface est orientée comme il faut là où elle est au bon
    endroit : un dos enfoncé s'y voit même quand la silhouette ne bouge pas.
    """
    from scipy.spatial import cKDTree

    pa, na = _points(rendu, 1)
    pb, nb = _points(verite, 2)
    arbre_b = cKDTree(pb)
    resultats = []
    for k in range(4):
        p, n = _quart_de_tour(pa, na, k)
        arbre_a = cKDTree(p)
        da, ia = arbre_b.query(p)
        db, _ = arbre_a.query(pb)
        chamfer = float((da.mean() + db.mean()) / 2)
        proches = da < 0.02
        accord = (float(np.abs((n[proches] * nb[ia[proches]]).sum(1)).mean())
                  if proches.any() else 0.0)
        resultats.append((chamfer, accord, k))
    resultats.sort()
    chamfer, accord, k = resultats[0]
    sans_rotation = [r for r in resultats if r[2] == 0][0][0]
    return {"chamfer_pct": round(100 * chamfer, 3),
            "normales": round(accord, 3),
            "quart_de_tour": k,
            "chamfer_sans_rotation_pct": round(100 * sans_rotation, 3)}


def _generer(nom, dossier, sortie):
    from ..pipeline import Engine, GenerateSettings

    d = DEMO / dossier
    vues = {"droite": d / "mesh_090.png", "dos": d / "mesh_180.png",
            "gauche": d / "mesh_270.png"}
    if os.environ.get("BANC3D_MONO") == "1":
        vues = {}
    reglages = GenerateSettings(
        images=[d / "mesh_000.png"], vues=vues or None,
        # LA LATÉRALITÉ EST IMPOSÉE : les quatre vues sont à leurs angles ronds
        # PAR CONSTRUCTION. Laisser la mesure en décider ajouterait une
        # variable qui n'a rien à voir avec ce qu'on compare.
        lateralite="normale",
        detail=os.environ.get("BANC3D_PALIER", "draft"),
        seed=SUJETS[nom][1])
    depart = time.time()
    r = Engine().generate(reglages, sortie)
    return r.glb_path, round(time.time() - depart, 1), r.faces, r.peak_vram_gb


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage : banc3d <dossier de sortie> [sujet ...]", file=sys.stderr)
        return 2
    vers = Path(argv[0]); vers.mkdir(parents=True, exist_ok=True)
    voulus = argv[1:] or list(SUJETS)

    lignes = []
    for nom in voulus:
        dossier, _ = SUJETS[nom]
        verite = DEMO / dossier / "objet.glb"
        if not verite.is_file():
            print("pas de vérité pour %s" % nom, file=sys.stderr)
            continue
        sortie = vers / nom
        sortie.mkdir(parents=True, exist_ok=True)
        try:
            glb, duree, faces, vram = _generer(nom, dossier, sortie)
        except Exception as exc:                              # noqa: BLE001
            # UN PLANTAGE CUDA EMPOISONNE LE CONTEXTE, et tout ce qui suit
            # échoue aussi : mesure du 9 septembre, la lanterne a fait tomber
            # les quatre sujets suivants d'une série. On sort franchement,
            # l'appelant relance dans un processus neuf.
            ligne = {"sujet": nom, "erreur": repr(exc)[:300]}
            (vers / ("%s.json" % nom)).write_text(
                json.dumps(ligne, indent=1, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(ligne, ensure_ascii=False), flush=True)
            return 3
        ligne = {"sujet": nom, "duree_s": duree, "faces": faces,
                 "pic_vram_go": vram}
        ligne.update(comparer(glb, verite))
        lignes.append(ligne)
        (vers / ("%s.json" % nom)).write_text(
            json.dumps(ligne, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(ligne, ensure_ascii=False), flush=True)

    if lignes:
        print()
        print("| sujet | chamfer % | normales | quart | durée |")
        print("|---|---|---|---|---|")
        for l in lignes:
            print("| %s | %.3f | %.3f | %d | %.0f s |"
                  % (l["sujet"], l["chamfer_pct"], l["normales"],
                     l["quart_de_tour"], l["duree_s"]))
        print("\nMOYENNE chamfer %.3f %%   normales %.3f"
              % (np.mean([l["chamfer_pct"] for l in lignes]),
                 np.mean([l["normales"] for l in lignes])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
