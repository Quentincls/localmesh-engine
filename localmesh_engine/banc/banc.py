# -*- coding: utf-8 -*-
"""Le banc : les mêmes sujets, les mêmes graines, le même tableau.

C'est la seule preuve acceptée par ce dépôt. Une proposition qui améliore la
fidélité, la vitesse ou la mémoire arrive avec la sortie de ce script, avant
et après.

    python -m localmesh_engine.banc.banc --sujets banc/sujets --vers resultats/
    python -m localmesh_engine.banc.banc --palier high --vers resultats/

Il ne mesure QUE ce qui se vérifie : ce que le moteur rapporte lui-même, et ce
qui se recompte dans le fichier livré. Aucune note subjective, aucun score de
ressemblance inventé.

LES BORDS OUVERTS SE COMPTENT APRÈS SOUDURE DES SOMMETS PAR POSITION. Le
dépliage UV duplique les sommets le long de chaque couture : sur les indices
bruts, un objet parfaitement fermé affiche jusqu'à 39 % d'arêtes de bord. La
faute a été commise, mesurée, corrigée ; le compte ci-dessous soude d'abord.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

#: Les sujets de référence. Une photo de face, et les trois côtés quand ils
#: existent. La graine est fixée : deux mesures qui ne partagent pas leur
#: graine ne se comparent pas.
SUJETS = [
    {"nom": "samourai_mono", "face": "samourai_face.webp", "graine": 42},
    {"nom": "feu_mono", "face": "feu_face.png", "graine": 7},
    {"nom": "samourai_mv", "face": "samourai_face.webp", "graine": 42,
     "vues": {"droite": "samourai_droite.webp", "gauche": "samourai_gauche.webp",
              "dos": "samourai_dos.webp"}},
    {"nom": "feu_mv", "face": "feu_face.png", "graine": 7,
     "vues": {"droite": "feu_droite.png", "gauche": "feu_gauche.png",
              "dos": "feu_dos.png"}},
    # LE BUSTE DE PIERRE, ET CE QU'IL ATTRAPE QUE LES DEUX AUTRES NE VOIENT
    # PAS : le DEDOUBLEMENT DES TRAITS FINS. Ses quatre photos ne s'accordent
    # pas sur les proportions internes du visage — ou tombe le nez dans le
    # visage — et une fusion qui moyenne a plat sculpte les deux reponses :
    # deux nez empiles, deux levres, vus a l'oeil par Quentin le 8 septembre
    # 2026. C'est le sujet qui a motive la ponderation par visibilite, et donc
    # le seul qui puisse dire si elle merite encore sa place.
    {"nom": "buste_mv", "face": "buste_face.png", "graine": 4,
     "vues": {"droite": "buste_droite.png", "gauche": "buste_gauche.png",
              "dos": "buste_dos.png"}},
]


def topologie(glb: Path) -> dict:
    """Ce qu'on recompte dans le fichier livré, sans croire le moteur."""
    import numpy as np
    import trimesh

    scene = trimesh.load(glb, process=False)
    g = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    v = np.asarray(g.vertices, dtype=np.float64)
    f = np.asarray(g.faces)

    # La soudure, sans laquelle chaque couture UV passe pour un trou.
    _, inverse = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    soude = inverse[f]
    aretes = np.sort(np.stack([soude[:, [0, 1]], soude[:, [1, 2]],
                               soude[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
    _, comptes = np.unique(aretes, axis=0, return_counts=True)

    materiau = getattr(getattr(g, "visual", None), "material", None)
    mr = getattr(materiau, "metallicRoughnessTexture", None)
    metal = float(np.asarray(mr)[..., 2].mean()) / 255.0 if mr is not None else None

    # LES MORCEAUX, ET POURQUOI ILS COMPTENT AUTANT QUE LES BORDS OUVERTS.
    #
    # Le 9 septembre 2026, le samouraï livré était FERMÉ — zéro bord ouvert —
    # et Quentin y voyait un trou. Il avait raison : sous la cape flottaient
    # 1 133 morceaux détachés, dont 1 102 de moins de 50 faces. Un maillage
    # peut être irréprochable en topologie et sale à l'œil. Le banc comptait
    # la première chose et pas la seconde ; il compte les deux maintenant.
    morceaux = eclats = None
    try:
        aretes_par_face = soude
        parent = np.arange(int(inverse.max()) + 1)

        def trouver(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for tri in aretes_par_face:
            r0 = trouver(tri[0])
            for autre in tri[1:]:
                r1 = trouver(autre)
                if r0 != r1:
                    parent[r1] = r0
        racines = np.array([trouver(i) for i in range(len(parent))])
        rac_face = racines[soude[:, 0]]
        _, tailles = np.unique(rac_face, return_counts=True)
        morceaux = int(len(tailles))
        eclats = int((tailles < 50).sum())
    except Exception:                                          # noqa: BLE001
        pass

    return {
        "faces": int(len(f)),
        "sommets_soudes": int(inverse.max() + 1),
        "aretes": int(len(comptes)),
        "bords_ouverts": int((comptes == 1).sum()),
        "non_manifold": int((comptes > 2).sum()),
        "morceaux": morceaux,
        "eclats_moins_de_50_faces": eclats,
        "double_face": bool(getattr(materiau, "doubleSided", False)),
        "metal_moyen": metal,
        "octets": glb.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Le banc de LocalMesh Engine.")
    p.add_argument("--sujets", type=Path, default=Path(__file__).parent / "sujets",
                   help="le dossier des photos de référence")
    p.add_argument("--vers", type=Path, default=Path("resultats"),
                   help="où déposer les objets et le tableau")
    p.add_argument("--palier", default="standard",
                   choices=["draft", "standard", "high", "max"])
    p.add_argument("--seulement", default=None,
                   help="ne lancer qu'un sujet, par son nom")
    a = p.parse_args(argv)

    from ..pipeline import Engine, GenerateSettings

    a.vers.mkdir(parents=True, exist_ok=True)
    moteur = Engine()
    lignes = []

    for sujet in SUJETS:
        if a.seulement and sujet["nom"] != a.seulement:
            continue
        face = a.sujets / sujet["face"]
        if not face.is_file():
            print("sujet absent, ignoré : %s" % face, file=sys.stderr)
            continue
        vues = {k: a.sujets / v for k, v in (sujet.get("vues") or {}).items()}
        if any(not c.is_file() for c in vues.values()):
            print("côtés absents, sujet ignoré : %s" % sujet["nom"], file=sys.stderr)
            continue

        sortie = a.vers / sujet["nom"]
        sortie.mkdir(parents=True, exist_ok=True)
        depart = time.time()
        resultat = moteur.generate(
            GenerateSettings(images=[face], vues=vues, detail=a.palier,
                             seed=sujet["graine"]),
            sortie)
        ligne = {
            "sujet": sujet["nom"],
            "palier": a.palier,
            "graine": resultat.seed,
            "duree_s": round(time.time() - depart, 1),
            "pic_vram_go": resultat.peak_vram_gb,
        }
        ligne.update(topologie(Path(resultat.glb_path)))
        lignes.append(ligne)
        print(json.dumps(ligne, ensure_ascii=False), flush=True)
        (a.vers / "banc.json").write_text(
            json.dumps(lignes, indent=1, ensure_ascii=False), encoding="utf-8")

    if not lignes:
        print("aucun sujet lancé : le dossier des photos est-il en place ?",
              file=sys.stderr)
        return 1

    # Le tableau, celui qu'on colle dans une proposition.
    colonnes = ["sujet", "faces", "duree_s", "pic_vram_go", "bords_ouverts",
                "morceaux", "eclats_moins_de_50_faces", "non_manifold",
                "double_face", "metal_moyen"]
    print()
    print("| " + " | ".join(colonnes) + " |")
    print("|" + "|".join(["---"] * len(colonnes)) + "|")
    for l in lignes:
        print("| " + " | ".join(str(l.get(c)) for c in colonnes) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
