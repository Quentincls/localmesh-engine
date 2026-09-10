"""La chaîne du banc, sans ComfyUI : nettoyer, remailler, décimer, texturer.

CE QUE CE MODULE CORRIGE. Le produit texturait avec `o_voxel.to_glb`, qui
déplie le maillage et cuit dans l'atlas les couleurs d'un VOLUME déjà
échantillonné. Sur un sujet complexe, ~293 000 faces dans un atlas de 2048
font 21 texels par triangle : la texture n'a plus assez de place pour dire
autre chose qu'une mosaïque. Ici la texture est GÉNÉRÉE pour le maillage
final, après décimation, dans un atlas de 4096 : 202 texels par triangle au
banc, dix fois plus.

L'ORDRE COMPTE, ET IL N'EST PAS ARBITRAIRE. Boucher, remailler, retirer les
débris, décimer — PUIS texturer. La décimation a lieu AVANT le dépliage UV
(que `postprocess_mesh` fait lui-même) : il n'y a donc plus de coutures
d'atlas à souder après coup, et le budget de triangles cesse d'être la cause
de la mosaïque que `pipeline._finish_scene` documente.

Les quatre nœuds de nettoyage du banc sont des enveloppes minces : chacun
fait un `copy.deepcopy` puis UN appel. Le deepcopy n'existe que parce qu'un
nœud ne doit pas muter l'entrée d'un graphe ; ici il ne dupliquerait que des
tenseurs CUDA pour rien — le MeshWithVoxel porte `attrs` et `coords`, le
volume entier. On travaille en place.

DEUX ÉTAPES DU BANC NE SONT PAS PORTABLES, et ne le seront pas : son retrait
de débris passe par pymeshlab (GPL3) et son second bouchage par meshlib
(licence non commerciale). LocalMesh est vendu. Les remplaçants sont dans
cumesh, qui est MIT et déjà installé.
"""
from __future__ import annotations

import gc
import logging

import numpy as np
import torch

log = logging.getLogger("localmesh_engine.remaillage")

#: QUAND LE CONTOURAGE DUAL DOIT DESCENDRE A 512.
#:
#: Sa memoire ne suit PAS le nombre de triangles en entree : elle suit le
#: nombre de cellules de la grille `dc_resolution` au cube que la SURFACE
#: traverse. Mesure du 29 aout 2026 : ramener l'entree de 7,12 M a 5,83 M
#: faces n'a RIEN change, le mur etait identique. Decimer enleve des
#: triangles, pas des cellules.
#:
#: Le mur, lu a la telemetrie du moteur pendant l'etape : `process_gb` a
#: 8,449 Go sur une carte de 7,996. L'excedent part en memoire partagee
#: Windows et l'etape passe de 60-90 secondes a plusieurs heures. Et
#: `reconstruct_mesh_dc_quad` est un seul appel sans point de controle
#: interne : pendant tout ce temps, « Annuler » ne peut RIEN. Un client sur
#: 8 Go attend sans recours et n'obtient pas son objet. C'est un garde-fou,
#: pas une optimisation.
#:
#: 512 divise la grille par huit. C'est deja le chemin du palier Leger
#: (`pipeline_type="512"`), donc il est eprouve.
#:
#: Seuil pose entre les deux populations mesurees le 29 aout : 5,21 / 5,32 /
#: 5,63 / 6,19 M passent en 56 a 91 s ; 7,12 et 7,49 M partent dans le mur.
#: 6,5 M ne touche donc aucun sujet qui fonctionnait.
#: LA REGLE EST GRADUEE, PAS BINAIRE. Tomber de 1024 a 512 d'un coup coute
#: de la finesse a un sujet qui ne depasse le seuil que de dix pour cent —
#: constate a l'oeil par Quentin le 29 aout sur un sujet a 7,14 M : « le
#: maillage a l'air ok, mais pas tres precis quand meme ». 768 divise la
#: grille par 2,4 au lieu de 8, ce qui suffit largement au mur mesure
#: (8,4 Go demandes a 1024, environ 3,5 a 768).
#: (Table historique, en triangles : >6,5 M -> 768, >8 M -> 512. Elle ne
#: decide plus rien depuis le 2 septembre 2026 ; voir `_budget_cellules`.)

#: LA MEME REGLE, MAIS SUR LA BONNE GRANDEUR.
#:
#: Le commentaire ci-dessus le dit lui-meme : la memoire suit les CELLULES
#: que la surface traverse, pas les triangles en entree. Or le seuil se
#: declenchait sur les triangles. Mesure le 2 septembre 2026 sur 52
#: remaillages du journal : correlation(faces en entree, cellules) = +0,76
#: seulement, avec des inversions franches —
#:
#:     23 690 774 faces -> 4,03 M cellules a 1024 -> RETROGRADE a 512
#:      5 942 896 faces -> 7,86 M cellules a 1024 -> laisse a 1024, 65 s
#:
#: Le sujet le plus lourd en triangles a ete coupe au plus dur alors qu il
#: coutait deux fois moins de memoire qu un autre laisse passer sans
#: incident. Sur 7 retrogradations mesurables, 3 etaient inutiles — et ce
#: sont les deux plus severes. Chaque retrogradation DOUBLE la dilatation du
#: contourage (une cellule d offset : 1,07 pour mille de la boite a 1024,
#: 2,15 a 512) : c est le « pas tres precis quand meme » du 29 aout.
#:
#: On compte donc les cellules : quantifier les sommets sur une grille de
#: 256 au cube (un `torch.unique`, quelques dizaines de millisecondes),
#: compter les occupees, extrapoler en (R / 256) au carre — une surface
#: croit au carre de la resolution. Seuil a 8 M cellules a 1024 : le plus
#: haut cout observe SANS incident (7,86 M et 7,80 M). C est une borne de
#: securite calee sur le cote qui passe ; aucune generation partie dans le
#: mur n a journalise sa ligne de cellules.
#:
#: LA TABLE EN TRIANGLES RESTE, EN PLAFOND DE SECOURS SEULEMENT : elle ne
#: retrograde plus que si les cellules la confirment. Un sujet dense en
#: triangles mais peu ajoure passe desormais a 1024.
#: (Table intermediaire, en cellules a 1024 : >8 M -> 768, >14 M -> 512.
#: Remplacee le meme jour par le budget proportionnel ci-dessous, qui lui
#: est identique sur 8 Go et s etend aux autres cartes.)

#: SUR TOUTES LES CARTES, ET PROPORTIONNEL A LEUR MEMOIRE (2 septembre 2026).
#:
#: Le garde-fou ne jouait que sous 20 Go. Avec Extreme (grille 1536, 3,4 fois
#: plus de cellules), une carte de 24 Go peut se retrouver exactement dans
#: la situation d une carte de 8 Go devant le masque a 1024. Doctrine de
#: Quentin : les regles qui protegent les petites cartes valent pour les
#: grandes. Le budget se lit dans la table ci-dessus : 8 M de cellules
#: passent sur 8 Go, et 14 M a 1024 valent 7,9 M a 768 — c est le MEME
#: budget, 1 M de cellules par gigaoctet, applique a chaque grille. On
#: choisit donc la grille la plus fine de l echelle dont le cout,
#: cellules x (grille / 1024)^2, tient dans ce budget.
_CELLULES_PAR_GO = 1_000_000
_ECHELLE_GRILLES = (1536, 1024, 768, 512)


def _budget_cellules() -> int:
    """Cellules (comptees a 1024) que la carte porte sans deborder."""
    try:
        go = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
              if torch.cuda.is_available() else 8.0)
    except Exception:                                         # noqa: BLE001
        go = 8.0
    return int(_CELLULES_PAR_GO * go)


def _grille_tenable(cellules_1024: int, dc_resolution: int) -> int:
    """La grille la plus fine, au plus `dc_resolution`, qui tient dans le
    budget de la carte. Rend `dc_resolution` telle quelle si elle tient."""
    budget = _budget_cellules()
    for grille in _ECHELLE_GRILLES:
        if grille > dc_resolution:
            continue
        if cellules_1024 * (grille / 1024) ** 2 <= budget:
            return grille
    return _ECHELLE_GRILLES[-1]
_GRILLE_COMPTAGE = 256


def _cellules_estimees(v, resolution: int) -> int:
    """Cellules de la grille `resolution` au cube que la surface traverse,
    estimees depuis un comptage a 256 puis extrapolees au carre."""
    vv = v.detach()
    lo = vv.min(dim=0).values
    hi = vv.max(dim=0).values
    etendue = (hi - lo).max().clamp_min(1e-9)
    q = ((vv - lo) / etendue * (_GRILLE_COMPTAGE - 1)).long()
    occupees = int(torch.unique(q, dim=0).shape[0])
    return int(occupees * (resolution / _GRILLE_COMPTAGE) ** 2)

def purger() -> None:
    """L'équivalent du `Trellis2CudaReset` du banc.

    Le graphe du banc en pose un sur le lien qui porte le trimesh vers la
    texturation, avec pour tout commentaire « purge juste avant le texturing :
    c'est là que le pic arrive ». Le `synchronize()` vient en premier : sans
    lui, `empty_cache()` rend des blocs que des noyaux en vol utilisent
    encore, et l'allocateur les reprend aussitôt.
    """
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 1 à 4 : bouchage, remaillage, débris, décimation. EN PLACE sur le
# MeshWithVoxel que rend `run()`.
# --------------------------------------------------------------------------
def _retirer_debris(v, f, *, ratio_aire: float, distance_mini: float,
                    part_maxi: float, notes=None):
    """Jeter les éclats détachés, et EUX SEULS.

    Un morceau part s'il est à la fois petit devant le plus gros morceau et
    décollé de lui. La mesure qui impose ces deux conditions est écrite en
    tête de `config.DEBRIS_RATIO_AIRE` ; en deux mots : sur le samouraï,
    1 133 morceaux, dont 941 posés à plat SUR la coque. Les jeter parce
    qu'ils sont petits ouvrirait 941 trous dans la surface.

    Rend (sommets, faces). NE LÈVE JAMAIS : sur un maillage d'une seule
    pièce, ou si la mesure de distance échoue, il rend son entrée telle
    quelle. Un nettoyage qui échoue ne doit pas coûter l'objet.
    """
    import cumesh

    cm = cumesh.CuMesh()
    cm.init(v.contiguous().float(), f.contiguous().int())
    cm.get_connected_components()
    nombre, etiquettes = cm.read_connected_components()
    del cm
    if nombre is None or int(nombre) <= 1:
        return v, f
    etiquettes = etiquettes.long()

    tri = v[f.long()]
    aire_face = 0.5 * torch.cross(tri[:, 1] - tri[:, 0],
                                  tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    aire = torch.zeros(int(nombre), device=v.device, dtype=aire_face.dtype)
    aire.index_add_(0, etiquettes, aire_face)
    total = float(aire.sum())
    corps = int(aire.argmax())
    if total <= 0 or float(aire[corps]) <= 0:
        return v, f

    petit = aire < ratio_aire * float(aire[corps])
    petit[corps] = False
    if not bool(petit.any()):
        return v, f

    # Décollé : la MÉDIANE des distances de ses sommets à la coque principale.
    # La médiane et non le minimum, parce qu'un éclat peut effleurer la coque
    # par un coin tout en flottant par ailleurs.
    etendue = float((v.max(dim=0).values - v.min(dim=0).values).max())
    seuil = distance_mini * etendue
    decolle = torch.zeros_like(petit)
    faces_corps = f[etiquettes == corps]
    if faces_corps.shape[0] == 0:
        return v, f
    try:
        bvh = cumesh.cuBVH(v.contiguous(), faces_corps.contiguous().int())
        for morceau in torch.nonzero(petit, as_tuple=True)[0].tolist():
            pts = v[f[etiquettes == morceau].long().reshape(-1)]
            if pts.shape[0] == 0:
                continue
            if pts.shape[0] > 512:
                pris = torch.randperm(pts.shape[0], device=pts.device)[:512]
                pts = pts[pris]
            d = bvh.unsigned_distance(pts.contiguous())
            if isinstance(d, (tuple, list)):
                d = d[0]
            if float(d.median()) > seuil:
                decolle[morceau] = True
        del bvh
    except Exception as exc:                                  # noqa: BLE001
        log.warning("distance au corps non mesurable (%s) : aucun débris "
                    "retiré, l'objet passe entier", exc)
        return v, f

    condamnes = petit & decolle
    if not bool(condamnes.any()):
        log.info("débris : %d petits morceaux, tous posés sur la coque, "
                 "tous gardés", int(petit.sum()))
        return v, f

    part = float(aire[condamnes].sum()) / total
    if part > part_maxi:
        log.info("débris : le filtre voulait retirer %.1f %% de l'aire "
                 "(%d morceaux) — au-dessus du plafond, rien n'est retiré",
                 100 * part, int(condamnes.sum()))
        if notes is not None:
            notes.append("Objet fait de beaucoup de petites pièces : elles "
                         "ont toutes été gardées.")
        return v, f

    garde = ~condamnes[etiquettes]
    avant = int(f.shape[0])
    f2 = f[garde].contiguous()
    if f2.shape[0] == 0:
        return v, f
    cm = cumesh.CuMesh()
    cm.init(v.contiguous().float(), f2.contiguous().int())
    cm.remove_unreferenced_vertices()
    v2, f2 = cm.read()
    del cm
    log.info("débris : %d morceaux détachés retirés (%d faces, %.2f %% de "
             "l'aire) ; %d petits morceaux posés sur la coque gardés",
             int(condamnes.sum()), avant - int(f2.shape[0]), 100 * part,
             int((petit & ~decolle).sum()))
    return v2, f2.int()


def nettoyer(mesh, *, dc_resolution: int, cible_triangles: int,
             perimetre_bouchage: float = 1.0,
             perimetre_fermeture: float = 4.0,
             ratio_aire: float = 0.001,
             distance_mini: float = 0.004,
             part_maxi: float = 0.05,
             jalon=None, notes=None):
    """Refait la topologie du maillage et le ramène au budget du palier."""
    import cumesh

    # Les tenseurs sortent de `run()`, qui est @torch.inference_mode() : ce
    # sont des « inference tensors ». Les faire voyager vers cumesh hors de ce
    # mode peut lever « Inference tensors cannot be used ... ». On reste donc
    # dans le même mode, ce qui ne coûte rien puisque rien ici n'a de gradient.
    with torch.inference_mode():
        # --- 1. BOUCHAGE LARGE ------------------------------------------
        # Sort SANS RIEN FAIRE si le maillage n'a aucun bord (`num_boundaries`
        # ou `num_boundary_loops` à zéro). `decode_latent` a déjà passé un
        # `fill_holes()` à 3e-2 ; celui-ci est cent fois plus permissif, parce
        # que le contourage dual qui suit a besoin d'un volume fermé pour
        # décider ce qui est dedans et ce qui est dehors.
        if jalon:
            jalon("filling holes", 0.47)
        mesh.fill_holes(max_hole_perimeter=perimetre_bouchage)

        # --- 2. REMAILLAGE DUAL-CONTOURING ------------------------------
        # La SEULE étape qui refait la topologie. `resolution` est le 3e
        # argument POSITIONNEL de la fonction ; `band` garde son défaut de 1,
        # comme au banc — le widget `remesh_band` de son nœud est mort, il
        # n'est jamais transmis.
        if jalon:
            jalon("remeshing", 0.50)

        # LE GARDE-FOU. Voir `_budget_cellules` en tete de module : au-dela
        # du budget de la carte, la grille demandee ne tient pas, et l'etape
        # devient ininterrompable pendant des heures.
        entree = int(mesh.faces.shape[0])
        # Les CELLULES decident, sur toutes les cartes ; le budget suit la
        # memoire de la carte (voir `_budget_cellules`). `_cellules_estimees`
        # compte a la resolution demandee ; on le ramene a 1024 pour parler
        # la langue de la table.
        cellules_1024 = int(_cellules_estimees(mesh.vertices, 1024))
        log.info("remaillage : %d faces en entree, ~%d cellules a 1024, "
                 "grille demandee %d, budget %d",
                 entree, cellules_1024, dc_resolution, _budget_cellules())
        grille = _grille_tenable(cellules_1024, dc_resolution)
        if grille < dc_resolution:
            log.info("contourage ramene a %d : ~%d cellules a 1024, la grille "
                     "%d ne tient pas sur cette carte", grille, cellules_1024,
                     dc_resolution)
            dc_resolution = grille
            if notes is not None:
                notes.append(
                    "Objet très ouvragé : la grille de remaillage a "
                    f"été ramenée à {grille} pour tenir dans la "
                    "mémoire de cette carte. Les creux se sont un peu "
                    "arrondis ; la matière n'est pas touchée.")

        v = mesh.vertices.detach().contiguous().cuda()
        f = mesh.faces.detach().int().contiguous().cuda()
        v, f = cumesh.remeshing.reconstruct_mesh_dc_quad(
            v, f, dc_resolution,
            verbose=False,
            # NON OPTIONNEL, malgré le défaut False. Le contourage se fait sur
            # `UDF = eps` et non sur zéro : un solide en ressort avec DEUX
            # feuillets, un dehors et un dedans. Le filtre calcule le
            # barycentre de chaque quad et ne garde que `sdf >= -eps*0.1`.
            # Sans lui, l'objet est à double paroi et pèse le double.
            remove_inner_faces=True,
        )
        purger()
        if int(f.shape[0]) == 0:
            raise RuntimeError("le remaillage n'a produit aucune face")
        log.info("remaillage DC %d : %d sommets, %d faces",
                 dc_resolution, int(v.shape[0]), int(f.shape[0]))

        # --- 3. RETRAIT DES DÉBRIS --------------------------------------
        # Remplacement licence-propre du `remove_floater2` du banc (pymeshlab,
        # GPL3, absent de l'environnement du produit). Tout se passe sur la
        # carte, sans aller-retour CPU.
        #
        # DEUX CRITÈRES, ET C'EST LE SECOND QUI COMPTE : voir la mesure en
        # tête de `config.DEBRIS_RATIO_AIRE`.
        if jalon:
            jalon("removing debris", 0.57)
        v, f = _retirer_debris(v, f, ratio_aire=ratio_aire,
                               distance_mini=distance_mini,
                               part_maxi=part_maxi, notes=notes)
        purger()

        # `.int()` explicitement : le setter d'attribut contourne le `.int()`
        # que le constructeur de `Mesh` applique, et `CuMesh.init` refuse
        # autre chose que de l'int32 contigu sur CUDA.
        mesh.vertices = v.to(mesh.device)
        mesh.faces = f.int().to(mesh.device)

        # --- 4. DÉCIMATION AU BUDGET DU PALIER --------------------------
        # `verbose=True` est le défaut de la signature : sans ce False, une
        # barre tqdm « Simplifying » sort sur le stdout du moteur en
        # production. Sort sec si le maillage est déjà sous la cible.
        if jalon:
            jalon("decimating", 0.60)
        mesh.simplify_with_cumesh(target=cible_triangles, verbose=False)
        purger()
        log.info("après décimation : %d faces", int(mesh.faces.shape[0]))

        # --- 5. SECOND BOUCHAGE, ET C EST LUI QUI FERME LES YEUX ---------
        #
        # Le banc en a un ici, et on ne l avait pas porte : il passe par
        # meshlib, dont la licence interdit l usage commercial. On l avait
        # saute en se fiant a la specification, qui le disait « marginal ».
        #
        # Il ne l est pas. Mesure du 27 aout, arêtes de bord comptees APRES
        # soudure des sommets par position (sinon on compte les coutures UV,
        # 25 a 32 % des aretes des deux cotes) :
        #
        #   banc, poupee standard        87 trous   0,04 %
        #   nous, sans ce bouchage    9 293 trous   2,17 %
        #
        # Cent fois plus. Et ca se voit : la poupee sortait avec les yeux et
        # les oreilles ouverts sur le noir, signale a l oeil par Quentin.
        #
        # POURQUOI LES YEUX. Le contourage dual retire les faces interieures
        # — obligatoire, sinon on obtient une double paroi — et dans une
        # cavite il coupe la paroi du fond. Le banc rebouche derriere.
        #
        # `fill_holes` de cumesh fait le meme travail et cumesh est en MIT
        # (Copyright (c) 2025 Jianfeng XIANG). Il ne referme QUE les bords
        # ouverts, il ne touche pas a une surface deja fermee — le repasser
        # apres la decimation ne coute donc rien sur un maillage sain.
        #
        # ORDRE : apres la decimation, pas avant. La decimation reconstruit
        # la table des faces et peut rouvrir ce qui venait d etre ferme.
        # C est exactement l ordre du banc : boucher, remailler, simplifier,
        # RE-boucher.
        # --- FERMER LES CAVITES. REPARER D ABORD, BOUCHER ENSUITE. -------
        #
        # Le contourage dual retire les faces interieures — obligatoire,
        # sinon on obtient une double paroi — et dans une cavite il coupe la
        # paroi du fond. La poupee sortait avec les yeux et les oreilles
        # ouverts sur le noir, vus a l oeil par Quentin avant que la mesure
        # ne les trouve.
        #
        # DEUX CHOSES ETAIENT FAUSSES DANS LA PREMIERE TENTATIVE.
        #
        # 1. `fill_holes` ne sait fermer qu une VRAIE boucle. Un bord qui se
        #    croise lui-meme — une arete partagee par trois faces — n en est
        #    pas une, et il le refuse en silence. Il faut donc reparer les
        #    aretes non-manifold AVANT.
        #
        # 2. Le defaut de `max_hole_perimeter` est 0,03, ce qui ne ferme
        #    quasiment rien sur un objet inscrit dans une boite unite. On
        #    avait essaye 1,0 puis un million ; les deux donnaient la meme
        #    chose, ce qui semblait dire que le perimetre n y etait pour
        #    rien. Il n y etait pour rien TANT QUE le non-manifold n etait
        #    pas repare.
        #
        # Mesure du 27 aout, sur le maillage reel avant texturation, en
        # chainant les aretes de bord en boucles :
        #
        #   depart                          3 024 aretes | 36 grandes boucles
        #   bouchage seul                   2 203 aretes | 26
        #   reparation + bouchage a 1,0       303 aretes |  2
        #   reparation + bouchage a 4,0           0 aretes |  0
        #
        # Zero. Le banc, lui, en laisse 87 — il fait mieux que sa reference,
        # et sans meshlib, dont la licence interdit l usage commercial.
        #
        # DEUX PASSES : la premiere fermeture peut recreer du non-manifold
        # la ou deux bouchons se rejoignent.
        if jalon:
            jalon("closing cavities", 0.63)
        v = mesh.vertices.detach().contiguous().cuda()
        fc = mesh.faces.detach().int().contiguous().cuda()
        for _ in range(2):
            cm = cumesh.CuMesh()
            cm.init(v.contiguous(), fc.contiguous())
            cm.repair_non_manifold_edges()
            cm.fill_holes(max_hole_perimeter=perimetre_fermeture)
            v, fc = cm.read()
            del cm
        purger()
        mesh.vertices = v.to(mesh.device)
        mesh.faces = fc.int().to(mesh.device)

        # Et une derniere decimation : boucher AJOUTE des faces — 148 567 a
        # 152 788 sur la mesure ci-dessus — et le budget du palier doit
        # rester tenu. Sort sec si le maillage est deja sous la cible.
        mesh.simplify_with_cumesh(target=cible_triangles, verbose=False)
        purger()
        log.info("après second bouchage : %d faces", int(mesh.faces.shape[0]))

    return mesh


# --------------------------------------------------------------------------
# 5 : conversion + réorientation Z-up -> Y-up
# --------------------------------------------------------------------------
def vers_trimesh(mesh):
    """MeshWithVoxel -> trimesh.Trimesh Y-up, SANS UV, SANS visual.

    LA ROTATION N'EST PAS DÉCORATIVE. `preprocess_mesh`, première chose que
    fait `texture_mesh`, applique (x,y,z) -> (x,-z,y) — l'inverse exact de
    celle-ci — et `postprocess_mesh` réapplique la rotation de sortie à la
    fin. Sauter celle-ci fait texturer dans le mauvais repère et sortir
    l'objet couché.

    SANS UV : c'est ce qui fait que `postprocess_mesh` déplie lui-même. Un
    maillage qui porte déjà `visual.uv` lui fait SAUTER le dépliage et recuire
    dans l'atlas existant, en silence — d'où l'interdiction de lui donner la
    sortie de `to_glb`.
    """
    import trimesh

    v = mesh.vertices.detach().cpu().numpy().astype(np.float64, copy=True)
    f = mesh.faces.detach().cpu().numpy()
    # GARDER L'IDIOME TUPLE : le membre droit est évalué en entier avant la
    # première affectation. Réécrire ces deux colonnes en deux lignes donne
    # les deux égales à ±ancien_z.
    v[:, 1], v[:, 2] = v[:, 2], -v[:, 1]
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


# --------------------------------------------------------------------------
# 6 : texturation générée POUR ce maillage
# --------------------------------------------------------------------------
#: Communs aux quatre paliers, donc ici et pas dans la table des paliers. Les
#: trois derniers sont ceux livrés avec les poids (pipeline.lumengen.json,
#: bloc `tex_slat_sampler`) ; le banc ne change que `guidance_rescale`, de 0,0
#: à 0,05. L'intervalle [0.6, 0.9] n'est donc pas une invention du banc.
_RESCALE = 0.05
_INTERVALLE = [0.6, 0.9]
_RESCALE_T = 3.0


def texturer(pipe, tm, image, *, seed: int, resolution: int,
             texture_size: int, steps: int, guidance: float, jalon=None):
    """Rend (trimesh texturé, baseColor PIL RGBA, metallicRoughness PIL RGB)."""
    originaux = _brancher_jalons(pipe, jalon)
    try:
        return pipe.texture_mesh(
            mesh=tm,
            image=image,                     # la PIL déjà détourée et prémultipliée
            seed=seed,
            tex_slat_sampler_params={
                "steps": steps,
                "guidance_strength": guidance,
                "guidance_rescale": _RESCALE,
                "guidance_interval": _INTERVALLE,
                "rescale_t": _RESCALE_T,
            },
            # 512 -> le DiT de texture 512 ; TOUTE autre valeur -> le 1024.
            # Indépendant du `pipeline_type` de la forme : c'est ce que
            # `run()` seul interdisait, et ce que le banc faisait.
            resolution=resolution,
            texture_size=texture_size,       # côté de l'atlas
            texture_alpha_mode="OPAQUE",
            # ACCEPTÉ ET IGNORÉ : `postprocess_mesh` code `doubleSided=True`
            # en dur dans son PBRMaterial. C'est `materials.apply_alpha_mode`
            # qui remet la simple face, après coup — et il devient donc
            # indispensable sur ce chemin.
            double_side_material=False,
            # INERTE sur une photo unique : `get_cond` ne s'en sert que pour
            # trancher une liste de vues. L'exposer comme un réglage de
            # qualité serait mentir.
            max_views=4,
            bake_on_vertices=False,
            # FALSE, ET LA MESURE A TRANCHÉ CONTRE LA LECTURE.
            #
            # La spécification recommandait True, en déduisant que trimesh
            # n'écrit l'accesseur glTF NORMAL que si « vertex_normals » est
            # déjà dans son cache. La déduction était juste sur le principe
            # et fausse sur le remède : à True, le fichier sort bien avec un
            # accesseur NORMAL, mais il contient des vecteurs qui ne
            # décrivent pas cette surface.
            #
            # Mesuré sur la première génération de la chaîne portée, sur les
            # normales lues dans le binaire du glTF, pas recalculées :
            #
            #   norme moyenne des normales stockées   0,643   (au lieu de 1)
            #   écart médian avec les vraies normales  90,0°
            #   et aucune rotation ne les rattrape — ce ne sont pas des
            #   normales tournées, ce sont les mauvaises données.
            #
            # Pour comparaison, sur les sorties du banc : 10,4° d'écart
            # médian entre normales voisines en Léger, 8,7° en Standard.
            # Chez nous, 90° : la surface paraissait taillée au couteau.
            #
            # Le banc passe False depuis ses 41 essais. Les normales lisses
            # sont alors calculées par trimesh à partir de la géométrie, et
            # c'est le `_ = g.vertex_normals` de `pipeline.py`, juste avant
            # l'export, qui les met dans le cache et les fait écrire.
            use_custom_normals=False,
            mesh_cluster_threshold_cone_half_angle_rad=60.0,
            sampler="euler",
            # NON VALIDÉ EN AMONT : toute chaîne autre que 'telea' bascule sur
            # Navier-Stokes en silence. Le banc a mesuré l'écart, il est dans
            # le bruit. Ne pas exposer ce réglage.
            inpainting="telea",
            verbose=False,
            dino_lock=0.0,
            dino_substeps=4,
            dino_foundation_cap=1.0,
        )
    finally:
        _debrancher(pipe, originaux)
        # BOGUE AMONT, contourné ici. Quand `resolution != 512`, la branche
        # de `texture_mesh` décharge `shape_slat_flow_model_1024` — le modèle
        # de FORME — au lieu du modèle de TEXTURE. Les 1232 Mo du DiT texture
        # survivraient donc d'un travail à l'autre.
        if getattr(pipe, "low_vram", False):
            if resolution != 512:
                pipe.unload_tex_slat_flow_model_1024()
            else:
                pipe.unload_tex_slat_flow_model_512()
            pipe.unload_shape_slat_encoder()
        purger()


def texturer_multivue(pipe, tm, face, vues, *, seed: int, resolution: int,
                      texture_size: int, steps: int, guidance: float,
                      jalon=None):
    """Comme `texturer`, mais chaque côté observé guide ses texels.

    `vues` : {"droite"|"gauche"|"dos": PIL}, déjà détourées par `matting`
    comme la face. `texture_mesh_multiview` pondère chaque texel par
    l'angle entre sa normale et chaque caméra (`blend_temperature`) : un
    dos observé n'est plus inventé par l'inpainting.

    TOUT CE QUI EST COMMENTÉ DANS `texturer` VAUT ICI À L'IDENTIQUE :
    OPAQUE, simple face, euler, telea, normales recalculées (False), et
    les paramètres d'échantillonnage mesurés du banc. Une divergence entre
    les deux fonctions serait un bogue, pas un réglage.

    CONVENTION D'ANGLES : droite->right, gauche->left, dos->back — la même
    table que la branche forme de `pipeline.py`, PAS UNE DEUXIÈME. Si la
    mesure montre les flancs échangés, c'est là-bas que ça s'inverse et
    ici on ne touche à rien : cette fonction reçoit des mots, pas des
    degrés.

    NON MESURÉ ENCORE : écrit le 27 août pendant que la carte était prise.
    À la première génération d'essai, vérifier les flancs ET le pic de
    mémoire (quatre conditionnements DINOv3 au lieu d'un).
    """
    originaux = _brancher_jalons(pipe, jalon)
    try:
        return pipe.texture_mesh_multiview(
            mesh=tm,
            front=face,
            right=vues.get("droite"),
            left=vues.get("gauche"),
            back=vues.get("dos"),
            seed=seed,
            tex_slat_sampler_params={
                "steps": steps,
                "guidance_strength": guidance,
                "guidance_rescale": _RESCALE,
                "guidance_interval": _INTERVALLE,
                "rescale_t": _RESCALE_T,
            },
            resolution=resolution,
            texture_size=texture_size,
            texture_alpha_mode="OPAQUE",
            double_side_material=False,
            bake_on_vertices=False,
            use_custom_normals=False,
            mesh_cluster_threshold_cone_half_angle_rad=60.0,
            sampler="euler",
            inpainting="telea",
            verbose=False,
            dino_lock=0.0,
            dino_substeps=4,
            dino_foundation_cap=1.0,
        )
    finally:
        _debrancher(pipe, originaux)
        # LE MEME BOGUE AMONT que `texture_mesh`, verifie dans le source :
        # la branche `resolution != 512` de `texture_mesh_multiview`
        # decharge elle aussi `shape_slat_flow_model_1024` au lieu du DiT
        # de texture.
        if getattr(pipe, "low_vram", False):
            if resolution != 512:
                pipe.unload_tex_slat_flow_model_1024()
            else:
                pipe.unload_tex_slat_flow_model_512()
            pipe.unload_shape_slat_encoder()
        purger()


def _brancher_jalons(pipe, jalon):
    """Trois jalons pendant `texture_mesh`, sans recopier `texture_mesh`.

    Même procédé que `pipeline._stage_previews` : on emballe des méthodes le
    temps de l'appel. La texturation dure la moitié du travail et n'a aucun
    point de progression à elle ; sans ça la barre reste figée trois minutes.

    DEUX CORRECTIONS, ET LA SECONDE RENDAIT « ANNULER » INERTE.

    Le jalon était posé APRÈS l'appel. Pendant l'échantillonnage de la
    texture — le poste le plus long, jusqu'à quatre-vingt-dix secondes — il
    ne se passait donc rien : ni progression, ni point de contrôle. Il est
    posé AVANT, et l'étape annoncée est celle qui commence, pas celle qui
    vient de finir.

    Et le `except Exception: pass` avalait le `JobCancelled` que `step` lève
    quand l'utilisateur annule. L'annulation était levée, puis jetée. Sur
    une texturation de deux minutes, le bouton ne faisait rien du tout —
    et une génération dure assez longtemps pour que l'annulation soit
    l'action la plus probable.

    Un jalon ne doit toujours pas casser un travail : une barre de
    progression qui rate un tic n'est pas une raison d'échouer. Mais une
    demande d'annulation n'est pas un incident de barre, c'est un ordre.
    """
    if jalon is None:
        return {}
    plan = {"encode_shape_slat": ("encodage de la forme", 0.70),
            "sample_tex_slat":   ("échantillonnage de la texture", 0.78),
            # Le chemin multi-vue passe par SON echantillonneur ; sans cette
            # entree, la barre restait muette pendant le poste le plus long
            # et « Annuler » perdait son point de controle.
            "sample_tex_slat_multiview":
                                 ("échantillonnage de la texture", 0.78),
            "decode_tex_slat":   ("cuisson de l'atlas", 0.86)}
    originaux = {}
    for nom, (libelle, frac) in plan.items():
        fn = getattr(pipe, nom, None)
        if fn is None:
            continue
        originaux[nom] = fn

        def emballe(_fn=fn, _l=libelle, _f=frac):
            def wrapped(*a, **k):
                _tic(jalon, _l, _f)
                return _fn(*a, **k)
            return wrapped
        setattr(pipe, nom, emballe())
    return originaux



def _tic(jalon, libelle, frac) -> None:
    """Poser un jalon, en laissant passer l'annulation.

    `JobCancelled` est reconnu par son NOM et non par son type : l'importer
    depuis `pipeline` créerait un cycle, puisque c'est `pipeline` qui appelle
    ce module.
    """
    try:
        jalon(libelle, frac)
    except BaseException as exc:                          # noqa: BLE001
        if type(exc).__name__ == "JobCancelled":
            raise
        log.warning("jalon %s non pose : %s", libelle, exc)


def _debrancher(pipe, originaux):
    for nom, fn in originaux.items():
        setattr(pipe, nom, fn)
