"""The generation engine: reference photo(s) -> textured, finished 3D asset.

One process owns one GPU and one pipeline. Loading is lazy and the pipeline is
kept resident between jobs - reloading 15 GB of checkpoints per job would dwarf
the generation itself.
"""
from __future__ import annotations

import gc
import os
import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Callable, Literal, Optional

from . import config

config.bootstrap()

import torch  # noqa: E402  (must follow bootstrap)
from PIL import Image  # noqa: E402

from .meshops import FinishSettings, finish  # noqa: E402

log = logging.getLogger("localmesh_engine.pipeline")

#: LE PLAFOND DE FACES QUE LE REMAILLAGE ACCEPTE, PAR GIGAOCTET de memoire
#: video : au-dela, la carte decroche et le pilote peut emporter la machine
#: (voir le point de controle). Mesure sur la 4060 de 8 Go : 8,88 M de
#: faces ont fini, 12,3 M ont mis la machine en ecran bleu ; 1 M par Go
#: donne 8 M sur 8 Go et 24 M sur 24 Go. LUMENGEN_PLAFOND_FACES impose une
#: valeur absolue (bancs). Quand le plafond est depasse, le travail REPREND
#: UN PALIER PLUS BAS au lieu d'echouer : voir PALIER_DE_REPLI.
FACES_PAR_GO = 1_000_000
PLAFOND_FACES_MINI = 4_000_000

#: LA NETTETE DU VOTE ENTRE LES VUES. Pour chaque voxel, les quatre vues
#: se partagent la prediction selon la position ; ce nombre dit a quel
#: point le partage est tranche : a 2 (valeur heritee du noeud ComfyUI) la
#: vue de face ne pese que 72 % au bout du nez, a 4 elle en pese 96 %.
#: Mesure sur le samourai du banc (4 septembre 2026, Standard, 4060) : le
#: 4 rend casque, motifs de la cape et socle plus nets, pour 5 % de temps
#: en plus ; choix de Quentin. LUMENGEN_BANC_MELANGE l'impose pour un banc.
MELANGE_MULTIVUE = 4.0


def _melange_multivue() -> float:
    regle = os.environ.get("LUMENGEN_BANC_MELANGE")
    if regle:
        try:
            return float(regle)
        except ValueError:
            log.warning("LUMENGEN_BANC_MELANGE illisible (%r), mélange %s", regle, MELANGE_MULTIVUE)
    return MELANGE_MULTIVUE
PALIER_DE_REPLI = {"standard": "draft", "high": "standard", "max": "high"}


def _plafond_de_faces() -> int:
    regle = os.environ.get("LUMENGEN_PLAFOND_FACES")
    if regle:
        try:
            return int(regle)
        except ValueError:
            log.warning("LUMENGEN_PLAFOND_FACES illisible (%r), plafond par carte", regle)
    try:
        go = float(config.gpu_report().total_vram_gb)
    except Exception:                                        # noqa: BLE001
        go = 8.0
    return max(PLAFOND_FACES_MINI, int(FACES_PAR_GO * go))

Detail = Literal["draft", "standard", "high", "max"]
ProgressFn = Callable[[str, float], None]


#: glTF alpha modes, in the words an artist uses.
_MODE_LABELS = {"OPAQUE": "opaque", "MASK": "découpe", "BLEND": "transparent"}

#: How many times run() ticks its progress bar. Counted from the source rather
#: than guessed; only used to turn ticks into a fraction, so being off by one
#: costs nothing but a slightly uneven bar.
_RUN_PROGRESS_TICKS = 12


@dataclass
class GenerateSettings:
    """Everything the artist panel can set, in artist terms."""

    # --- reference ---------------------------------------------------------
    #: First image drives geometry. The rest are used by the texture pass.
    images: list[Path] = field(default_factory=list)
    #: Les autres côtés du sujet, par nom d'angle : {"dos": Path(...)}.
    #: Clés admises : "droite", "gauche", "dos" — la face est `images[0]`,
    #: le même vocabulaire que `multivue.AZIMUTS` et que la route splat.
    #: Vide = une seule photo, le chemin d'hier, inchangé.
    vues: dict[str, Path] = field(default_factory=dict)
    #: DE QUEL CÔTÉ LES DEUX PROFILS ONT ÉTÉ PRIS.
    #:
    #: Ce n'est pas une préférence, c'est un fait sur les photos, et
    #: l'emplacement où l'utilisateur a rangé l'image ne le dit pas : une
    #: étiquette dit où la photo a été classée, pas d'où elle a été prise.
    #: "auto" prend la convention et le résultat porte la mention ;
    #: "inversee" échange les deux profils.
    lateralite: str = "auto"

    # --- quality -----------------------------------------------------------
    detail: Detail = "standard"
    #: LES CHAMPS A None SONT DECIDES PAR LE PALIER (config.PALIERS).
    #:
    #: Ils valaient des nombres fixes envoyes par l'interface, identiques
    #: aux quatre paliers : choisir "Apercu" ou "Detaille" ne changeait donc
    #: ni les pas, ni l'echantillonneur, ni l'atlas. La valeur explicite
    #: reste possible — c'est ce dont le banc a besoin — mais elle n'est
    #: plus le chemin normal.
    #: Force du guidage de FORME. None laisse celle des poids (7,5).
    fidelity: Optional[float] = None
    steps_structure: Optional[int] = None
    steps_shape: Optional[int] = None
    steps_texture: Optional[int] = None
    seed: int = -1  # -1 -> random
    variations: int = 1

    # --- quality knobs ------------------------------------------------------
    # Not exposed in the interface. They exist so the benchmark can measure
    # them; whichever wins becomes the default here, and the artist never has
    # to learn what a Runge-Kutta sampler is.
    #: QUEL CHEMIN MULTIVUE : "auto" (la regle du palier), "porte" (le chemin
    #: corrige, de force) ou "ancien" (l'ancien melangeur, de force).
    #:
    #: SUR LA DEMANDE ET NON DANS L'ENVIRONNEMENT, et c'est ce qui compte : une
    #: variable d'environnement demande un redemarrage du moteur, donc la
    #: fermeture de la fenetre de l'artiste, pour chaque essai. Le 9 septembre
    #: 2026 il a fallu trois generations pour fermer deux pistes sur un buste
    #: deforme ; la quatrieme aurait coute une fenetre. Comme `steps_*`, ce
    #: champ existe pour que le banc mesure sans rien casser.
    chemin_mv: str = "auto"
    #: LES ANGLES DE PRISE DE VUE : "auto" (mesures quand ils sont surs) ou
    #: "nominaux" (le tour parfait 0/90/180/270, sans mesure).
    #:
    #: A mesurer, pas a supposer : sur le buste fissure du 9 septembre le
    #: moteur mesure droite 289 degres et gauche 70 la ou les photos font un
    #: tour regulier -- mais le samourai est tout aussi irregulier et son
    #: resultat est excellent. Tant que l'essai n'a pas eu lieu, le defaut
    #: reste "auto".
    angles_mv: str = "auto"
    #: LE DETOURAGE DES VUES : "auto" (un cadre commun aux quatre, cle sur la
    #: hauteur du sujet) ou "separe" (chaque vue sur sa propre boite, l'ancien
    #: comportement). Le lot ne change rien sur un objet plus haut que large ;
    #: il ne bouge que les objets allonges.
    cadre_mv: str = "auto"
    #: LA FORCE DE PONDERATION DES VUES. None = la regle mesuree (elle suit le
    #: desaccord des photos) ; un nombre l'impose. 0 melange les quatre vues a
    #: plat, 3 laisse chaque vue peser la ou elle voit.
    force_mv: Optional[float] = None
    #: "euler" | "heun" | "rk4" | "rk5" - integration accuracy per step.
    #:
    #: heun is the default on measurement, not preference. Benchmarked against
    #: euler on two subjects at a fixed seed: on a generated eye, euler tore
    #: the iris into fragments and heun produced it intact; on an ornate crown
    #: the two were indistinguishable. Same VRAM either way.
    #:
    #: The time comparison from that benchmark is NOT trustworthy - the first
    #: variant of each run pays the model load - so no speed claim is made.
    #: MESURE LE 26 AOUT 2026, sur la meme photo a graine fixe : heun 25 pas
    #: contre euler 12 pas, 236-294 s contre 114-144 s. heun fait DEUX
    #: evaluations du modele par pas ; a qualite egale sur les sujets du banc,
    #: c'est le temps double pour rien. Les quatre paliers du banc tournent
    #: tous en euler. None laisse le palier decider.
    sampler: Optional[str] = None
    #: COMMENT LES TROUS DE LA GRILLE SE BOUCHENT.
    #:
    #: Le defaut vendorise, "remove_small_holes", est accompagne d'un
    #: `hole_structure` de 1 : le seuil d'aire vaut 1x1, donc il ne bouche
    #: RIEN. Le banc passe "flood_fill" — fermeture morphologique, puis
    #: remplissage, puis on ne garde que la plus grande composante connexe.
    #: C'est le remede en amont aux debris flottants, pour un cout CPU
    #: negligeable sur une grille 32 cube.
    #:
    #: RESERVE : "ne garder que la plus grande composante" peut manger une
    #: piece legitimement detachee (une anse, un crochet). Mesure avant
    #: d'etre mis par defaut — voir ESSAI-TRELLIS.md.
    hole_fill: str = "flood_fill"
    #: 0 disables. Higher keeps the generation anchored to the reference's
    #: DINOv3 features while it denoises.
    dino_lock: float = 0.0
    dino_substeps: int = 4
    dino_foundation_cap: float = 0.92
    #: Resolution of the first, cheapest stage. 32 is the shipped default;
    #: 64 and above are flagged experimental upstream and only apply to the
    #: cascade presets.
    structure_resolution: int = 32

    # --- texture -----------------------------------------------------------
    #: Force du guidage de la TEXTURE. None laisse le palier decider.
    #:
    #: Separe de `fidelity`, qui ne touche que la forme : les poids livres
    #: disent 7,5 pour la forme et 1,0 pour la texture, et le meme nombre
    #: pousse dans les deux ne veut rien dire. A 1,0 l'echantillonneur ne
    #: fait qu'UNE evaluation par pas ; au-dessus, deux.
    tex_guidance: Optional[float] = None
    #: None laisse le palier decider. 1024 | 2048 | 4096
    texture_size: Optional[int] = None
    #: Bake the relief lost to the poly budget into a normal map. Pointless
    #: without a budget (nothing was removed) and pointless for 3D printing
    #: (a printer prints geometry, not a lighting trick).
    bake_normal_map: bool = True

    # --- finishing ---------------------------------------------------------
    finish: FinishSettings = field(default_factory=FinishSettings)

    #: "auto" reads the mode off the alpha the model produced; the explicit
    #: values let the artist overrule a wrong reading.
    transparency: Literal["auto", "opaque", "mask", "blend"] = "auto"

    #: Millimetres for the largest dimension. None keeps TRELLIS' unit cube.
    real_size_mm: Optional[float] = None
    up_axis: Literal["y", "z"] = "y"
    origin_at_base: bool = True

    def resolved_seed(self) -> int:
        return random.randint(0, 2 ** 31 - 1) if self.seed < 0 else int(self.seed)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["images"] = [str(p) for p in self.images]
        # Des Path ici tuaient le flux SSE de la session (TypeError dans
        # json.dumps) et le manifeste du maillage multi-vue avec lui.
        d["vues"] = {k: str(v) for k, v in (self.vues or {}).items()}
        return d


def _plafond_de_tokens() -> int:
    """Combien de tokens la carte peut porter.

    Le defaut vendorise est 49 152, pose pour une carte de datacenter. Le
    banc a etabli la regle : 2 600 tokens par gigaoctet de memoire video,
    et c'est CE plafond — pas la resolution — qui gouverne la memoire. Une
    carte de 8 Go recoit donc 20 800, une de 24 Go en recoit 62 400.

    PORTEE REELLE : la boucle de troncature ne descend jamais sous 1024
    quand la resolution vaut 1024, donc en cascade 1024 ce plafond est sans
    effet — le cout de cette cascade est intrinseque. Il mord sur la
    cascade 1536, ou il est le seul garde-fou.
    """
    try:
        go = config.gpu_report().total_vram_gb
    except Exception:                                        # noqa: BLE001
        return 49_152
    return max(8_192, int(2_600 * go))


class StageTimer:
    """Wall-clock per stage, so the cost model is fitted on real measurements
    rather than a guess about which part is slow."""

    def __init__(self):
        self._marks: list[tuple[str, float]] = []
        self.start = time.time()

    def mark(self, stage: str):
        self._marks.append((stage, time.time()))

    def durations(self) -> dict[str, float]:
        out: dict[str, float] = {}
        prev = self.start
        for name, at in self._marks:
            out[name] = round(at - prev, 2)
            prev = at
        return out


@dataclass
class GenerateResult:
    glb_path: Path
    seed: int
    detail: Detail
    faces: int
    vertices: int
    duration_s: float
    peak_vram_gb: float
    finish_report: dict
    #: Small render of the finished asset. Comparing several seeds by clicking
    #: each card in turn is unusable; a thumbnail makes the choice immediate.
    #: Declared here, after the required fields, because a defaulted field
    #: cannot precede one without a default.
    thumbnail: Optional[Path] = None
    stages: dict = field(default_factory=dict)
    file_size_mb: float = 0.0
    notes: list[str] = field(default_factory=list)
    #: "delta_carte" | "pytorch" | "indisponible" — d'ou vient `peak_vram_gb`.
    vram_portee: str = "pytorch"
    #: Part de la carte occupee au pic, en pourcent. 0 si non mesurable.
    vram_occupation_pct: float = 0.0

    def as_dict(self) -> dict:
        return {
            "glb": str(self.glb_path),
            "thumbnail": str(self.thumbnail) if self.thumbnail else None,
            "seed": self.seed,
            "detail": self.detail,
            "faces": self.faces,
            "vertices": self.vertices,
            "duration_s": round(self.duration_s, 1),
            "peak_vram_gb": round(self.peak_vram_gb, 2),
            "vram_portee": self.vram_portee,
            "vram_occupation_pct": self.vram_occupation_pct,
            "file_size_mb": round(self.file_size_mb, 2),
            "stages": self.stages,
            "finish": self.finish_report,
            "notes": self.notes,
        }


#: AU-DELA DE CET ECART, LES QUATRE VUES NE MONTRENT PLUS LE MEME OBJET.
#:
#: Tourner autour d'un objet ne le fait pas grandir : sa hauteur dans l'image
#: doit rester la meme sous les quatre angles. Sa largeur change, c'est normal ;
#: sa hauteur, non. Un ecart de hauteur est donc une preuve directe que les
#: quatre images ne sont pas quatre vues d'un objet, et aucune fusion ne
#: rattrape ca.
#:
#: MESURE SUR QUATRE SUJETS (8 sept. 2026), detourage reel :
#:
#:     corps     0,20 %   chemin porte : bon
#:     tete      2,66 %   chemin porte : bon
#:     samourai  5,04 %   chemin porte : bon
#:     buste     8,68 %   chemin porte : VISAGE DEDOUBLE
#:
#: CE SEUIL NE DETOURNE PLUS RIEN, IL AVERTIT. Il a servi un temps a renvoyer
#: les sujets discordants vers l'ancien melangeur ; la ponderation par
#: visibilite a supprime le defaut qui justifiait ce detour, et le buste sort
#: correct a 8,7 % de desaccord. Il ne reste que la mention dans le resultat.
#: Reglable au banc, et SEULEMENT au banc : sans ce levier on ne peut pas
#: mesurer ce que le chemin porte donne sur un sujet que cette regle ecarte,
#: donc on ne peut pas savoir si la regle a raison. Le produit ne le pose
#: jamais.
ECART_DE_VUES_MAX = float(os.environ.get("LUMENGEN_ECART_VUES_MAX", "6.0"))


def _accord_des_vues(geo_vues: dict) -> dict:
    """Les quatre vues s'accordent-elles sur la taille de l'objet ?

    Rend `{"mesurable": bool, "ecart": float | None}`. On refuse de juger
    quand une vue est coupee par le bord de l'image : sa hauteur est alors
    tronquee, et la comparer aux autres ne veut rien dire.
    """
    if set(geo_vues) != {"front", "right", "back", "left"}:
        return {"mesurable": False, "ecart": None}
    hauteurs = [g.get("sujet_hauteur_relative") for g in geo_vues.values()]
    if any(h is None or h <= 0 for h in hauteurs):
        return {"mesurable": False, "ecart": None}
    if any(g.get("sujet_touche_le_bord") for g in geo_vues.values()):
        return {"mesurable": False, "ecart": None}
    moyenne = sum(hauteurs) / len(hauteurs)
    return {"mesurable": True,
            "ecart": (max(hauteurs) - min(hauteurs)) / moyenne * 100.0}


def _multivue_porte(nb_vues: int, settings, geo_vues: dict | None = None) -> bool:
    """Le chemin multivue porte peut-il prendre ce travail ?

    Quatre conditions. Les trois premieres sont techniques : les trois cotes
    sont la, le palier est celui sur lequel la recette a ete mesuree, les poids
    sont poses.

    LA QUATRIEME EST UN CHOIX DE QUALITE, ET C'EST LA PLUS INTERESSANTE. Le
    chemin porte fond les quatre vues cellule par cellule ; quand elles se
    contredisent, il pose le meme oeil deux fois. L'ancien melangeur mele des
    predictions entieres et laisse la vue de face dominer l'avant, donc il
    encaisse mieux le desaccord. Aucun des deux n'est meilleur partout : on
    mesure, et on prend celui qui convient.

    Tout ce qui ne passe pas repart sur l'ancien melangeur, sans bruit et sans
    erreur.
    """
    # UN INTERRUPTEUR DE BANC, ET RIEN D'AUTRE. Comparer les deux melangeurs
    # sur un meme sujet demande de pouvoir eteindre le premier sans deplacer
    # ses poids. Il n'est lu que par le banc ; le produit ne le pose jamais.
    if os.environ.get("LUMENGEN_SANS_MULTIVUE_PORTE"):
        return False
    if nb_vues != 3:
        return False

    # L'INTERRUPTEUR DE COMPARAISON. Une seule variable, sinon la mesure ne
    # vaut rien : meme photo, meme graine, meme palier, et un seul chemin qui
    # change. Sans lui, comparer l'ancien melangeur au chemin porte obligeait
    # a changer de palier -- donc aussi le nombre de pas et la resolution de
    # forme -- et le resultat n'aurait rien prouve.
    #
    # Il sert a repondre a UNE question posee le 9 septembre 2026 : le buste
    # fissure rendu bosselé l'est-il a cause du chemin porte, ou est-ce que
    # l'ancien melangeur faisait pareil ? Tant que la reponse n'est pas
    # mesuree, on ne touche a rien.
    if (os.environ.get("LUMENGEN_MV_ANCIEN") == "1"
            or getattr(settings, "chemin_mv", "auto") == "ancien"):
        return False
    if getattr(settings, "chemin_mv", "auto") == "porte":
        return True

    # QUELS PALIERS CE CHEMIN PEUT SERVIR, ET POURQUOI PAS LES DEUX DERNIERS.
    #
    # Il a longtemps ete reserve au SEUL Standard. Le cout etait invisible et
    # lourd : tous les correctifs du 9 septembre 2026 -- angles mesures, force
    # adaptative, lateralite -- vivent ici, donc une generation a quatre vues
    # en Detaille n'en recevait AUCUN et repartait sur l'ancien melangeur.
    # Quentin l'a decouvert a l'ecran, en plein essai : deux sabres et une main
    # dans le dos sur un samourai en Detaille, defauts que ce chemin-ci ne fait
    # plus en Standard. Son rapport le prouvait en creux -- pas une seule note
    # de cotes.
    #
    # CE QUI DECIDE, C'EST LA RESOLUTION DE FORME QUE LE PALIER PROMET.
    #
    #   Brouillon   512    ce chemin rend 1024 : PLUS que promis, plus lent
    #   Standard   1024    exactement ce qui est promis
    #   Detaille   1024    exactement ce qui est promis
    #   Max        1536    MOINS que promis
    #   Extreme    1536    MOINS que promis
    #
    # Les poids multivue n'existent qu'en 512 et 1024 (`Poids.forme_512`,
    # `forme_1024`) : au-dela de 1024, ce chemin NE PEUT PAS tenir la promesse
    # du palier, et personne ne la tiendra a sa place. Max et Extreme restent
    # donc sur l'ancien melangeur, qui suit le palier. Ce n'est pas un renoncement
    # cache : c'est la seule facon de ne pas vendre du Standard sous une autre
    # etiquette.
    #
    # LUMENGEN_MV_TOUS_PALIERS=1 ouvre quand meme les deux derniers, pour
    # mesurer si un 1024 avec les correctifs bat un 1536 sans. Tant que cette
    # mesure n'existe pas, le defaut ne bouge pas.
    if (config.recette(settings.detail).pipeline_type.startswith("1536")
            and os.environ.get("LUMENGEN_MV_TOUS_PALIERS") != "1"):
        return False
    # LA REGLE D'ACCORD NE DETOURNE PLUS, ET C'EST UNE BONNE NOUVELLE.
    #
    # Elle renvoyait a l'ancien melangeur les sujets dont les quatre vues ne
    # s'accordent pas, parce que le chemin porte y sortait des traits en double
    # — deux nez sur le buste de pierre. Ce defaut venait de la fusion a
    # moyenne plate, pas du desaccord lui-meme : chaque vue pese maintenant ce
    # qu'elle voit, et le buste sort avec un seul nez a 8,7 % de desaccord.
    #
    # Detourner serait donc renoncer sans raison. La MESURE reste, et le
    # resultat la dit : quatre photos qui ne montrent pas le meme objet
    # restent une information utile pour celui qui les a prises.
    try:
        from .multivue import Poids
        return Poids.depuis(config.MODELS_ROOT).pretes()
    except Exception:                                         # noqa: BLE001
        return False


#: Le vocabulaire de l'application vers celui des modeles, en UN endroit.
_NOM_DU_ROLE = {"droite": "right", "gauche": "left", "dos": "back"}


#: LE TRANSPORT DE CADRE NE PART PAS, ET C'EST UNE MESURE, PAS UN DOUTE.
#:
#: L'idee etait juste : chaque photo etant recadree sur son propre sujet, un
#: meme point de l'objet ne tombe pas au meme endroit d'une vue a l'autre, et
#: revenir au cadre d'origine les remet d'accord. L'arithmetique est exacte
#: (ecart maximal 1,4e-14) et la garde est bonne.
#:
#: Elle est quand meme perdante. Samourai, quatre vues, graine 42, cotes justes
#: des deux cotes, seule difference ce transport, trois lancements chacun :
#:
#:     sans : 148 776 a 149 529 faces, 1 191 a 1 331 non-manifold, 463 a 521 s
#:     avec : 149 354 a 149 731 faces,   961 a 2 434 non-manifold, 625 a 657 s
#:
#: ET UN AVERTISSEMENT SUR LA PREUVE. Le compte de BORDS OUVERTS ne mesure
#: rien ici : la MEME configuration relancee donne 1 puis 472. Le remaillage
#: et la finition ne sont pas deterministes, et cette colonne saute de
#: plusieurs centaines a entree identique. Elle avait d'abord servi a fonder
#: ce verdict ; elle ne le peut pas.
#:
#: Ce qui tient : le transport coute 25 a 40 pour cent de temps en plus, il
#: pousse la forme brute au-dessus du plafond de la carte donc declenche une
#: reduction que l'autre voie evite, et a l'oeil, en argile, la cape se lisse,
#: l'arbre s'epaissit, la base se simplifie. Il perd du detail sans rien
#: rendre de visible.
#:
#: POURQUOI. Les modeles ont ete entraines sur des sujets qui REMPLISSENT le
#: cadre — c'est exactement ce que fait notre detourage. Transporter vers le
#: cadre d'origine rend les quatre vues coherentes ENTRE ELLES, mais les rend
#: toutes les quatre incoherentes avec l'entrainement : l'objet n'occupe plus
#: qu'une fraction de la carte de traits echantillonnee, donc on l'echantillonne
#: plus grossierement. Le desaccord avec l'entrainement coute plus cher que le
#: desaccord entre les vues.
#:
#: Le code reste, mesure et garde, parce qu'un vrai cadre commun — l'union des
#: quatre recadrages, ou le sujet remplit encore le cadre — reste la piste
#: propre. Ce n'est pas ce qui est ecrit ici, et on ne l'invente pas au chausse-
#: pied.
#: PILOTABLE POUR LE BANC. Astra l'a mesure gagnant sur deux sujets
#: synthetiques a cameras connues (robot, chaise : silhouettes +2 a +4
#: points sur 8 vues, essais 157-168) et n'a jamais pu l'essayer sur les
#: notres. Le banc a verite terrain du 9 septembre le peut : il faut donc
#: pouvoir l'allumer sans toucher au produit.
TRANSPORT_DE_CADRE = os.environ.get("LUMENGEN_TRANSPORT_CADRE") == "1"


def _cadre_commun(geo_vues: dict) -> dict | None:
    """Les quatre photos partagent-elles un cadre ou l'on peut les recaler ?

    LE MULTIVUE A BESOIN D'UN REPERE COMMUN. Chaque photo est recadree sur son
    propre sujet, donc un meme point de l'objet ne tombe pas au meme endroit
    d'une vue a l'autre. Revenir au cadre d'origine remet les quatre d'accord,
    MAIS SEULEMENT SI CE CADRE EST LE MEME : quatre images prises au meme
    endroit avec le meme appareil, ce qu'est un tour d'objet.

    Quand ce n'est pas le cas, on ne bricole pas : on rend `None`, la chaine
    projette dans les cadres recadres comme avant, et la note le dit. Un
    recalage sur des cadres qui n'ont rien a voir serait pire que pas de
    recalage du tout.
    """
    if not TRANSPORT_DE_CADRE:
        return None
    if set(geo_vues) != {"front", "right", "back", "left"}:
        return None
    if not all(g and g.get("valid") for g in geo_vues.values()):
        return None
    tailles = {tuple(g["source_size"]) for g in geo_vues.values()}
    return geo_vues if len(tailles) == 1 else None


def _mv_images(face, vues_pil) -> dict:
    """Les quatre vues detourees, dans le vocabulaire des modeles.

    Les pixels sont EXACTEMENT ceux de la voie normale : meme detourage, meme
    mise au carre. C'est ce qui rend les deux voies comparables a l'ecran.
    """
    import numpy as np

    sortie = {"front": np.asarray(face.convert("RGB"), dtype=np.float32) / 255.0}
    for angle, image in vues_pil.items():
        sortie[_NOM_DU_ROLE[angle]] = np.asarray(
            image.convert("RGB"), dtype=np.float32) / 255.0
    return sortie


class Engine:
    """Owns the loaded TRELLIS.2 pipeline. Not thread-safe by design: one GPU,
    one job at a time. Concurrency is handled by the job queue above it."""

    def __init__(self, weights: Path | None = None, low_vram: bool | None = None):
        self.weights = Path(weights or config.TRELLIS_WEIGHTS)
        self._pipeline = None
        self._loaded_at: float | None = None
        #: Pixal3D is a separate 24 GB checkpoint set. Only one model is kept
        #: resident at a time: holding both would cost VRAM for nothing, since
        #: a job uses exactly one. La residence est tenue par `_active` et
        #: `unload()`, pas par un cache : un dictionnaire de pipelines a
        #: existe ici, jamais lu ni ecrit, et il decrivait une politique
        #: appliquee ailleurs.
        self._active: str = "trellis2"

        if low_vram is None:
            # LA DECISION SE PREND SUR CE QUI EST LIBRE, PAS SUR CE QUE LA
            # CARTE VAUT.
            #
            # Elle se prenait sur `gpu_report().total_vram_gb`, c est-a-dire la
            # capacite gravee sur la carte. Une RTX 5090 annonce 32 Go et se
            # voyait donc accorder le mode « tout chaud » — MEME QUAND DIX
            # GIGAOCTETS ETAIENT DEJA PRIS PAR AUTRE CHOSE. LocalMesh en
            # prenait dix-huit de plus, il restait trois gigaoctets a Windows,
            # le pilote se mettait a paginer vers la RAM, et c est le PC
            # ENTIER qui tombait — pas seulement LocalMesh.
            #
            # Constate chez un client le 27 aout 2026 : 10 Go occupes au
            # depart, 28,6 puis 31,3 sur 32 a la fin, machine inutilisable.
            # Le meme maillage a un pic MESURE de 5,6 Go sur une 4060 en mode
            # econome : le travail n a jamais eu besoin de dix-huit.
            #
            # `mem_get_info()` rend (libre, total) et le produit s en sert
            # deja ailleurs (telemetry.py). On decide donc sur le LIBRE, et on
            # exige large : le mode « tout chaud » ne se borne pas tout seul,
            # alors qu on connait le cout de l autre — 5,6 Go. Ce qui reste a
            # Windows n est pas du gachis, c est ce qui l empeche de paginer.
            #
            # Et en cas de doute, c est le mode ECONOME qui gagne : il coute
            # une trentaine de pour cent sur une generation repetee, la ou se
            # tromper dans l autre sens coute la machine.
            libre_go = None
            try:
                libre, _total = torch.cuda.mem_get_info()
                libre_go = libre / 1024 ** 3
            except Exception:                                 # noqa: BLE001
                pass
            if libre_go is None:
                gpu = config.gpu_report()
                libre_go = gpu.total_vram_gb
            # ECONOME PARTOUT (2 septembre 2026). Le mode « tout chaud »
            # gardait tous les sous-modeles residents des que 24 Go etaient
            # libres. Avec Extreme (grille 1536, pic estime 22 a 27 Go sur un
            # sujet ouvrage), 6,6 Go de poids residents en plus font deborder
            # une 4090 et frolent une 5090 — le scenario du client du 27 aout,
            # PC entier tombe. Et garder chaud ne fait pas gagner de temps
            # (mesure : moteur chaud 740 s, froid 474 s ; rechargement 5 a
            # 11 s). Doctrine de Quentin : memes regles sur toutes les cartes.
            low_vram = True
            log.info("mode econome sur toutes les cartes : %.1f Go libres",
                     libre_go)
        self.low_vram = low_vram

    # -- lifecycle ----------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self, progress: ProgressFn | None = None, model: str = "trellis2"):
        """Load (or switch to) the requested model, keeping only it resident."""
        if model != self._active:
            self.unload()
            self._active = model

        if self._pipeline is not None:
            return self._pipeline

        pixal = model == "pixal3d"
        self.weights = (config.PIXAL3D_WEIGHTS if pixal else config.TRELLIS_WEIGHTS)
        if _descripteur(self.weights) is None:
            raise FileNotFoundError(
                f"{'Pixal3D' if pixal else 'TRELLIS.2'} weights not found at "
                f"{self.weights}."
                + (" Le niveau « Extrême » demande un téléchargement de 24 Go."
                   if pixal else " Lancez le téléchargement initial.")
            )
        t0 = time.time()
        if progress:
            progress("loading models", 0.0)

        Trellis2ImageTo3DPipeline = config.trellis(
            "pipelines").Trellis2ImageTo3DPipeline

        from . import patches
        patches.apply()
        from . import leurre_nvdiffrast as _leurre
        _leurre.verifier()      # le journal dit quel nvdiffrast est monté

        config_name = _write_local_pipeline_config(self.weights)
        # The vendored pipeline loads each sub-model on demand and takes extra
        # switches; the standalone fork takes neither. Probe rather than assume,
        # so the engine keeps working against whichever copy is present.
        import inspect

        kwargs = {"config_file": config_name}
        accepted = inspect.signature(
            Trellis2ImageTo3DPipeline.from_pretrained).parameters
        if "keep_models_loaded" in accepted:
            # 24 GB of VRAM: keep the checkpoints resident between jobs rather
            # than paying the 80 s reload every time.
            kwargs["keep_models_loaded"] = not self.low_vram
        if "isPixal3D" in accepted:
            kwargs["isPixal3D"] = pixal
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(self.weights), **kwargs)
        pipe.low_vram = self.low_vram
        # LE MODELE QUE LE PRODUIT NE CHARGEAIT JAMAIS. La texturation passe
        # par `encode_shape_slat` -> `load_shape_slat_encoder`, qui code son
        # chemin EN DUR, hors de la config : `ckpts_fp8/...` si `use_fp8`,
        # sinon `ckpts/...fp16` — un dossier qui n'existe pas sur disque (les
        # poids livres n'ont que `ckpts_fp8/`). Sans cette ligne, le premier
        # appel part en hf_hub_download avec repo_id=<le chemin du runtime> et casse.
        #
        # APRES `from_pretrained`, JAMAIS EN ARGUMENT : passe en argument, il
        # force `config_file="pipeline_fp8.json"` et le produit perd la
        # substitution BiRefNet_HR ecrite par `_write_local_pipeline_config` —
        # donc son detourage repasserait sur un modele non commercial.
        #
        # Verifie le 27 aout 2026 sur une installation d'essai : FlexiDualGridVae-
        # Encoder, 354,4 M parametres, 338,2 Mo, 0 cle manquante, 0 cle
        # inattendue. La verification compte parce que `models.from_pretrained`
        # charge en `strict=False` : un mauvais fichier passerait en silence.
        pipe.use_fp8 = True
        if not self.low_vram:
            pipe.cuda()
        else:
            # En low_vram les poids restent sur le CPU et chaque etage est
            # monte sur le GPU juste avant de servir, avec `.to(self.device)`.
            # Or `device` se deduit du premier modele charge : tant qu'aucun
            # n'est monte, il vaut CPU, et ce `.to()` ne fait rien — pendant
            # que l'extracteur DINOv3, lui, force son entree sur CUDA.
            # Resultat : « Input type (torch.cuda.FloatTensor) and weight type
            # (torch.FloatTensor) should be the same », au premier
            # conditionnement d'image et nulle part ailleurs.
            # Dire une fois pour toutes ou le calcul a lieu leve l'ambiguite.
            pipe._device = torch.device("cuda")
        self._pipeline = pipe
        self._loaded_at = time.time()
        log.info("pipeline loaded in %.1fs (low_vram=%s)", time.time() - t0, self.low_vram)
        if progress:
            progress("models ready", 1.0)
        return pipe

    def unload(self):
        self._pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        settings: GenerateSettings,
        out_dir: Path,
        progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_preview: Callable[[str, Path], None] | None = None,
    ) -> GenerateResult:
        """Genere, en mesurant la carte pendant tout le travail.

        LE PIC N'EST PAS CELUI DE PYTORCH. `max_memory_allocated()` ne
        compte que l'allocateur de PyTorch ; la wheel o_voxel qui cuit
        l'atlas alloue en CUDA brut, hors de cette comptabilite. Les
        manifestes annoncaient 2,07 Go la ou le pilote lisait 4,8 a 5,6 —
        un chiffre de ce genre est pire qu'aucun chiffre, parce qu'on s'en
        sert pour affirmer que ca tient dans 8 Go. `vrammetre.Metre`
        echantillonne NVML au niveau de la carte, comme la conversion le
        faisait deja de son cote.
        """
        from . import vrammetre

        with vrammetre.Metre() as metre:
            res = self._generate(settings, out_dir, progress,
                                 should_cancel, on_preview)
        if metre.pic_gb > 0.0:
            res.peak_vram_gb = metre.pic_gb
            res.vram_portee = metre.portee
            res.vram_occupation_pct = metre.occupation_pct
        return res

    def _generate(
        self,
        settings: GenerateSettings,
        out_dir: Path,
        progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_preview: Callable[[str, Path], None] | None = None,
    ) -> GenerateResult:
        if not settings.images:
            raise ValueError("at least one reference image is required")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        notes: list[str] = []

        # UNE BARRE DE PROGRESSION NE RECULE JAMAIS, et c'est le seul endroit
        # d'ou tenir cette promesse.
        #
        # Deux emetteurs se marchent dessus. `_stage_previews` annonce
        # « shape ready, texturing » a 0,45 des que la forme est decodee,
        # tandis que l'echantillonneur vendu continue de ticker sa propre
        # barre, qui vaut 0,12 a 0,45 — et il n'en est qu'a son troisieme tic
        # sur douze. La barre montait donc a 45 %, puis retombait a 20 %,
        # remontait a 23 %... Releve dans le journal du 27 aout a 09:01:45,
        # DONC AVANT la coupe du volume-filet : ce n'est pas une consequence
        # de ce changement-la, le defaut est plus ancien.
        #
        # On pourrait courir apres chaque emetteur. On tient plutot
        # l'invariant ici : un tic qui propose MOINS que le plus haut deja
        # atteint est un tic perime, et on le laisse tomber — son libelle
        # avec, sinon « generating shape » viendrait recouvrir « texturing »
        # sans faire avancer quoi que ce soit.
        #
        # L'ANNULATION, ELLE, EST TOUJOURS VERIFIEE. Un tic perime reste un
        # passage du code : le jeter ne doit pas jeter l'ordre de l'artiste.
        plus_haut = 0.0

        def step(stage: str, frac: float):
            nonlocal plus_haut
            if should_cancel and should_cancel():
                raise JobCancelled()
            if frac < plus_haut - 1e-6:
                return
            plus_haut = frac
            log.info("[%.0f%%] %s", frac * 100, stage)
            if progress:
                progress(stage, frac)

        t0 = time.time()
        timer = StageTimer()
        was_loaded = self.loaded

        uses_pixal = settings.detail in config.PIXAL3D_PRESETS
        pipe = self.load(lambda s, f: step(s, f * 0.05),
                         model="pixal3d" if uses_pixal else "trellis2")
        timer.mark("load")
        seed = settings.resolved_seed()

        # LE PALIER DECIDE, sauf si l'appelant a explicitement demande autre
        # chose. Un seul endroit resout les deux cas, et tout ce qui suit lit
        # `r` : plus aucun reglage de qualite ne se decide en aval.
        r = config.recette(settings.detail)
        pipeline_type = r.pipeline_type
        sampler = settings.sampler or r.sampler
        pas_structure = settings.steps_structure or r.steps_structure
        pas_forme = settings.steps_shape or r.steps_shape
        pas_texture = settings.steps_texture or r.steps_texture
        cote_atlas = settings.texture_size or r.texture_size
        # LE BUDGET DE TRIANGLES EST SERVI PAR LA CHAINE, PLUS PAR `finish`.
        # `simplify_with_cumesh` decime AVANT le depliage UV : il n'y a donc
        # plus de couture d'atlas a souder ensuite, et le palier peut redevenir
        # un budget de triangles sans ressusciter la mosaique.
        cible_triangles = settings.finish.target_faces or r.faces
        guidage_texture = (settings.tex_guidance if settings.tex_guidance
                           is not None else r.tex_guidance)

        step("reading reference", 0.07)
        image = Image.open(settings.images[0])

        # Always cut the background out ourselves. The vendored pipeline no
        # longer does it (its rembg_model is never instantiated), and handing
        # it an uncut photo makes it model the backdrop as geometry.
        step("cutting out the subject", 0.09)
        from . import matting

        # LES AUTRES COTES DU SUJET, detoures au meme endroit que la face :
        # BiRefNet est deja monte, chaque vue de plus ne coute que son
        # passage. Une vue illisible est ecartee avec une note, jamais une
        # erreur — trois bonnes vues valent mieux qu'un travail rendu pour
        # une quatrieme corrompue.
        vues_pil: dict[str, Image.Image] = {}
        a_lire: dict[str, Image.Image] = {"front": image}
        for angle in ("droite", "gauche", "dos"):
            chemin_vue = (settings.vues or {}).get(angle)
            if not chemin_vue:
                continue
            try:
                a_lire[_NOM_DU_ROLE[angle]] = Image.open(chemin_vue)
            except Exception as exc:                          # noqa: BLE001
                log.warning("vue %s illisible : %s", angle, exc)
                notes.append(
                    f"La vue « {angle} » n'a pas pu être lue et n'a pas "
                    f"servi : {exc}")

        # LES QUATRE VUES SONT DETOUREES ENSEMBLE, A LA MEME ECHELLE.
        #
        # Vue par vue, le recadrage prend pour cote la plus grande dimension
        # du sujet. Sur un objet plus haut que large sous tous les angles ca
        # ne change rien — et le lot ne change rien non plus, au pixel pres.
        # Sur un objet ALLONGE, si : une moto cadree sur sa longueur de profil
        # arrive 1,4 fois plus petite que cadree sur sa hauteur de face, la
        # fusion la croit aussi large que longue, et elle l'epaissit. Vu a
        # l'ecran par Quentin le 10 septembre 2026 : « elle est juste un peu
        # epaisse ».
        #
        # `separe` remet le detourage vue par vue, pour le banc.
        if len(a_lire) > 1 and getattr(settings, "cadre_mv", "auto") != "separe":
            lot = matting.matter().preparer_lot(a_lire)
        else:
            lot = {r: matting.matter().preparer_avec_geometrie(im)
                   for r, im in a_lire.items()}
        image, geo_front = lot["front"]
        geo_vues = {"front": geo_front}
        for angle in ("droite", "gauche", "dos"):
            role = _NOM_DU_ROLE[angle]
            if role in lot and role != "front":
                vues_pil[angle], geo_vues[role] = lot[role]
        # LE DETOURAGE REND SA MEMOIRE AVANT LE TRAVAIL, PAS APRES. BiRefNet_HR
        # pese 424 Mo sur la carte, et `server.py` ne le liberait qu'APRES le
        # retour de `generate()` : il etait donc sous TOUS les pics, y compris
        # celui de la texturation, qui est le plus haut de la chaine. Il a fini
        # son travail ici, quelques lignes plus haut.
        if self.low_vram:
            matting.liberer()

        # DE QUEL COTE LES DEUX PROFILS ONT ETE PRIS : MESURE, PAS SUPPOSE.
        #
        # L'emplacement ou l'utilisateur a range une photo dit ou elle a ete
        # CLASSEE, pas d'ou elle a ete PRISE. Se tromper de cote ne degrade pas
        # un peu le resultat, il le retourne : un torse frontal et deux lames
        # degainees a l'arriere, sculptes, pas peints. Une convention ne suffit
        # pas — sur six sujets mesures, cinq la suivent et le sixieme sort avec
        # deux visages sans que rien ne previenne.
        #
        # ICI, ET PAS AILLEURS. La carte vient d'etre rendue par le detourage
        # et rien d'autre n'est encore resident : c'est le seul moment de la
        # chaine ou un modele de 2,6 Go passe sans rien deranger. Il rend la
        # carte avant de sortir.
        #
        # LA MESURE NE PEUT PAS FAIRE ECHOUER UN TRAVAIL. Sans le modele, ou si
        # elle hesite, on retombe sur la convention et la note le dit : c'est
        # un choix binaire que l'utilisateur corrige d'un mot.
        cotes_mesures = None
        #: LES ANGLES, ET PAS SEULEMENT LE COTE. La même mesure qui tranche
        #: gauche/droite sait d'où chaque photo a été prise ; on ne gardait
        #: que le choix binaire et on jetait les angles. Voir l'en-tête de
        #: `multivue/cameras.py` pour ce que ça coûtait.
        azimuts_mv = None
        if (vues_pil and _multivue_porte(len(vues_pil), settings, geo_vues)
                and getattr(settings, "lateralite", "auto") == "auto"):
            from .multivue import cotes as _cotes
            if _cotes.disponible(config):
                step("checking the two profiles", 0.10)
                # LES VUES DEJA DETOUREES, pas les photos d'origine : c'est
                # sur celles-la que la regle de decision a ete mesuree, et
                # elles sont deja dans la memoire.
                mesurables = {"front": image}
                mesurables.update({_NOM_DU_ROLE[a]: v
                                   for a, v in vues_pil.items()})
                rapport_cotes = _cotes.mesurer(config, mesurables)
                cotes_mesures = rapport_cotes.get("decision")
                azimuts_mv = (None
                              if getattr(settings, "angles_mv", "auto") == "nominaux"
                              else _cotes.azimuts_mesures(rapport_cotes))
                if azimuts_mv:
                    log.info("angles de prise de vue mesurés : %s",
                             ", ".join("%s %.1f°" % (r, a)
                                       for r, a in azimuts_mv.items()))
                log.info("cotes mesures : %s%s", cotes_mesures,
                         "" if "erreur" not in rapport_cotes
                         else " (%s)" % rapport_cotes["erreur"])
            else:
                log.info("cotes non mesurables : %s",
                         ", ".join(_cotes.manquants(config)))

        if len(settings.images) > 1:
            notes.append(
                f"{len(settings.images)} references given; geometry uses the first. "
                "Multi-view texturing is a separate pass."
            )

        # UNE ETAPE FIXE, DONC TRADUISIBLE. Le palier etait interpole dans le
        # libelle — « generating shape (standard) » — donc absent des sept
        # catalogues, donc affiche en anglais brut a tout le monde. Et c est
        # l etape qui couvre 0,12 a 0,45 de la barre : le libelle le plus
        # longtemps lu de toute une generation. Le palier, lui, est deja
        # ecrit dans la carte et dans le manifeste.
        step("generating shape", 0.12)
        # LE GUIDAGE N'EST PAS UN SEUL NOMBRE.
        #
        # `fidelity` etait injectee telle quelle dans les TROIS etages. Les
        # poids livres disent 7,5 pour la structure et la forme, et 1,0 pour
        # la texture ; la valeur par defaut de 3,0 affaiblissait donc de
        # moitie le guidage de forme ET faisait passer la texture de une a
        # DEUX evaluations par pas — le melange qui coute le plus cher pour
        # le resultat le moins fidele.
        #
        # Ici : structure et forme gardent la valeur des poids sauf demande
        # explicite, et la texture suit le palier (1,0 en Apercu, ou une
        # evaluation suffit ; 5,0 au-dessus).
        guidage_forme = ({"guidance_strength": settings.fidelity}
                        if settings.fidelity is not None else {})

        # The vendored run() ticks a ComfyUI progress bar, and one of its call
        # sites is missing the None guard the others have - so it needs a real
        # bar, not None. Ours forwards to the UI, which turns that constraint
        # into live progress inside the sampling stage instead of a frozen bar.
        from . import comfy_shim

        pbar = comfy_shim.ProgressBar(_RUN_PROGRESS_TICKS)
        # LA BARRE S'ARRETE A 0,45, PAS A 0,60. `run()` ne fait plus tout le
        # travail : le nettoyage, le remaillage, la decimation et surtout la
        # texturation viennent apres lui et occupent 0,47 a 0,90. Laisser
        # l'echantillonnage monter a 0,60 faisait reculer la barre au premier
        # jalon de la chaine — le seul mouvement qu'un utilisateur lit comme
        # une panne. 0,45 est aussi ce qu'annonce `_stage_previews` quand la
        # forme est prete : les deux disent enfin la meme chose.
        comfy_shim.set_progress_sink(
            lambda cur, tot: step("generating shape",
                                  0.12 + 0.33 * min(cur / max(tot, 1), 1.0))
        )

        sampler_params = dict(
            sparse_structure_sampler_params={
                "steps": pas_structure, **guidage_forme},
            shape_slat_sampler_params={
                "steps": pas_forme, **guidage_forme},
            tex_slat_sampler_params={
                "steps": pas_texture,
                "guidance_strength": guidage_texture},
        )

        with _stage_previews(pipe, out_dir, step, on_preview):
            if uses_pixal:
                # Pixal3D projects the photo's pixels onto the volume, so it
                # needs to know where the camera was. MoGe estimates that from
                # the single image - no input from the artist.
                # Cette voie est celle de la recette « extreme » (Pixal3D),
                # qu'aucune interface ne sait demander ; le palier Extrême
                # du produit est la recette « max », qui passe par le
                # mélange multi-vue comme les autres. Le nom est dit juste
                # pour le jour où cette voie s'ouvrira.
                if vues_pil:
                    notes.append(
                        "La recette Pixal3D projette la photo de face ; les "
                        "autres vues n'ont pas servi.")
                    vues_pil = {}
                step("estimating the camera", 0.10)
                pipe.load_moge_model()
                camera = pipe.get_moge_camera_config(image)
                pipe.unload_moge_model()
                notes.append(
                    f"Caméra estimée : champ {camera.get('camera_angle_x', 0):.2f} rad, "
                    f"distance {camera.get('distance', 0):.2f}"
                )
                meshes = pipe.run_pixal3d(
                    image=image, camera_params=camera, num_samples=1, seed=seed,
                    pipeline_type=pipeline_type, generate_texture_slat=True,
                    **sampler_params)
            elif vues_pil and _multivue_porte(len(vues_pil), settings, geo_vues):
                # LE CHEMIN PORTE. Chaque vue est projetee sur la grille par
                # sa camera, et un modele entraine pour cette fusion decide,
                # au lieu d'une moyenne ponderee par la position supposee de
                # chaque camera. C'est ce qui fait disparaitre la lentille
                # inventee a l'arriere d'un feu tricolore et la deuxieme lame
                # d'un samourai.
                #
                # IL NE PREND LA MAIN QUE SI SES POIDS SONT POSES. Sinon on
                # retombe sur l'ancien melangeur, qui rend un objet imparfait
                # plutot qu'une erreur : quelqu'un qui n'a pas telecharge la
                # brique doit quand meme obtenir quelque chose.
                from .multivue import chemin as _mv
                log.info("multi-vue porte : face + %s", ", ".join(vues_pil))
                notes.append("Généré depuis %d vues." % (1 + len(vues_pil)))
                images_mv = _mv_images(image, vues_pil)
                cadre = _cadre_commun(geo_vues)
                if TRANSPORT_DE_CADRE and cadre is None:
                    notes.append(
                        "Les quatre photos n'ont pas le même cadrage : chacune "
                        "a été utilisée telle qu'elle a été recadrée.")
                # L'ACCORD DES PHOTOS SE MESURE AVANT DE GENERER : c'est
                # lui qui decide si la ponderation a un litige a arbitrer.
                _accord_avant = _accord_des_vues(geo_vues)
                # LES PAS DU PALIER, PAS CEUX DE STANDARD. Les deux etages
                # de la cascade se partagent le budget de forme, comme les
                # douze pas de la voie a une photo servent sa cascade unique.
                # LE JOURNAL DOIT DIRE SOUS QUEL REGIME ON A TOURNE.
                # Le 9 septembre 2026, devant un buste deforme, rien ne
                # permettait de savoir si la ponderation avait joue ou si les
                # quatre vues avaient ete melangees a plat : la force etait
                # calculee, utilisee, et jamais ecrite nulle part.
                maillage_mv, mesure_mv = _mv.generer(
                    config, images_mv, seed, accord=_accord_avant,
                    lateralite=getattr(settings, "lateralite", "auto"),
                    decision_mesuree=cotes_mesures, geometrie=cadre,
                    azimuts=azimuts_mv,
                    force_imposee=getattr(settings, "force_mv", None),
                    pas_structure=r.steps_structure,
                    pas_forme=max(1, r.steps_shape // 2),
                    jalon=step)
                log.info("multi-vue porte : %d cellules, lateralite %s, "
                         "force %.1f (%s)",
                         mesure_mv["cellules"],
                         mesure_mv["lateralite"]["source"],
                         mesure_mv.get("force", -1.0),
                         "imposee" if getattr(settings, "force_mv", None)
                         is not None else "mesuree")
                # LA MENTION EST LA MOITIE DU CORRECTIF. Le module promet que
                # le resultat dit quand la lateralite a ete supposee ; sans
                # cette note, celui dont les profils sortent du mauvais cote
                # cherche ce qui ne va pas au lieu de changer un mot.
                # QUATRE PHOTOS QUI NE MONTRENT PAS LE MEME OBJET, C'EST UNE
                # INFORMATION UTILE. Elle ne change plus rien au traitement,
                # mais celui qui a pris les photos peut agir dessus.
                _accord = _accord_avant
                if _accord["mesurable"] and _accord["ecart"] > ECART_DE_VUES_MAX:
                    notes.append(
                        "Vos quatre vues ne montrent pas l'objet à la même "
                        "taille (%.1f %% d'écart). L'objet a été fabriqué "
                        "quand même." % _accord["ecart"])

                # QUAND LES ANGLES SONT LUS, ILS DISENT DEJA LE COTE, et
                # parler de profils échangés n'aurait plus de sens : un angle
                # de 306 degrés dit où était l'appareil, pas dans quel
                # emplacement la photo a été rangée. On dit donc autre chose,
                # et on ne le dit que si le tour n'était pas régulier —
                # l'utilisateur n'a rien à faire d'un écart de deux degrés.
                _azimuts_dits = mesure_mv.get("azimuts")
                _cotes_dits = mesure_mv["lateralite"]
                if _azimuts_dits:
                    # CE QU'ON MESURE, C'EST LA RÉGULARITÉ DU TOUR, PAS
                    # L'ÉTIQUETTE. Première version de cette note : elle
                    # comparait chaque angle à celui de l'emplacement où
                    # l'utilisateur avait rangé la photo, et annonçait « 173°
                    # d'écart » sur un jeu de photos parfaitement utilisable —
                    # les deux profils étaient simplement rangés dans l'autre
                    # sens, ce qui n'est pas un défaut de prise de vue.
                    #
                    # On trie donc les quatre angles et on regarde les écarts
                    # entre vues successives : un tour régulier en donne quatre
                    # de 90°. Sur le samouraï : 54, 133, 76 et 97 — soit 43° de
                    # plus grand écart au quart de tour.
                    _tries = sorted(a % 360 for a in _azimuts_dits.values())
                    _pas = [(_tries[(i + 1) % 4] - _tries[i]) % 360 for i in range(4)]
                    _ecart = max(abs(p - 90) for p in _pas)
                    if _ecart >= 15:
                        notes.append(
                            "Vos quatre photos ne font pas le tour à pas "
                            "réguliers (jusqu'à %d° d'écart avec le quart de "
                            "tour). L'objet a été construit depuis les angles "
                            "réels ; un tour régulier donnerait mieux."
                            % round(_ecart))
                elif _cotes_dits["source"] == "convention":
                    notes.append(
                        "Côtés supposés : si les profils du modèle sortent "
                        "inversés, relancez en indiquant que les deux profils "
                        "sont échangés.")
                elif _cotes_dits["source"] == "mesuree":
                    notes.append(
                        "Côtés lus sur les photos : les deux profils ont été "
                        "échangés." if _cotes_dits["swap_sides"] else
                        "Côtés lus sur les photos : les deux profils étaient "
                        "à leur place.")
                elif _cotes_dits["swap_sides"]:
                    notes.append("Les deux profils ont été échangés, comme demandé.")
                # LE MULTIVUE NE SE REPLIE PAS D'UN PALIER, IL REDUIT.
                # Se replier jetterait quatre encodages, une fusion et deux
                # cascades pour recommencer plus grossierement, alors que le
                # depassement est souvent de quelques pour cent : le feu
                # tricolore sort a 8 234 430 faces contre 7 995 605.
                from .multivue import budget as _mv_budget
                maillage_mv, _reduit = _mv_budget.reduire_si_trop_lourd(
                    maillage_mv, _plafond_de_faces(), notes)
                meshes = [maillage_mv]
                del images_mv
            elif vues_pil:
                # PLUSIEURS COTES : `run_multiview`, la voie native de
                # TRELLIS.2 — un conditionnement PAR VUE et un melange
                # spatial des directions, pour la structure comme pour la
                # forme. Le retour est le meme `MeshWithVoxel` que `run` :
                # toute la chaine de nettoyage s'applique telle quelle.
                #
                # CONVENTION D'ANGLES, MESUREE (buste et samourai, 4 sept.
                # 2026) : « droite » suit `multivue.AZIMUTS` (la camera
                # tourne vers la droite de l'IMAGE de face) et vaut le
                # `right` du melange, `gauche` son `left`, `dos` son `back`.
                # Les poids par vue sont calcules dans le repere de la
                # grille (flow_euler._scores_des_vues) ; si un sujet montre
                # un jour ses flancs echanges, c'est LA que ca se regle.
                # La note est TRADUITE PAR L'APP (notes.ts, par gabarit) : un
                # seul nombre à capturer ; la liste des côtés va au journal.
                log.info("multi-vue : face + %s", ", ".join(vues_pil))
                notes.append("Généré depuis %d vues." % (1 + len(vues_pil)))
                meshes = pipe.run_multiview(
                    front=image,
                    generate_texture_slat=False,
                    blend_temperature=_melange_multivue(),
                    right=vues_pil.get("droite"),
                    left=vues_pil.get("gauche"),
                    back=vues_pil.get("dos"),
                    seed=seed,
                    pbar=pbar,
                    pipeline_type=pipeline_type,
                    sampler=sampler,
                    dino_lock=settings.dino_lock,
                    dino_substeps=settings.dino_substeps,
                    dino_foundation_cap=settings.dino_foundation_cap,
                    sparse_structure_resolution=settings.structure_resolution,
                    max_num_tokens=_plafond_de_tokens(),
                    hole_fill_algorithm=settings.hole_fill,
                    **sampler_params,
                )
            else:
                # LE FILET EST COUPE, ET C'EST LE PLUS GROS GAIN GRATUIT DE
                # LA CHAINE. Par defaut `run()` echantillonne un slat de
                # texture puis decode un volume PBR qui ne sert QU'A UNE
                # CHOSE : le repli `to_glb` de `_texturer_avec_repli`, quand
                # la texturation echoue. La condition posee pour le couper
                # etait « voir la chaine finir deux fois » ; elle a fini
                # ONZE fois sur onze pendant la campagne de mesure, et le
                # repli n'a jamais servi.
                #
                # Mesure sur la 4060, poupee Standard, graine 4242, a froid :
                # echantillonnage 154,0 -> 80,8 s (-73 s), travail entier
                # 387,6 -> 299,8 s, pic carte 5,36 -> 5,28 Go, et le controle
                # rend la MEME qualite (0 trou, 122 texels par triangle,
                # 9,1 degres d'ecart de normales).
                #
                # Le prix est reel et il est assume : sans volume, une
                # texturation qui echoue fait echouer le travail au lieu de
                # rendre une texture de volume. `_texturer_avec_repli` le
                # dit alors honnetement — voir le garde-fou la-bas. Pixal3D
                # garde son volume : lui s'en sert pour cuire, pas pour se
                # rattraper.
                meshes = pipe.run(
                    image,
                    num_samples=1,
                    seed=seed,
                    pbar=pbar,
                    pipeline_type=pipeline_type,
                    sampler=sampler,
                    generate_texture_slat=False,
                    dino_lock=settings.dino_lock,
                    dino_substeps=settings.dino_substeps,
                    dino_foundation_cap=settings.dino_foundation_cap,
                    sparse_structure_resolution=settings.structure_resolution,
                    max_num_tokens=_plafond_de_tokens(),
                    hole_fill_algorithm=settings.hole_fill,
                    **sampler_params,
                )
        comfy_shim.set_progress_sink(None)
        mesh = meshes[0]
        timer.mark("sample")

        if uses_pixal:
            # PIXAL3D GARDE L'ANCIEN CHEMIN, et c'est definitif. Il reprojette
            # les pixels de la photo dans le volume ; `texture_mesh`
            # regenererait ce qu'il vient d'aller chercher. Le noeud du banc
            # le refuse d'ailleurs explicitement.
            #
            # o_voxel cuit le volume PBR dans des cartes UV ; `decimation_target`
            # y est un plafond technique, pas le budget de l'artiste — decimer
            # avant la cuisson reviendrait a cuire du detail sur une geometrie
            # qui n'existe plus.
            step("baking PBR materials", 0.70)
            import o_voxel
            from . import leurre_nvdiffrast as _leurre
            _leurre.verifier()      # au journal si le vrai nvdiffrast est monté

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=max(cible_triangles, 300_000),
                texture_size=cote_atlas,
                remesh=False,
                verbose=False,
            )
            chemin = "to_glb"
        else:
            from . import remaillage

            # LE PLAFOND, ET IL EXISTE PARCE QU IL A MANQUE.
            #
            # Dans la nuit du 28 aout 2026, un multi-vue au palier
            # Standard a sorti une forme de 12 338 894 faces et le
            # moteur l a envoyee au remaillage sans se demander une
            # seconde si la carte pouvait l encaisser. Huit minutes plus
            # tard le pilote graphique decrochait, et la machine tombait
            # en ecran bleu 0x116 VIDEO_TDR_FAILURE, parametre
            # 0xC000009A (ressources insuffisantes). Rien, nulle part, ne
            # bornait cette taille.
            #
            # CE QUE CE PLAFOND EST, ET CE QU IL N EST PAS. Ce n est pas
            # un jugement de qualite : c est une ceinture de securite
            # pour la machine. Il est regle AU DESSUS de tout ce qui a
            # ete mesure de sain (7 520 462 faces ont fini sans encombre
            # sur la meme carte de 8 Go) et SOUS le cas qui a tue le
            # pilote. Le nombre de faces d un mono-vue legitime au palier
            # Standard n a jamais ete releve : tant qu il ne l est pas, un
            # seuil plus serre risquerait de refuser du travail sain.
            #
            # Il vaut mieux un travail qui echoue en une ligne claire
            # qu une machine qui redemarre.
            _faces_brutes = len(mesh.faces)
            _plafond = _plafond_de_faces()
            log.info("le remaillage recoit %d faces (plafond %d)",
                     _faces_brutes, _plafond)
            if _faces_brutes > _plafond:
                # LE REPLI PLUTOT QUE LE REFUS. Une forme trop lourde vient
                # presque toujours d'un multi-vue au-dessus de Brouillon ;
                # la meme demande un palier plus bas passe (banc du
                # 4 septembre 2026 : 5,9 M de faces en Standard, 1,6 M en
                # Brouillon). On libere la carte, on recommence, et la note
                # du maillage dit ce qui s'est passe.
                _palier_bas = PALIER_DE_REPLI.get(settings.detail)
                if _palier_bas and not os.environ.get("LUMENGEN_SANS_REPLI"):
                    log.warning("forme trop lourde au palier %s (%d faces, "
                                "plafond %d) : reprise au palier %s",
                                settings.detail, _faces_brutes, _plafond,
                                _palier_bas)
                    del mesh, meshes
                    remaillage.purger()
                    from dataclasses import replace as _replace
                    # LA RECETTE DU PALIER BAS, PAS SEULEMENT SON NOM : les
                    # pas, l'atlas, l'échantillonneur et le guidage sont
                    # remis à None pour que _generate les reprenne de la
                    # recette du palier bas. Le budget de faces, lui, n'est
                    # jamais résolu par le serveur : s'il est explicite, il
                    # reste celui que l'utilisateur a demandé ; sinon
                    # `cible_triangles` prend celui de la recette du palier.
                    # La graine résolue est gardée : même objet, palier
                    # plus léger. Et la barre ne recule pas : la progression
                    # de la reprise est remise à l'échelle dans ce qui
                    # reste au-dessus du point atteint.
                    _base = plus_haut
                    _progres = ((lambda st, fr: progress(st, _base + fr * (1 - _base)))
                                if progress else None)
                    res = self._generate(
                        _replace(settings, detail=_palier_bas, seed=seed,
                                 steps_structure=None, steps_shape=None,
                                 steps_texture=None, texture_size=None,
                                 sampler=None, tex_guidance=None),
                        out_dir, _progres, should_cancel, on_preview)
                    res.notes.insert(0, (
                        "Forme trop lourde pour la carte (%s faces) : "
                        "reprise au palier inférieur."
                        % f"{_faces_brutes:,}".replace(",", " ")))
                    return res
                raise RuntimeError(
                    "Shape too heavy to remesh: %d faces, ceiling %d. "
                    "The job was refused before saturating the graphics "
                    "card." % (_faces_brutes, _plafond))

            step("cleaning the mesh", 0.47)
            remaillage.nettoyer(mesh,
                                dc_resolution=r.dc_resolution,
                                cible_triangles=cible_triangles,
                                perimetre_bouchage=config.BOUCHAGE_PERIMETRE,
                                perimetre_fermeture=config.FERMETURE_PERIMETRE,
                                ratio_aire=config.DEBRIS_RATIO_AIRE,
                                distance_mini=config.DEBRIS_DISTANCE_MINI,
                                part_maxi=config.DEBRIS_PART_MAXI,
                                jalon=step, notes=notes)
            tm = remaillage.vers_trimesh(mesh)

            # LA PURGE DU BANC, EXACTEMENT ICI. Son graphe pose un
            # `Trellis2CudaReset` sur le lien qui porte le trimesh vers la
            # texturation, avec pour commentaire « purge juste avant le
            # texturing : c'est la que le pic arrive ».
            #
            # LE VOLUME RESTE VIVANT TANT QUE `generate_texture_slat` VAUT
            # True : `attrs` et `coords` sont le repli `to_glb` si la
            # texturation echoue, et on ne coupe pas le filet avant d'avoir
            # vu la chaine finir deux fois. Ils partiront avec lui.
            remaillage.purger()

            # TRACE D ENQUETE, sous drapeau. Quand LUMENGEN_TRACE_MAILLAGE vaut
            # 1, on depose le maillage TEL QU IL ENTRE dans la texturation.
            # C est la seule facon de savoir si un defaut vient de notre chaine
            # de nettoyage ou de `texture_mesh` : le .glb final ne dit que le
            # cumul des deux. Rien n est ecrit sans le drapeau.
            if os.environ.get("LUMENGEN_TRACE_MAILLAGE") == "1":
                try:
                    tm.export(out_dir / "avant_texture.glb")
                    log.info("trace : maillage avant texturation depose")
                except Exception as exc:                      # noqa: BLE001
                    log.warning("trace non deposee : %s", exc)

            step("texturing", 0.65)
            glb, replie = self._texturer_avec_repli(
                pipe, tm, image, mesh, seed=seed, r=r,
                cote_atlas=cote_atlas, pas_texture=pas_texture,
                guidage_texture=guidage_texture, step=step, notes=notes,
                vues=vues_pil)
            chemin = "to_glb" if replie else "texture_mesh"
            del tm
            mesh.attrs = None
            mesh.coords = None
            del mesh, meshes
            remaillage.purger()

        timer.mark("bake")

        step("finishing", 0.90)
        # `finish` NE DECIME PLUS sur ce chemin : la chaine a deja servi le
        # budget, et c'est la SOUDURE de `finish` (« welded: 114306 » sur un
        # maillage de 283 000 sommets) qui casse les coutures UV et donne la
        # mosaique. On ne la laisse tourner que si l'artiste a demande une
        # chose qu'elle seule sait faire : des quads, ou un remaillage force.
        reglages = settings
        if chemin == "texture_mesh" and not (
                settings.finish.topology == "quad" or settings.finish.force_remesh):
            reglages = replace(settings,
                               finish=replace(settings.finish, target_faces=None))
        glb, finish_report = self._finish_scene(glb, reglages, step)

        # `_repad_atlas` repare les mouchetures que `to_glb` laisse dans les
        # gouttieres entre iles UV. Sur le chemin `texture_mesh`,
        # `postprocess_mesh` les a DEJA inpaintees au cv2 — quatre passes,
        # rayon 3 pour la couleur, 1 pour metallic/roughness/alpha. Le garder
        # ne corrigerait rien et couterait une seconde passe uvraster au
        # format de l'atlas.
        if chemin == "to_glb":
            step("cleaning the atlas", 0.96)
            self._repad_atlas(glb, notes)

        glb = self._orient_and_scale(glb, settings)

        from . import materials

        material_report = materials.apply_alpha_mode(glb, settings.transparency)
        if material_report:
            notes.append(
                f"Matériau : {_MODE_LABELS[material_report.mode]}"
                + (" (forcé)" if material_report.forced
                   else f" — {material_report.reason}")
            )

        # ÉCRIT À CÔTÉ, PUIS BASCULÉ. C'est la discipline que le reste du
        # produit applique partout — les manifestes, les vignettes, les
        # modèles quantifiés — et le livrable principal en était le seul
        # exempt. Un disque plein en cours d'export laissait un .glb tronqué
        # dans la bibliothèque : il porte le bon nom, il s'affiche dans la
        # liste, et il ne s'ouvre pas. Un fichier qui manque se comprend ;
        # un fichier qui ment, non.
        # LES NORMALES DOIVENT ETRE DANS LE CACHE AU MOMENT DE L'EXPORT.
        # trimesh n'ecrit l'accesseur glTF NORMAL que si « vertex_normals » se
        # trouve deja dans `mesh._cache.cache` — il ne les calcule pas pour
        # l'occasion. `texture_mesh` ne les y met qu'avec `use_custom_normals`,
        # et `_recoudre_orientation` peut avoir vide ce cache en reparant
        # l'enroulement. Sans NORMAL, le glTF est rendu a plat : un maillage
        # lisse qui parait facette, et qu'on prend pour un defaut de
        # remaillage. Y toucher suffit a le remplir.
        for g in _geometries(glb):
            try:
                _ = g.vertex_normals
            except Exception as exc:                          # noqa: BLE001
                log.warning("normales non recalculees : %s", exc)

        out_path = out_dir / f"model_{seed}.glb"
        # Le .glb reste EN DERNIER dans le nom : trimesh choisit son
        # exporteur sur l extension, et un fichier finissant par .tmp lui fait
        # rendre « tmp exporter not available ». Mesure a la premiere
        # generation qui a suivi.
        provisoire = out_path.with_name(out_path.stem + ".partiel.glb")
        # PAS DE WEBP DANS LE MODELE LIVRE. `extension_webp=True` faisait
        # ecrire `EXT_texture_webp` dans `extensionsRequired` : une extension
        # requise est un tout-ou-rien, et le lecteur qui ne la connait pas
        # refuse le FICHIER ENTIER — geometrie comprise. Unreal Engine repond
        # « There was no data to import », Maya et 3ds Max de meme. Blender la
        # lit depuis la 3.6, ce qui l'a masque tout du long : ce qu'on
        # ouvrait ici passait, ce que le client importait, non.
        #
        # L'atlas pese ~8 fois plus en PNG (2,9 Mo -> 22 Mo en 4096). C'est le
        # prix d'un fichier qui s'ouvre partout, et c'est le poids normal d'un
        # atlas PBR. Voir `gltf_compat`, qui repare de la meme facon les
        # modeles deja sur le disque au moment de les exporter.
        glb.export(provisoire)
        provisoire.replace(out_path)
        timer.mark("finish")

        # A couple of seconds, and it is what makes several seeds comparable at
        # a glance instead of one click and one viewport load at a time.
        step("rendering the thumbnail", 0.98)
        thumb: Optional[Path] = None
        try:
            from . import preview as preview_mod

            # LA VIGNETTE PORTE LE NOM DE SON MODÈLE, pas un nom fixe.
            #
            # `thumbnail.png` était le seul fichier qu'une génération voisine
            # écrasait À COUP SÛR, sans même avoir besoin de la même graine :
            # deux travaux tombés dans le même dossier, et les deux cartes
            # affichaient la même image — celle du dernier arrivé. C'est ce
            # que le rapport décrivait comme « il m'a régénéré le même,
            # exactement ». Le dossier ne se partage plus, mais une vignette
            # doit de toute façon appartenir au modèle qu'elle montre.
            #
            # Personne ne lit ce nom : le chemin voyage dans le résultat.
            thumb = preview_mod.render_turntable(
                out_path, out_path.with_suffix(".png"),
                views=(30.0,), resolution=384)
        except Exception as exc:
            log.warning("thumbnail failed: %s", exc)

        # LES APERCUS D ETAPE NE SURVIVENT PAS AU RESULTAT.
        #
        # `preview_structure.glb` et `preview_shape.glb` sont montres dans la
        # carte PENDANT le calcul, pour qu'on voie la forme arriver. Une fois
        # le modele final ecrit, ils ne decrivent plus rien — et ils sont plus
        # gros que lui : mesure sur vingt-deux generations, 50,7 Mo
        # d'intermediaires pour 9,2 Mo de livrable, soit 85 % de la
        # bibliotheque en fichiers que rien ne relit. Un client a vingt
        # modeles par jour accumulait un gigaoctet quotidien de dechet.
        for trace in ("preview_structure.glb", "preview_shape.glb",
                      "preview_texture.glb"):
            try:
                (out_dir / trace).unlink(missing_ok=True)
            except OSError as exc:                            # noqa: BLE001
                log.warning("apercu %s non efface : %s", trace, exc)

        faces = int(sum(len(g.faces) for g in _geometries(glb)))
        verts = int(sum(len(g.vertices) for g in _geometries(glb)))

        gc.collect()
        torch.cuda.empty_cache()
        step("done", 1.0)

        stages = timer.durations()
        stages["models_were_resident"] = was_loaded

        return GenerateResult(
            glb_path=out_path,
            thumbnail=thumb,
            seed=seed,
            detail=settings.detail,
            faces=faces,
            vertices=verts,
            duration_s=time.time() - t0,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1024 ** 3,
            file_size_mb=out_path.stat().st_size / 1024 ** 2,
            finish_report=finish_report,
            stages=stages,
            notes=notes,
        )

    # -- helpers ------------------------------------------------------------

    def _texturer_avec_repli(self, pipe, tm, image, mesh, *, seed, r,
                             cote_atlas, pas_texture, guidage_texture,
                             step, notes, vues=None):
        """Texture le maillage remaille, avec `to_glb` en filet.

        Rend `(glb, replie)`. Le filet n'existe que tant que `run()` produit
        encore le volume PBR (`generate_texture_slat=True`) : c'est ce qui
        permet de faire tourner la chaine neuve sans risquer de rendre un
        travail sans texture. Il disparaitra avec le volume.

        LA MESURE IMBRIQUEE EST TEMPORAIRE. Le pic propre de `texture_mesh`
        n'a jamais ete isole, ni au banc ni ici : tous les chiffres du dossier
        sont des pics de travail entier. C'est la mesure qui manque le plus,
        et elle ne se lit qu'ici. Deux reserves a garder en tete en lisant sa
        sortie :
          - `Metre.__enter__` remet a zero le compteur de pointe de PyTorch,
            donc le metre EXTERIEUR perd sa composante torch pour le debut du
            travail. Son delta NVML, lui, continue d'etre echantillonne : le
            pic annonce au client reste bon tant qu'il vient de la carte.
          - le volume `attrs`/`coords` est encore resident pendant la mesure,
            puisqu'il est le filet. Sa taille est imprimee a cote pour qu'on
            puisse retrancher ce que l'etape 3 fera disparaitre.
        """
        from . import remaillage, vrammetre

        volume_mo = 0.0
        for t in (getattr(mesh, "attrs", None), getattr(mesh, "coords", None)):
            if t is not None:
                volume_mo += t.numel() * t.element_size() / 1024 ** 2

        alloue_avant = torch.cuda.memory_allocated() / 1024 ** 2
        reserve_avant = torch.cuda.memory_reserved() / 1024 ** 2
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        try:
            with vrammetre.Metre() as metre:
                if vues:
                    # La texture regarde les MEMES vues que la forme :
                    # `texture_mesh_multiview` pondere chaque texel par
                    # l'angle entre sa normale et chaque camera. Un dos
                    # observe n'est plus invente.
                    glb, _base, _mr = remaillage.texturer_multivue(
                        pipe, tm, image, vues, seed=seed,
                        resolution=r.tex_resolution,
                        texture_size=cote_atlas,
                        steps=pas_texture,
                        guidance=guidage_texture,
                        jalon=step)
                else:
                    glb, _base, _mr = remaillage.texturer(
                        pipe, tm, image, seed=seed,
                        resolution=r.tex_resolution,
                        texture_size=cote_atlas,
                        steps=pas_texture,
                        guidance=guidage_texture,
                        jalon=step)
        except JobCancelled:
            raise
        except Exception as exc:                              # noqa: BLE001
            # LE REPLI N'EXISTE QUE S'IL Y A UN VOLUME DERRIERE. Depuis que
            # le chemin standard echantillonne avec `generate_texture_slat`
            # a False, `attrs` et `coords` sont absents : appeler `to_glb`
            # ici leverait une erreur d'attribut qui MASQUERAIT la vraie
            # cause, et l'artiste lirait un message parlant de volume pour
            # un probleme de texturation. On releve l'echec tel quel, avec
            # son origine — un travail rate qui se nomme vaut mieux qu'un
            # travail rate qui accuse autre chose.
            #
            # Pixal3D, lui, garde son volume : pour lui le filet est intact
            # et la branche ci-dessous s'execute comme avant.
            if getattr(mesh, "attrs", None) is None \
                    or getattr(mesh, "coords", None) is None:
                log.warning("texturation echouee (%s) — aucun volume en "
                            "repli, l'echec est releve tel quel", exc)
                raise

            log.warning("texturation echouee (%s) — repli sur to_glb", exc)
            notes.append(
                "La texturation a echoue ; la texture vient du volume "
                f"(ancien chemin). Cause : {exc}")
            import o_voxel

            remaillage.purger()
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces,
                attr_volume=mesh.attrs, coords=mesh.coords,
                attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=300_000, texture_size=cote_atlas,
                remesh=False, verbose=False)
            return glb, True

        faces = int(len(glb.faces))
        log.info(
            "MESURE TEXTURATION — %.1f s ; pic carte %.2f Go (%.1f %% de la "
            "carte) ; torch alloue %.0f -> pic %.0f Mo, reserve %.0f -> pic "
            "%.0f Mo ; volume PBR encore resident %.0f Mo ; atlas %d, %d "
            "faces, %.0f texels par triangle",
            time.time() - t0, metre.pic_gb, metre.occupation_pct,
            alloue_avant, torch.cuda.max_memory_allocated() / 1024 ** 2,
            reserve_avant, torch.cuda.max_memory_reserved() / 1024 ** 2,
            volume_mo, cote_atlas, faces,
            (cote_atlas ** 2) / max(faces, 1))
        return glb, False

    def _finish_scene(self, glb, settings: GenerateSettings, step):
        """Run the artist finishing pass on the baked geometry.

        The UVs coming out of to_glb are what the baked textures are addressed
        with, so any pass that rebuilds topology invalidates them. When the
        artist asked for a topology change we re-unwrap and the texture is
        re-baked by the caller; otherwise we decimate while preserving UVs.
        """
        import numpy as np

        geoms = _geometries(glb)
        if not geoms:
            return glb, {}

        geom = geoms[0]

        # LE PALIER CHOISI EST UN BUDGET, PAS SEULEMENT UNE FINESSE.
        #
        # `to_glb` decime jusqu'a un plafond technique de 300 000 faces, pose
        # la pour que la texture soit cuite sur une geometrie encore riche.
        # Ce plafond n'a jamais ete un choix artistique — mais faute de
        # `target_faces`, la finition etait sautee et il devenait le resultat :
        # un "Apercu" et un "Standard" livraient tous deux ~290 000 faces.
        #
        # On resout donc le budget depuis le palier quand l'artiste n'a rien
        # demande. S'il a demande quelque chose, c'est lui qui commande.
        # LA FINITION NE S ARME PLUS TOUTE SEULE.
        #
        # Ces quatre lignes donnaient a chaque palier le budget de triangles de
        # config.FACES_PRESETS, pour corriger un vrai defaut : sans budget, la
        # finition etait sautee et tous les paliers livraient la meme densite,
        # ~290 000 faces. Corriger ca a introduit bien pire.
        #
        # CE QUE LA MESURE A MONTRE. Meme sujet, meme graine, meme palier, la
        # seule difference etant ce budget :
        #
        #   target_faces null  -> finition sautee  -> 293 644 faces -> propre
        #   target_faces 150k  -> finition faite   -> 136 692 faces -> mosaique
        #
        # Et le rapport de finition dit ce qu elle fait : « welded: 114306 ».
        # Elle soude 114 306 sommets sur 283 000, par proximite de position.
        # Aux coutures de l atlas, deux sommets au meme endroit portent des UV
        # DIFFERENTS — c est la definition d une couture. Les souder n en garde
        # qu un, et les triangles d un cote vont chercher leur couleur dans la
        # chart d en face. C est la mosaique.
        #
        # La densite par palier reste un vrai sujet, et il faudra le reprendre —
        # mais une texture juste passe avant une echelle de densite.
        want = settings.finish

        if want.target_faces is None and not want.force_remesh and want.topology == "tri":
            # Rien a finir, mais l'orientation reste a recoudre : elle sort
            # incoherente de `to_glb` quel que soit le chemin.
            _recoudre_orientation(geom)
            return glb, {"skipped": "no finishing requested"}

        v = torch.tensor(np.asarray(geom.vertices), dtype=torch.float32, device="cuda")
        f = torch.tensor(np.asarray(geom.faces), dtype=torch.int32, device="cuda")

        nv, nf, uvs, report = finish(v, f, want, progress=lambda s, p: step(s, 0.85 + p * 0.1))
        payload = report.as_dict()

        topology_changed = uvs is not None and (
            int(nf.shape[0]) != int(f.shape[0])
            or int(nv.shape[0]) != int(v.shape[0])
        )

        # A re-unwrap invalidates every texture baked against the old UVs. They
        # stay perfectly valid-looking images that paint the wrong parts of the
        # model - so they are re-projected here, in the same ray-cast pass that
        # produces the normal map. This is not optional: skipping it is what
        # turned a generated eye into confetti.
        if topology_changed:
            step("transferring the textures", 0.92)
            from . import normalbake
            from PIL import Image

            mat = getattr(getattr(geom, "visual", None), "material", None)
            old_uv = getattr(getattr(geom, "visual", None), "uv", None)

            to_move = {}
            for slot in ("baseColorTexture", "metallicRoughnessTexture",
                         "emissiveTexture", "occlusionTexture"):
                img = getattr(mat, slot, None) if mat else None
                if img is not None:
                    to_move[slot] = np.asarray(img)

            rgb, bake_report, moved = normalbake.bake(
                v, f, nv, nf, uvs,
                size=settings.texture_size or config.recette(settings.detail).texture_size,
                high_uv=(torch.as_tensor(np.asarray(old_uv), dtype=torch.float32)
                         if old_uv is not None else None),
                transfer=to_move,
            )

            if mat is not None:
                for slot, array in moved.items():
                    setattr(mat, slot, Image.fromarray(array))
                if not moved and to_move:
                    report.notes.append(
                        "Transfert des textures impossible — les couleurs peuvent "
                        "être décalées. Utilisez « Brut » pour ce modèle.")

                if settings.bake_normal_map and rgb is not None:
                    mat.normalTexture = Image.fromarray(rgb)
                    payload["normal_map"] = bake_report.as_dict()
                    report.notes.append(
                        f"Normal map {bake_report.size}² — le relief des "
                        f"{bake_report.faces_high:,} faces d'origine est conservé"
                        + (f" ({bake_report.pierced:.0%} de la surface trop fine "
                           f"pour être cuite)" if bake_report.pierced > 0.05 else "")
                    )

        geom.vertices = nv.detach().cpu().numpy()
        geom.faces = nf.detach().cpu().numpy()
        if uvs is not None and geom.visual is not None:
            try:
                geom.visual.uv = uvs.detach().cpu().numpy()
            except Exception as exc:  # pragma: no cover - depends on trimesh visual type
                report.notes.append(f"UVs not reattached ({exc}); texture may be offset")

        # APRES la decimation, jamais avant : `finish` reconstruit la table des
        # faces, et une couture posee en amont ne survit pas au remaillage.
        # Premiere version de ce correctif : recousu a l'entree, incoherent a
        # la sortie — le defaut etait toujours la, mais plus personne ne le
        # cherchait.
        _recoudre_orientation(geom)

        payload["notes"] = report.notes
        return glb, payload

    @staticmethod
    def _repad_atlas(glb, notes: list[str]) -> None:
        """Fix the speckled gutters to_glb leaves between UV islands.

        Applies to every asset, not just decimated ones: the noise is produced
        by the exporter itself and is what makes an otherwise clean model look
        flecked and torn along its seams.
        """
        import numpy as np
        import torch
        from PIL import Image

        from . import normalbake

        for geom in _geometries(glb):
            visual = getattr(geom, "visual", None)
            mat = getattr(visual, "material", None)
            uv = getattr(visual, "uv", None)
            if mat is None or uv is None:
                continue

            maps = {}
            for slot in ("baseColorTexture", "metallicRoughnessTexture",
                         "normalTexture", "emissiveTexture"):
                img = getattr(mat, slot, None)
                if img is not None:
                    maps[slot] = np.asarray(img)
            if not maps:
                continue

            try:
                fixed = normalbake.repad_atlas(
                    torch.as_tensor(np.asarray(geom.vertices), dtype=torch.float32,
                                    device="cuda"),
                    torch.as_tensor(np.asarray(geom.faces), dtype=torch.int32,
                                    device="cuda"),
                    torch.as_tensor(np.asarray(uv), dtype=torch.float32,
                                    device="cuda"),
                    maps,
                )
            except Exception as exc:
                log.warning("atlas re-padding failed: %s", exc)
                continue

            for slot, array in fixed.items():
                setattr(mat, slot, Image.fromarray(array))
            if fixed:
                notes.append(
                    f"Atlas repadé ({len(fixed)} carte"
                    f"{'s' if len(fixed) > 1 else ''}) — supprime les mouchetures "
                    f"le long des coutures UV")

    @staticmethod
    def _orient_and_scale(glb, settings: GenerateSettings):
        """Put the asset where a DCC app expects it: correct up axis, real-world
        size, origin on the ground plane."""
        import numpy as np
        import trimesh

        geoms = _geometries(glb)
        if not geoms:
            return glb

        if settings.up_axis == "z":
            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            for g in geoms:
                g.apply_transform(rot)

        if settings.real_size_mm:
            extents = np.max([g.extents for g in geoms], axis=0)
            largest = float(np.max(extents))
            if largest > 0:
                s = (settings.real_size_mm / 1000.0) / largest
                for g in geoms:
                    g.apply_scale(s)

        if settings.origin_at_base:
            up = 2 if settings.up_axis == "z" else 1
            lowest = min(float(g.bounds[0][up]) for g in geoms)
            offset = [0.0, 0.0, 0.0]
            offset[up] = -lowest
            for g in geoms:
                g.apply_translation(offset)

        return glb


#: Stock pipeline.json cuts the background with briaai/RMBG-2.0, which is a
#: gated repo *and* non-commercial. Its architecture is BiRefNet, and upstream
#: BiRefNet is MIT, ungated, and better on fine structures (filigree, hair) -
#: exactly the cases where a bad matte ruins the geometry. We rewrite that one
#: field into a side-car config rather than patching the vendored source.
_REMBG_REPLACEMENT = "ZhengPeng7/BiRefNet_HR"
_LOCAL_CONFIG_NAME = "pipeline.lumengen.json"


def _recoudre_orientation(geom) -> None:
    """Rendre l'orientation des faces coherente d'une plaque a l'autre.

    Le maillage sorti de TRELLIS n'est pas retourne — son volume est positif —
    mais son enroulement est incoherent : des plaques entieres tournent a
    l'envers de leurs voisines. A l'ecran ca ne se lit pas comme une erreur de
    geometrie, ca se lit comme une matiere sale : des zones eclairees par
    l'interieur, un aspect marbre qu'on prend pour un defaut de texture.

    La chaine splat->mesh reglait ca par un tri au winding number dans
    `splatfield`, qui demande un paquet compile. Le chemin TRELLIS n'y passe
    pas, et rien ne reprenait le relais.

    `fix_winding` propage une orientation de proche en proche depuis une face
    de depart. Il rend le maillage coherent, mais RIEN ne garantit qu'il
    choisisse le meme sens que celui d'origine : une fois sur deux il recoud
    tout a l'envers.

    On ne peut pas trancher au signe du volume — le maillage n'est pas
    etanche, et ce volume ne veut alors plus dire grand-chose ; `fix_inversion`
    s'y fie et retourne l'objet entier. Mesure : volume -0,014 apres coup.

    Le critere sur : comparer les normales AVANT et APRES. TRELLIS oriente
    correctement la majorite de ses faces ; si la couture a fait basculer
    cette majorite, on la remet dans son sens. Aucune hypothese sur la forme,
    aucune etancheite requise.

    Mesure sur un personnage de 283 k faces : 6,6 s.
    """
    try:
        import numpy as np
        import trimesh

        if geom.is_winding_consistent:
            return

        avant = np.array(geom.face_normals, dtype=np.float64, copy=True)
        trimesh.repair.fix_winding(geom)
        apres = np.asarray(geom.face_normals, dtype=np.float64)

        if avant.shape == apres.shape and len(avant):
            # aire en poids : une grande facette pese plus qu'un eclat
            aires = np.asarray(geom.area_faces, dtype=np.float64)
            accord = float((aires * np.einsum("ij,ij->i", avant, apres)).sum())
            if accord < 0:
                geom.invert()
    except Exception as exc:  # noqa: BLE001 - un maillage sale vaut mieux que rien
        log.warning("orientation des faces non recousue : %s", exc)


#: Les deux noms sous lesquels le descripteur du modele se presente.
#: Le depot d'origine ecrit `pipeline.json` ; la conversion fp8 publiee sur
#: Hugging Face ecrit `pipeline_fp8.json`, meme contenu, autre nom. Exiger le
#: premier faisait echouer une installation pourtant complete.
_NOMS_DESCRIPTEUR = ("pipeline.json", "pipeline_fp8.json")


def _descripteur(weights: Path):
    """Le descripteur present sous ce dossier de poids, ou None."""
    for nom in _NOMS_DESCRIPTEUR:
        chemin = weights / nom
        if chemin.exists():
            return chemin
    return None


def _write_local_pipeline_config(weights: Path) -> str:
    import json

    src = _descripteur(weights)
    if src is None:
        raise FileNotFoundError(
            "aucun descripteur de modele sous %s : attendu %s"
            % (weights, " ou ".join(_NOMS_DESCRIPTEUR)))
    dst = weights / _LOCAL_CONFIG_NAME
    data = json.loads(src.read_text(encoding="utf-8"))

    rembg = data.get("args", {}).get("rembg_model", {})
    if rembg.get("args", {}).get("model_name", "").startswith("briaai/"):
        rembg["args"]["model_name"] = _REMBG_REPLACEMENT

    # Checkpoint paths stay RELATIVE. The loaders prefix them with the weights
    # folder themselves - except the sparse-structure decoder, which is a
    # Hugging Face reference into the older TRELLIS repo and is resolved as-is.
    # Rewriting them to absolute paths gets them prefixed twice.
    payload = json.dumps(data, indent=4)
    if not dst.exists() or dst.read_text(encoding="utf-8") != payload:
        dst.write_text(payload, encoding="utf-8")
    return _LOCAL_CONFIG_NAME


@contextmanager
def _stage_previews(pipe, out_dir: Path, step, on_preview):
    """Emit a preview as each generation stage completes.

    Implemented by wrapping two pipeline methods for the duration of the call,
    rather than reimplementing `run()` here. `run()` branches across four
    pipeline types and upstream changes it regularly; a copy would rot, while
    these two entry points are the stable part.
    """
    if on_preview is None:
        yield
        return

    from . import previews

    originals = {}

    def wrap(name, handler):
        fn = getattr(pipe, name, None)
        if fn is None:
            return
        originals[name] = fn

        def wrapped(*args, **kwargs):
            result = fn(*args, **kwargs)
            try:
                handler(result)
            except Exception as exc:  # a preview must never break a generation
                log.warning("preview hook %s failed: %s", name, exc)
            return result

        setattr(pipe, name, wrapped)

    def after_structure(coords):
        grid = int(coords[:, 1:].max().item()) + 1 if coords.numel() else 64
        step("rough shape ready", 0.2)
        path = previews.voxel_preview(coords, grid, out_dir / "preview_structure.glb")
        if path:
            on_preview("structure", path)

    def after_shape(result):
        # The cascade variant returns (slat, resolution); the plain one a slat.
        slat = result[0] if isinstance(result, tuple) else result
        step("shape ready, texturing", 0.45)
        path = previews.shape_preview(pipe, slat, out_dir / "preview_shape.glb")
        if path:
            on_preview("shape", path)

    wrap("sample_sparse_structure", after_structure)
    wrap("sample_shape_slat", after_shape)
    wrap("sample_shape_slat_cascade", after_shape)
    # LE MULTI-VUE APPELLE D AUTRES METHODES, ET N ETAIT DONC PAS INSTRUMENTE.
    #
    # `run_multiview` passe par `sample_sparse_structure_multiview` et
    # `sample_shape_slat[_cascade]_multiview` — des noms voisins que ces
    # crochets ne connaissaient pas. Consequences, toutes constatees la nuit
    # du 28 aout 2026 sur une generation a quatre vues :
    #
    #   - aucun apercu de structure ni de forme n arrivait a l ecran ;
    #   - les deux jalons de progression qui vivent DANS ces crochets
    #     (« rough shape ready » a 20 %, « shape ready, texturing » a 45 %)
    #     ne se declenchaient jamais : la barre sautait de 26 % a 47 % ;
    #   - et surtout, le journal ne portait AUCUN chiffre — ni voxels, ni
    #     faces. Impossible de savoir ou le multi-vue passe son temps, ce qui
    #     est exactement la question qu on se posait.
    #
    # Le meme crochet marche tel quel : `after_shape` sait deja lire un
    # couple (slat, resolution) rendu par la variante en cascade.
    wrap("sample_sparse_structure_multiview", after_structure)
    wrap("sample_shape_slat_multiview", after_shape)
    wrap("sample_shape_slat_cascade_multiview", after_shape)
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(pipe, name, fn)


class JobCancelled(RuntimeError):
    """Raised inside the pipeline when the user cancels a running job."""


def _geometries(glb) -> list:
    """trimesh gives back either a Trimesh or a Scene depending on content."""
    if hasattr(glb, "geometry"):
        return list(glb.geometry.values())
    return [glb]
