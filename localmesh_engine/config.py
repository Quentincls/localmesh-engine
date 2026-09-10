"""Runtime configuration and the environment setup TRELLIS.2 needs.

`bootstrap()` MUST run before anything imports trellis2 - the attention and
sparse-convolution backends are read from the environment at import time, and
picking them late silently leaves you on a backend that isn't installed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

def _runtime_root() -> Path:
    """Where the models, the library and the Python live.

    Two answers, in order. `LUMENGEN_ROOT` wins when set, because that is how
    an application tells the engine where it put things. Otherwise a portable
    copy carries its runtime beside itself and is found with no configuration
    at all - unzip anywhere, run.

    The order matters more than it looks. With a fallback tried first, a
    package missing its runtime does not say so: it quietly points at a path
    that exists on the machine that built it and nowhere else, and the failure
    surfaces much later, on someone else's computer, as "no weights found".

    THERE IS NO THIRD ANSWER, AND THAT IS THE POINT. This function used to end
    on a fixed drive letter from the author's own machine. It worked there and
    nowhere else, and it is exactly the kind of line that makes an engine
    unusable for everyone but its writer. Say what is missing instead.
    """
    # DEUX NOMS, ET C'EST VOULU. `LOCALMESH_ROOT` est celui du paquet ;
    # `LUMENGEN_ROOT` est celui que l'application historique pose depuis des
    # dizaines de versions installees. Retirer le second casserait toutes les
    # installations existantes pour une question de vocabulaire.
    override = os.environ.get("LOCALMESH_ROOT") or os.environ.get("LUMENGEN_ROOT")
    if override:
        return Path(override)
    # LE REPLI DOIT TENIR LA PROMESSE DU MESSAGE CI-DESSOUS, qui dit « a
    # `runtime/` folder next to the package ». Il ne remontait que d'une seule
    # profondeur, celle de l'arborescence privee (engine/localmesh_engine), et
    # designait donc un dossier situe DEUX crans au-dessus du paquet. Chez
    # quelqu'un qui clone le depot, il pointait hors de sa copie de travail :
    # aucune disposition de fichiers ne pouvait satisfaire le message.
    #
    # On essaie donc les deux : a cote du paquet d'abord, puis un cran plus
    # haut pour ne pas casser l'arborescence existante.
    ici = Path(__file__).resolve()
    for niveau in (1, 2):
        beside = ici.parents[niveau] / "runtime"
        if (beside / "engine").is_dir():
            return beside
    raise RuntimeError(
        "LocalMesh Engine cannot find its runtime. Set LOCALMESH_ROOT to the "
        "folder holding `models/`, or place a `runtime/` folder next to the "
        "package. Nothing is downloaded automatically: see the README.")


#: Everything heavy lives off the system drive; the app itself stays on C:.
RUNTIME_ROOT = _runtime_root()

ENGINE_ROOT = RUNTIME_ROOT / "engine"
MODELS_ROOT = RUNTIME_ROOT / "models"
#: LA BIBLIOTHEQUE N'APPARTIENT PAS AU MOTEUR.
#:
#: Le moteur ecrit ou on lui dit d'ecrire : `generate()` prend son dossier de
#: sortie en argument. Ce chemin ne reste ici que parce que l'application qui
#: entoure ce paquet range ses projets a cet endroit, et qu'elle le lit ici
#: depuis toujours. Aucun module du moteur ne s'en sert pour generer.
LIBRARY_ROOT = RUNTIME_ROOT / "library"
CACHE_ROOT = RUNTIME_ROOT / "cache"

#: QUELLE COPIE DU PAQUET trellis2 ON IMPORTE.
#:
#: Il y en avait DEUX, et seule la mauvaise tournait. Le dépôt en portait
#: une, versionnée, sous `_vendu_trellis2` — jamais importée par personne.
#: Le runtime en portait une autre, sous `engine/_wheels_src`, posée à la
#: main pendant l'essai et hors de tout dépôt : c'était celle-là qui
#: s'exécutait. Un correctif écrit dans la copie versionnée n'avait donc
#: aucun effet, et un correctif écrit dans celle qui tourne disparaissait
#: à la réinstallation suivante.
#:
#: La copie du dépôt fait foi. Elle est versionnée, elle voyage avec
#: `engine/lumengen` que tauri.conf.json empaquette déjà — donc le paquet
#: arrive avec l'application, sans passer par une archive de runtime.
#: Les deux autres restent en secours pour une machine d'essai qui aurait
#: encore l'ancienne disposition.
_TRELLIS_CANDIDATES = [
    Path(__file__).resolve().parent / "_vendu_trellis2",
    ENGINE_ROOT / "_wheels_src",
    ENGINE_ROOT / "trellis2_fork",
]


def _trellis_src() -> Path:
    for candidate in _TRELLIS_CANDIDATES:
        if (candidate / "trellis2" / "pipelines").is_dir():
            return candidate
    return _TRELLIS_CANDIDATES[-1]


TRELLIS_SRC = _trellis_src()
TRELLIS_WEIGHTS = MODELS_ROOT / "TRELLIS.2-4B"
#: Pixal3D ships as a separate 24 GB repository, downloaded on demand.
PIXAL3D_WEIGHTS = MODELS_ROOT / "Pixal3D"

#: TripoSplat: gaussian splats rather than meshes. A flat script folder rather
#: than a package, so it is imported off sys.path.
TRIPOSPLAT_SRC = ENGINE_ROOT / "triposplat"
TRIPOSPLAT_WEIGHTS = MODELS_ROOT / "TripoSplat"

_bootstrapped = False


def ensure_cuda_toolkit() -> bool:
    """Point gsplat at the CUDA toolkit, which it will not find on its own.

    gsplat compiles its kernels on first use and then refuses to load them
    unless it can still see a toolkit, so this is needed on every run and not
    only the first. It looks for `CUDA_HOME`, or for nvcc on PATH - and on
    Windows the installer sets neither for a non-interactive process, so the
    newest versioned folder is located and declared here. Returns whether one
    was found; nothing else in the engine depends on it.
    """
    # ninja lives beside the interpreter, and PyTorch shells out to it by name
    # when it loads a compiled extension - even one that is already built. The
    # app does not run with the virtual environment activated, so without this
    # gsplat reports "Ninja is required" on a machine where ninja is installed.
    scripts = Path(sys.executable).parent
    if str(scripts) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{scripts}{os.pathsep}{os.environ.get('PATH', '')}"

    # Compiled kernels go to one fixed place, built for this card only. Left to
    # itself PyTorch names the build directory after whatever architectures it
    # can see, so the same extension gets rebuilt under a different name in a
    # different process - which is how the application ended up compiling at
    # generation time, printing a spinner into a pipe Windows had given the ANSI
    # codepage, and reporting gsplat unavailable on a machine where it works.
    # Pinned like this it is built once, by `tools/warm_kernels.py`, and only
    # ever loaded afterwards.
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(CACHE_ROOT / "kernels"))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            import torch

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:                                     # pragma: no cover
            pass

    if os.environ.get("CUDA_HOME") and Path(os.environ["CUDA_HOME"]).is_dir():
        return True
    roots = [Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")]
    if os.environ.get("CUDA_PATH"):
        roots.insert(0, Path(os.environ["CUDA_PATH"]).parent)
    for root in roots:
        if not root.is_dir():
            continue
        versions = sorted((d for d in root.iterdir()
                           if (d / "bin" / "nvcc.exe").exists()
                           or (d / "bin" / "nvcc").exists()),
                          key=lambda d: d.name, reverse=True)
        if versions:
            home = versions[0]
            os.environ["CUDA_HOME"] = str(home)
            os.environ["CUDA_PATH"] = str(home)
            os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ['PATH']}"
            return True
    return False


def ensure_msvc() -> bool:
    """Make the C++ compiler visible, if Visual Studio Build Tools are here.

    PyTorch checks for `cl` before loading a compiled extension - even one that
    is already built and cached - so gsplat refuses to start from a plain
    process even though nothing needs compiling. Build Tools set their
    environment only inside their own developer prompt, so it is imported here:
    vcvars64 is run once and whatever it changed is copied in.

    Returns whether `cl` ended up on PATH. Nothing else in the engine needs it,
    and shipping a prebuilt wheel would remove the requirement altogether.
    """
    import shutil
    import subprocess

    if shutil.which("cl"):
        return True
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
             Path(os.environ.get("ProgramFiles", r"C:\Program Files"))]
    for root in roots:
        for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
            bat = (root / "Microsoft Visual Studio" / "2022" / edition / "VC"
                   / "Auxiliary" / "Build" / "vcvars64.bat")
            if not bat.exists():
                continue
            try:
                out = subprocess.run(f'"{bat}" >nul && set', shell=True,
                                     capture_output=True, text=True, timeout=120)
            except Exception:
                continue
            for line in out.stdout.splitlines():
                key, sep, value = line.partition("=")
                if sep and key.upper() in ("PATH", "INCLUDE", "LIB", "LIBPATH"):
                    os.environ[key.upper()] = value
            if shutil.which("cl"):
                return True
    return False


def bootstrap() -> None:
    """Prepare process environment. Idempotent, call before importing trellis2."""
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    # Dense attention: flash-attn has no Windows build, sdpa is PyTorch-native
    # and plenty fast on Blackwell with torch 2.8+cu128.
    os.environ.setdefault("ATTN_BACKEND", "sdpa")

    # Sparse attention is a different code path and only implements
    # xformers / flash_attn / flash_attn_3 - there is no sdpa fallback. Neither
    # works on Blackwell (see sdpa_backend for the details), so we declare
    # "xformers" and then shadow xformers.ops with our own SDPA implementation.
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    from . import sdpa_backend
    sdpa_backend.install()

    # flex_gemm is the default sparse conv backend and is one of our prebuilt
    # Blackwell wheels, so we keep it rather than pulling in spconv.
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

    # EXR is how the HDRI environment maps are loaded for the preview render.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    # Keeps peak allocation down on the long 1536 cascade. Windows' CUDA
    # allocator ignores expandable_segments and warns; harmless, so keep it for
    # the Linux/WSL case and silence nothing.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    os.environ.setdefault("HF_HOME", str(MODELS_ROOT / "hf"))

    # WINDOWS N A PAS LE DROIT DE CREER UN LIEN SYMBOLIQUE, ET C EST LA REGLE,
    # PAS L EXCEPTION.
    #
    # `huggingface_hub` range les octets dans `blobs/` et fabrique un LIEN vers
    # eux depuis `snapshots/`. Sur Windows, creer un lien demande le mode
    # developpeur ou les droits d administrateur — que personne n a. La
    # bibliotheque sait retomber sur une copie, mais pas toujours a temps : chez
    # un client, le 27 aout 2026, le telechargement de BiRefNet s est arrete net
    # sur
    #
    #     OSError: [WinError 1314] Le client ne dispose pas d un privilege
    #     necessaire : '..\..lobsaa0bca6...' -> '...\snapshots\...\handler.py'
    #
    # et le panneau des Reglages a affiche « TRELLIS.2 — echec » sur une
    # installation neuve. Il a fallu qu il relance le telechargement a la main.
    #
    # Le produit posait deja `..._WARNING` — il faisait donc TAIRE
    # l avertissement qui annoncait precisement cette panne, sans en traiter la
    # cause. On coupe les liens pour de bon : la bibliotheque copie, ce qui
    # coute un doublon sur disque et ne demande aucun privilege.
    #
    # A poser AVANT tout import de `huggingface_hub` : ses constantes lisent
    # l environnement au moment de l import, une fois pour toutes.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

    if str(TRELLIS_SRC) not in sys.path:
        sys.path.insert(0, str(TRELLIS_SRC))

    # The vendored trellis2 lives inside a ComfyUI node and imports two of its
    # modules; stand-ins keep it importable outside ComfyUI.
    from . import comfy_shim
    comfy_shim.install(MODELS_ROOT)

    # AVANT le premier import d o_voxel, et c est tout l interet : sa wheel
    # importe nvdiffrast au niveau module, or nvdiffrast interdit l usage
    # commercial et n a rien a faire dans un produit vendu. Le leurre laisse
    # l import passer ; patches.py remplace ensuite la seule fonction qui
    # s en servait.
    from . import leurre_nvdiffrast
    leurre_nvdiffrast.installer()

    # LE PAQUET VENDORISE IMPORTE `uvraster` NU, ET IL N EST PAS SUR LE CHEMIN.
    #
    # `trellis2_image_to_3d.py` et `trellis2_texturing.py` ecrivent
    # `import uvraster`, sans point. Dans ComfyUI ce module est un fichier pose
    # a cote d eux ; ici c est `localmesh_engine.uvraster`, et rien ne repond a ce nom.
    #
    # Le banc d essai ne le voyait pas : une copie du fichier trainait dans son
    # `site-packages`, posee a la main un jour ou l autre. Une installation de
    # client n en a pas, et TOUTE generation s arretait a
    # `ModuleNotFoundError: No module named uvraster` — quarante-cinq secondes
    # apres le clic, sur « loading models ». Constate le 27 aout 2026 sur une
    # vraie installation, a la premiere generation, et invisible partout
    # ailleurs.
    #
    # On enregistre donc le notre sous le nom que le paquet attend, plutot que
    # de retoucher deux fichiers vendorises qu une mise a jour du paquet
    # ecraserait sans bruit. Meme geste que le leurre ci-dessus, meme raison.
    # `setdefault` : si un vrai `uvraster` est sur le chemin, il gagne.
    from . import uvraster as _uvraster
    sys.modules.setdefault("uvraster", _uvraster)

    # trellis2 is reached only through `t2pkg` - see that module for why the
    # nesting is required and why aliasing it to a second name is not.
    from . import t2pkg  # noqa: F401

    # Les dossiers du runtime existent avant que quoi que ce soit y écrive.
    # Ces quatre lignes vivaient APRÈS le `return` de `trellis()` : jamais
    # exécutées. Elles ne manquaient à personne parce que l'installateur crée
    # ces dossiers — mais un runtime posé à la main, lui, n'a personne.
    for d in (MODELS_ROOT, LIBRARY_ROOT, CACHE_ROOT):
        d.mkdir(parents=True, exist_ok=True)


#: Import path of the vendored trellis2. Never import `trellis2` directly.
TRELLIS_PKG = f"{__name__.rsplit('.', 1)[0]}.t2pkg.trellis2"


def trellis(submodule: str = ""):
    """Import a trellis2 submodule through the one supported path."""
    import importlib

    bootstrap()
    name = f"{TRELLIS_PKG}.{submodule}" if submodule else TRELLIS_PKG
    return importlib.import_module(name)


#: ------------------------------------------------------------------------
#: LES PALIERS, ET TOUT CE QU'ILS DECIDENT.
#:
#: Avant, un palier ne choisissait qu'un `pipeline_type` ; les pas, l'echan-
#: tillonneur, l'atlas et le budget de triangles etaient des valeurs fixes
#: envoyees par l'interface, identiques aux quatre paliers. Choisir "Apercu"
#: ou "Detaille" ne changeait donc presque rien au calcul, et rien du tout au
#: nombre de triangles livres. Un palier qui ne decide pas n'est pas un
#: palier, c'est une etiquette.
#:
#: Ici un palier decide de TOUT ce qui le distingue, en un seul endroit. Les
#: valeurs viennent du banc du 26 aout 2026 (BANC.md, 41 essais, RTX 4060
#: Laptop 8 Go) ; les ecarts au banc sont notes ci-dessous.
#:
#: Ce que le banc a etabli, et qui gouverne ce tableau :
#:   - euler contre heun : heun fait DEUX evaluations par pas. A qualite
#:     egale sur les sujets mesures, c'est le temps double pour rien.
#:   - le nombre de triangles ne coute RIEN en temps (60 k contre 150 k :
#:     269 s contre 268). Il sert la lisibilite du maillage, pas la vitesse.
#:   - le guidage de TEXTURE a 1.0 ne demande qu'une evaluation ; au-dessus,
#:     deux. MESURE le 26 aout 2026 SUR LE CHEMIN DU PRODUIT (et non repris
#:     du banc, qui texture par pipeline.texture_mesh(), un chemin que
#:     LocalMesh n'execute pas) : a graine fixe, 1,0 contre 3,0 contre 5,0,
#:     sur deux sujets. Le temps est le meme a trois secondes pres — le
#:     guidage de texture ne coute donc rien ici, contrairement a ce qu'on
#:     attendait. La qualite tranche : a 1,0 la texture est propre mais
#:     delavee et le second oeil du personnage disparait ; a 5,0 la palette
#:     est celle de la photo, les deux yeux sont la, et le texte imprime sur
#:     la cassette redevient lisible.
#:   - la cascade 1024 prenait 1 019 s et 93 % d'une carte de 8 Go pour un
#:     resultat indistinct du palier en dessous : elle est reservee a `max`,
#:     refuse sous `vram_min_gb`.
#:
#: ECART ASSUME AVEC LE BANC : le banc decouplait la resolution de forme
#: (512) de celle de texture (1024) par deux noeuds separes. Le `run()` du
#: paquet vendorise les couple — "512" veut dire forme 512 ET texture 512,
#: "1024" les deux en 1024. Standard prend donc "1024" : la forme y gagne, et
#: la mesure du produit (114-144 s) reste sous celle du banc, qui payait en
#: plus une passe `texture_mesh()` que LocalMesh n'execute pas.
@dataclass(frozen=True)
class Recette:
    """Ce qu'un palier decide. Tout est ici, rien n'est ailleurs."""
    #: Type de pipeline TRELLIS.2 : "512" | "1024" | "1024_cascade" | "1536_cascade"
    pipeline_type: str
    steps_structure: int
    steps_shape: int
    steps_texture: int
    #: "euler" | "heun" | "rk4" | "rk5"
    sampler: str
    #: Cote de l'atlas UV cuit par o_voxel.
    texture_size: int
    #: Budget de triangles livre. Un `finish.target_faces` explicite prime.
    faces: int
    #: Force du guidage de la texture. 1.0 = une evaluation par pas.
    tex_guidance: float
    #: Sous cette memoire video, le palier n'est pas propose ni accepte.
    vram_min_gb: float

    #: DEUX RESOLUTIONS QUE LA CHAINE DE REMAILLAGE A BESOIN DE CONNAITRE, et
    #: qui sont DERIVEES tant que la table n'a pas ete rejugee. Ce sont des
    #: proprietes et non des champs pour une raison precise : les poser en
    #: champs obligerait a reecrire les cinq lignes de PALIERS, donc a trancher
    #: aujourd'hui l'arbitrage « Standard en 512 ou en 1024 » qui se tranche
    #: par la mesure, pas par l'argument. Elles deviendront des champs quand
    #: cette mesure existera — c'est ce qui permettra de decoupler la forme du
    #: contourage, ce que la regle ci-dessous ne sait pas faire.

    @property
    def dc_resolution(self) -> int:
        """Resolution du contourage dual, 3e argument positionnel de
        `cumesh.remeshing.reconstruct_mesh_dc_quad`.

        Regle du banc : la grille du contourage suit celle de la forme —
        512, 1024, ou 1536 pour la cascade 1536 (Extreme). Remailler a 1024
        une forme echantillonnee a 1536 jetterait exactement le detail que
        ce palier achete. Le garde-fou de `remaillage.nettoyer` peut ensuite
        redescendre selon la memoire de la carte.
        """
        if self.pipeline_type.startswith("1536"):
            return 1536
        return 512 if self.pipeline_type.startswith("512") else 1024

    @property
    def tex_resolution(self) -> int:
        """Resolution de la grille PBR decodee, ET choix du DiT de texture :
        512 -> tex_slat_flow_model_512, TOUTE autre valeur -> le 1024.

        INDEPENDANT du `pipeline_type` par nature — `texture_mesh` choisit son
        DiT sur ce nombre-la, pas sur le type de pipeline. C'est ce que `run()`
        seul interdisait, et c'est ce qui permet au banc de faire une forme a
        512 et une texture au DiT 1024.
        """
        return 512 if self.pipeline_type.startswith("512") else 1024


#: Perimetre maximal d'un trou bouche AVANT le remaillage. Sur un objet
#: inscrit dans [-0.5, 0.5], 1.0 veut dire « ferme tout » : c'est ce que le
#: contourage dual demande, puisqu'il decide du dedans et du dehors. C'est la
#: valeur du banc. CE QU'ELLE REFERME REELLEMENT N'A JAMAIS ETE OBSERVE — une
#: bouche, une anse, un tube sont des candidats.
BOUCHAGE_PERIMETRE = 1.0

#: JUSQU'OU LE SECOND BOUCHAGE VA, ET POURQUOI 4,0 NE SUFFISAIT PAS.
#:
#: `remaillage.nettoyer` repare le non-manifold puis referme les bords ouverts
#: — c'est l'etape qui avait ferme les yeux de la poupee. Son plafond etait
#: 4,0, mesure le 27 aout sur ce sujet-la, ou il rendait zero bord.
#:
#: LE SAMOURAI SORT AU-DESSUS. Mesure du 9 septembre 2026 : une seule boucle
#: de bord, 953 aretes, 8,43 de perimetre pour une diagonale de 1,38. Elle
#: court le long de l'ourlet du hakama et des lambeaux de cape — la ou
#: l'etoffe devient plus mince que la grille et sort en simple feuillet.
#:
#: ESSAI CHIFFRE, sur le maillage reel, apres soudure des sommets :
#:
#:     sans reparation, seuil 4 / 12 / 40   889 bords   (le seuil n'y peut rien)
#:     avec reparation, seuil  4            953 bords
#:     avec reparation, seuil 12              0 bord
#:
#: Et le bouchage NE SOUDE PAS les lambeaux : silhouette identique avant et
#: apres, verifiee a l'image. Ce que ca change est invisible dans notre
#: visionneuse — elle force la double face — et tres visible chez le client,
#: qui ouvre le fichier dans un moteur qui, lui, ecarte les faces arriere.
FERMETURE_PERIMETRE = 12.0

#: CE QUE LE FILTRE A DEBRIS REGARDE, ET POURQUOI IL LUI FAUT DEUX CRITERES.
#:
#: L'ancien seuil etait une AIRE ABSOLUE de 1e-5, reprise de `o_voxel.to_glb`,
#: et le commentaire qui l'accompagnait disait deja « a calibrer sur un sujet
#: a debris ». LE SUJET EST ARRIVE le 9 septembre 2026 : le samourai de
#: Quentin, quatre vues, palier Standard.
#:
#: MESURE SUR L'OBJET LIVRE. 1 133 morceaux distincts, dont 1 102 de moins de
#: 50 faces. L'aire du plus gros morceau vaut 3,16 : le seuil de 1e-5 est donc
#: trois cent mille fois plus petit que lui, et bien plus petit que le moindre
#: eclat. IL NE JETAIT RIEN. Les eclats se voyaient a l'oeil nu par
#: l'ouverture de la cape, et Quentin les a vus avant nous.
#:
#: ETRE PETIT NE SUFFIT PAS, et la lecon est deja ecrite dans `meshops` : un
#: maillage reconstruit depuis des voxels est parseme de petites plaques
#: POSEES SUR la coque principale. Visuellement, elles SONT la surface ; les
#: jeter y ouvre des trous — c'est exactement ce qui avait transforme un oeil
#: genere en confettis. Un morceau n'est donc du debris que s'il est A LA FOIS
#: petit ET decolle du corps.
#:
#: CE QUE LES DEUX CRITERES DONNENT SUR CE SUJET : 185 morceaux vraiment
#: decolles, 1,21 % de l'aire. Les 941 autres touchent la coque et restent.
#: Le plafond ne mord donc pas ici ; il est la pour le sujet en filigrane,
#: fait de dizaines de coquilles legitimes, ou tout le filtre doit se taire.

#: Petitesse, en fraction de l'aire du PLUS GROS morceau — jamais une aire
#: absolue, qui ne veut rien dire d'un sujet a l'autre.
DEBRIS_RATIO_AIRE = 0.001
#: Decollement, en fraction de la plus grande dimension de l'objet.
DEBRIS_DISTANCE_MINI = 0.004
#: Part de l'aire que le filtre s'autorise a retirer. Au-dela il se tait et
#: le dit : le sujet est fait de petites pieces, pas de debris.
DEBRIS_PART_MAXI = 0.05


PALIERS: dict[str, Recette] = {
    # Iteration : on juge la forme, pas la matiere. L atlas reste a 2048 :
    # descendre a 1024 ne gagne que 8 s sur 106 (mesure du 26 aout 2026,
    # bake+finition 29,6 s contre 19,2), et un apercu qui ment sur la couleur
    # est un moins bon apercu. Pour reference, le meme poste coute 73,8 s a
    # 4096 — c est la moitie de l ecart entre Standard et Detaille.
    "draft": Recette(
        pipeline_type="512", steps_structure=8, steps_shape=8, steps_texture=8,
        sampler="euler", texture_size=2048, faces=80_000,
        tex_guidance=1.0, vram_min_gb=0.0),
    # Le defaut. Meilleur rapport qualite / temps mesure au banc.
    # L ATLAS PASSE A 4096, ET C EST LA MESURE QUI L IMPOSE.
    #
    # Le nombre qui compte n est ni le cote de l atlas ni le nombre de faces,
    # c est leur rapport : combien de texels chaque triangle recoit. En
    # dessous d une dizaine de pixels de cote, le rembourrage pose autour de
    # chaque ile UV mord sur la voisine, et chaque triangle va chercher sa
    # couleur chez son voisin dans l atlas. C est la mosaique.
    #
    # Mesure du 27 aout 2026, meme sujet et meme graine que la reference du
    # banc S1-ref-4096-12 :
    #
    #   banc     114 895 faces  atlas 4096  ->  146 texels par triangle
    #   produit  142 620 faces  atlas 2048  ->   29 texels par triangle
    #
    # A 4096 sur le meme maillage : 118. On rentre dans la plage du banc.
    #
    # La forme reste en 1024 : mesuree a 259 s et 2,81 Go sur cette carte de
    # 8 Go, elle tient largement, et une forme echantillonnee a 1024 est plus
    # fine qu a 512. La specification proposait de descendre a 512 pour coller
    # au banc — la mesure dit que ce n est pas necessaire.
    "standard": Recette(
        pipeline_type="1024", steps_structure=12, steps_shape=12, steps_texture=12,
        sampler="euler", texture_size=4096, faces=150_000,
        tex_guidance=5.0, vram_min_gb=0.0),
    # Plus de passes, plus de triangles, un atlas deux fois plus fin.
    #
    # LE 8K A ETE ESSAYE ET ECARTE (2 septembre 2026). Masque oni, meme
    # graine, meme carte de 8 Go :
    #
    #    atlas 4096 : 556 s, pic 10,3 Go,  56 Mo,  68 texels par triangle
    #    atlas 8192 : 808 s, pic 11,4 Go, 162 Mo, 271 texels par triangle
    #
    # Quatre fois plus de texels, et Quentin, les deux sous les yeux : « la
    # texture 8K ne sert a rien du tout ». Le modele genere sa matiere a une
    # resolution interne fixe ; l atlas 8192 n est qu une toile plus grande
    # pour le meme dessin, payee +45 % de temps, un fichier trois fois plus
    # gros et quatre minutes de carte pleine a la cuisson (244 s contre 55).
    # Ce que Detaille achete vraiment, c est la FORME : 250 000 triangles et
    # 16 passes. Ne pas re-essayer le 8K sans une mesure de detail a l image.
    "high": Recette(
        pipeline_type="1024", steps_structure=16, steps_shape=16, steps_texture=16,
        sampler="euler", texture_size=4096, faces=250_000,
        tex_guidance=5.0, vram_min_gb=0.0),
    # EXTREME = LA GRILLE 1536. Decision de Quentin, 2 septembre 2026.
    #
    # L ancienne recette etait la cascade 512 -> 1024 : elle finissait sur la
    # MEME grille que Detaille, et le banc du 26 aout l a trouvee
    # « indiscernable de Detaille » — normal, elle achetait de la robustesse,
    # pas du detail. Le seul levier qui ajoute du vrai detail geometrique
    # au-dela de Detaille est la finesse de la grille de forme : 1536, soit
    # 1,5 fois plus fin par axe et 3,4 fois plus de cellules. C est le reglage
    # « qualite maximale » de TRELLIS.2 lui-meme, et il tourne avec les poids
    # deja installes (les modeles 512 et 1024 appliques sur une grille 1536).
    #
    # 400 000 faces, parce que la grille 1536 en produit 2,25 fois plus que
    # la 1024 et qu on veut en garder. Atlas 8192 NON PAS pour la nettete
    # (essaye et ecarte a l oeil le 2 septembre sur Detaille) mais pour le
    # rangement : 400 000 faces sur 4096 feraient 42 texels par triangle,
    # sous le seuil de mosaique (~50) ; sur 8192, 168.
    #
    # 24 Go MINIMUM. La memoire suit les cellules (r = +0,92 au banc) : le
    # masque oni fait 8,3 M de cellules a 1024 pour un pic reel de 10 a 12 Go ;
    # a 1536 il en fera ~18,7 M, soit 22 a 27 Go. Une carte de 16 Go
    # deborderait de 8 Go dans la RAM — le « parfois ca plante » qu on vient
    # d eliminer. 16 passes et non 24 : les passes n achetent presque rien
    # (mesure sur Detaille), la grille achete tout.
    #
    # NON MESURE SUR SA CIBLE : cette machine a 8 Go. Les chiffres ci-dessus
    # sont extrapoles de la loi des cellules, pas observes. A rejouer sur la
    # 5090 des son retour, AVANT d en parler dans une note de version.
    "max": Recette(
        pipeline_type="1536_cascade", steps_structure=16, steps_shape=16,
        steps_texture=16, sampler="euler", texture_size=8192, faces=400_000,
        tex_guidance=5.0, vram_min_gb=24.0),
    #: Pixal3D : un autre modele, qui reprojette les pixels de la photo au
    #: lieu de regenerer la surface. Ses poids (24 Go) ne sont telecharges par
    #: aucun chemin du produit — ce palier n'est atteignable depuis aucune
    #: interface et n'est garde que pour ne pas perdre le branchement.
    "extreme": Recette(
        pipeline_type="1536_cascade", steps_structure=24, steps_shape=24,
        steps_texture=24, sampler="euler", texture_size=4096, faces=500_000,
        tex_guidance=5.0, vram_min_gb=24.0),
}

#: Le palier servi quand celui demande ne tient pas dans la carte.
PALIER_DEFAUT = "standard"


def recette(detail: str) -> Recette:
    return PALIERS.get(detail) or PALIERS[PALIER_DEFAUT]


def palier_tenable(detail: str, vram_gb: float | None = None) -> bool:
    """La carte peut-elle servir ce palier ? Le meme test des deux cotes."""
    if vram_gb is None:
        vram_gb = gpu_report().total_vram_gb
    return vram_gb + 0.5 >= recette(detail).vram_min_gb


#: Vues derivees, pour les appelants qui ne veulent qu'un champ. La table
#: ci-dessus reste la seule source.
DETAIL_PRESETS: dict[str, str] = {k: v.pipeline_type for k, v in PALIERS.items()}
FACES_PRESETS: dict[str, int] = {k: v.faces for k, v in PALIERS.items()}

#: Presets servis par Pixal3D plutot que TRELLIS.2.
PIXAL3D_PRESETS = {"extreme"}

@dataclass(frozen=True)
class GpuReport:
    name: str
    total_vram_gb: float
    compute_capability: tuple[int, int]
    torch_version: str
    cuda_version: str | None

    @property
    def is_blackwell(self) -> bool:
        return self.compute_capability[0] >= 12

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "total_vram_gb": round(self.total_vram_gb, 1),
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "torch": self.torch_version,
            "cuda": self.cuda_version,
            "blackwell": self.is_blackwell,
        }


def gpu_report() -> GpuReport:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            # Ancien nom du produit, et exigence fausse : LocalMesh tourne
            # sur 8 Go, mesure. Seul le palier Extreme demande 12 Go, et il se
            # refuse tout seul. Un message d erreur qui reclame une carte que
            # le client a deja envoie chercher un probleme qui n existe pas.
            "No CUDA GPU visible. LocalMesh needs an NVIDIA graphics card. "
            "8 GB of video memory is enough for every level except Extreme, "
            "which asks for 12."
        )
    props = torch.cuda.get_device_properties(0)
    return GpuReport(
        name=props.name,
        total_vram_gb=props.total_memory / 1024 ** 3,
        compute_capability=torch.cuda.get_device_capability(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )
