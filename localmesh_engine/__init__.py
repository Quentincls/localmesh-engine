"""LocalMesh Engine — une photo, ou quatre, et un objet 3D.

Ce paquet est le cœur de génération de LocalMesh, et rien d'autre : il prend
des photos détourées, il rend un fichier glTF texturé, avec sa matière et sa
finition. Il ne sait rien de la bibliothèque, du carnet, des projets, des
nuages de points ni de l'atelier d'images — ces choses appartiennent à
l'application qui l'utilise.

**La frontière n'a pas été choisie à la main.** Elle a été mesurée : la
fermeture transitive des imports relatifs de `pipeline` compte vingt et un
noms au premier niveau du paquet (mesure du 11 octobre 2026) et n'atteint
aucun module du produit. C'est cette fermeture, exactement, qui vit ici.

Ce qu'il contient
-----------------

- `pipeline` : la chaîne complète, du détourage à l'export.
- `config` : les recettes par palier, les chemins du runtime, la lecture de la
  carte graphique et les budgets qui en découlent.
- `matting`, `remaillage`, `normalbake`, `materials`, `meshops`, `uvraster` :
  détourage, nettoyage du maillage, cuisson des cartes, matériau, atlas.
- `projection` : les pixels des photos reposés sur la texture dessinée par
  le modèle, après la texturation (1.2.0).
- `_vendu_trellis2` : le modèle de Microsoft, sous sa licence MIT d'origine,
  avec nos correctifs signalés dans le code.
- `owned_tiled_mesh` : un propriétaire par quad aux frontières des tuiles,
  correctif d'un défaut du contourage livré en amont.

Ce qu'il attend de son hôte
---------------------------

Les poids, posés sous `config.MODELS_ROOT`. Le paquet ne télécharge rien : ce
qui décide d'aller chercher un modèle, et avec quelle barre de progression,
est une décision de produit.

Licence
-------

Apache 2.0, sauf le code recopié d'ailleurs, qui garde la sienne : le MIT de
Microsoft Research, de Tencent et de visualbruno pour `_vendu_trellis2` et
`_vendu_o_voxel`, le MIT de Microsoft Research pour `_vendu_moge` (MoGe) et
l'Apache 2.0 de Meta pour son sous-arbre `model/dinov2` (DINOv2), l'Apache
de valeo.ai pour `multivue/champ`. Un seul fichier
n'était pas libre, `multivue/champ/layers/rope.py`, sous l'accord DINOv3 de
Meta : il a été réécrit et il est sous Apache 2.0 comme le reste. Les poids,
eux, gardent chacun leur licence et ne sont pas redistribués. Voir NOTICE.
"""

#: LA VERSION DU MOTEUR, ET ELLE COMPTE.
#: Le contrat avec l'application est une version epinglee : une recette qui
#: change est un changement de version, pas un detail. Elle se lit par
#: `localmesh_engine.__version__` (c'est ce que demande le formulaire de
#: bogue), et c'est elle qu'un integrateur epingle. Le champ `version` de
#: `/health`, dans l'application, est celle de l'application, pas celle-ci.
#: `pyproject.toml` et `CITATION.cff` portent chacun la leur, ECRITE A LA
#: MAIN : la monter ici sans eux les laisse en arriere (c'est arrive pendant
#: toute la 1.1.0).
__version__ = "1.2.0"

__all__ = ["Engine", "GenerateResult", "GenerateSettings", "JobCancelled",
           "config", "__version__"]


def __getattr__(nom):
    """Ne rien monter tant qu'on n'a rien demande.

    CE PAQUET S'IMPORTAIT EN EXIGEANT SON RUNTIME. `from . import config` en
    tete de fichier resolvait la racine des poids des l'import, donc
    `import localmesh_engine` — et jusqu'a `python -m localmesh_engine --help`
    — mourait sur une machine qui n'avait pas encore pose les modeles.
    Quelqu'un qui decouvre le depot ne pouvait pas lire l'aide avant d'avoir
    telecharge plusieurs gigaoctets, ce qui est exactement l'inverse de
    l'ordre dans lequel on essaie un logiciel.

    Les noms publics restent les memes : `from localmesh_engine import Engine`
    passe par ici et monte ce qu'il faut, au moment ou on le nomme.
    """
    # `import_module` ET PAS `from . import`. La seconde forme redemande
    # l'attribut au paquet pour savoir s'il existe deja, donc elle rappelle
    # cette fonction, qui la rappelle : recursion infinie des le premier
    # acces. L'importateur direct monte le sous-module sans repasser par la.
    import importlib

    if nom == "config":
        return importlib.import_module(".config", __name__)
    if nom in ("Engine", "GenerateResult", "GenerateSettings", "JobCancelled"):
        return getattr(importlib.import_module(".pipeline", __name__), nom)
    raise AttributeError("module %r n'a pas d'attribut %r" % (__name__, nom))
