"""Compatibility patches applied to the vendored TRELLIS.2 at load time.

Kept here, out of the vendored tree, so the fork can be re-pulled without
losing them - and so each patch carries the reason it exists.
"""
from __future__ import annotations

import logging

log = logging.getLogger("localmesh_engine.patches")

_applied = False


def apply() -> None:
    """Idempotent. Call after trellis2 is importable, before it runs."""
    global _applied
    if _applied:
        return
    _applied = True
    _patch_dinov3_layer_lookup()
    _patch_rembg_dtype()
    _sortir_nvdiffrast()


def _sortir_nvdiffrast() -> None:
    """Rasteriser l'atlas UV sans nvdiffrast, qui interdit l'usage commercial.

    La wheel `o_voxel` telechargee cuit l'atlas avec `nvdiffrast`, sous
    NVIDIA Source Code License : usage NON COMMERCIAL. LocalMesh est vendu.
    Le remplacant, `uvraster`, est ecrit dans ce projet et la substitution a
    ete verifiee plutot que supposee — 99,7 % des texels partages, erreur
    maximale 0,0014.

    POURQUOI UN REMPLACEMENT D'OBJET ET NON UN FICHIER RECOPIE. Le correctif
    vivait dans le `postprocess.py` de site-packages, edite a la main :
    invisible du depot, perdu a la premiere reinstallation, et illisible
    pour qui cherchait ce qui avait ete change. Ici la version corrigee est
    versionnee (`_vendu_o_voxel/postprocess.py`), et la wheel d'origine peut
    rester intacte sur le disque.

    Ne leve jamais : sans o_voxel il n'y a pas de generation du tout, et
    c'est `pipeline.py` qui doit le dire, pas un correctif.
    """
    try:
        import o_voxel
    except Exception as exc:                                  # noqa: BLE001
        log.warning("o_voxel introuvable, correctif non applique : %s", exc)
        return

    if getattr(o_voxel.postprocess, "_lumengen_sans_nvdiffrast", False):
        return

    from ._vendu_o_voxel import postprocess as sans_nvdiffrast

    o_voxel.postprocess.to_glb = sans_nvdiffrast.to_glb
    o_voxel.postprocess._lumengen_sans_nvdiffrast = True
    log.info("atlas UV rasterise par uvraster (nvdiffrast non commercial)")


def _patch_rembg_dtype() -> None:
    """Make background removal dtype-agnostic.

    TRELLIS' BiRefNet wrapper normalises the image to float32 and feeds it
    straight to the model. BiRefNet-HR ships fp16 weights, so the first real
    photo run dies with:

        RuntimeError: Input type (float) and bias type (struct c10::Half)
                      should be the same

    This never showed up in testing because the bundled example images are
    RGBA: TRELLIS skips background removal entirely when an alpha channel is
    already present, so the whole path stayed unexercised until a plain photo
    arrived. Casting the input to whatever dtype the model actually holds fixes
    it for fp16 and fp32 checkpoints alike.
    """
    import torch

    from . import config

    # rembg/__init__.py re-exports the class, so this name is the class itself.
    cls = config.trellis("pipelines.rembg").BiRefNet

    if getattr(cls, "_lumengen_dtype_patched", False):
        return

    original = cls.__call__

    def __call__(self, image):
        try:
            dtype = next(self.model.parameters()).dtype
        except StopIteration:  # pragma: no cover - a model with no parameters
            return original(self, image)

        if dtype == torch.float32:
            return original(self, image)

        image_size = image.size
        tensor = self.transform_image(image).unsqueeze(0).to("cuda", dtype=dtype)
        with torch.no_grad():
            preds = self.model(tensor)[-1].sigmoid().float().cpu()
        from torchvision import transforms

        mask = transforms.ToPILImage()(preds[0].squeeze()).resize(image_size)
        image.putalpha(mask)
        return image

    cls.__call__ = __call__
    cls._lumengen_dtype_patched = True
    log.info("patched BiRefNet background removal for fp16 checkpoints")


def _patch_dinov3_layer_lookup() -> None:
    """TRELLIS.2 reaches into DINOv3's transformer blocks as `model.layer`.

    transformers 5.x moved them behind an encoder wrapper (`model.model.layer`),
    so the stock code dies with `'DINOv3ViTModel' object has no attribute
    'layer'`. Pinning transformers back is worse: 5.x is what the rest of the
    stack is built against.

    We only re-point the lookup. The forward pass is reproduced exactly as
    upstream wrote it - embeddings, rope, blocks, then a *non-affine* layer
    norm. Using `last_hidden_state` instead would silently apply DINOv3's own
    learned final norm and shift every conditioning vector.
    """
    import torch
    import torch.nn.functional as F

    from . import config

    ife = config.trellis("modules.image_feature_extractor")

    def _blocks(model):
        for path in (("layer",), ("model", "layer"), ("layers",),
                     ("encoder", "layer"), ("encoder", "layers")):
            node = model
            for attr in path:
                node = getattr(node, attr, None)
                if node is None:
                    break
            if node is not None:
                return node
        raise AttributeError(
            f"cannot locate transformer blocks on {type(model).__name__}; "
            "DINOv3's layout changed again")

    def extract_features(self, image: "torch.Tensor") -> "torch.Tensor":
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for layer_module in _blocks(self.model):
            hidden_states = layer_module(
                hidden_states, position_embeddings=position_embeddings)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    ife.DinoV3FeatureExtractor.extract_features = extract_features
    log.info("patched DinoV3FeatureExtractor.extract_features for transformers 5.x")
