from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class DinoLockMixin:
    """
    Shared DINO-lock functionality for any GuidanceInterval sampler.

    When ``dino_lock > 0`` each step computes both the CFG-guided velocity
    and the pure positive-conditioned (DINO-only) velocity, then blends
    toward the DINO direction.  The schedule builds the initial shape from
    DINOv3 features first, then hands off to CFG for detail:

        +-----------+---------------------------+
        | Steps     | Lock strength             |
        +-----------+---------------------------+
        | 0 – 40 %  | 0.92 (full DINO: shape)   |
        | 40 – 70 % | ramp 0.92 → dino_lock     |
        | 70 – 100 %| dino_lock (CFG guardrail)  |
        +-----------+---------------------------+

    The mixin intercepts ``sample()`` to add the ``dino_lock`` and
    ``dino_substeps`` keyword arguments.  When ``dino_lock <= 0`` it
    falls straight through to the underlying sampler's ``sample()``.
    """

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_pos_only_v(self, model, x_t, t, cond, **kwargs):
        """Run model with guidance_strength=1 (positive cond only)."""
        pos_kw = dict(kwargs)
        pos_kw["guidance_strength"] = 1.0
        return self._inference_model(model, x_t, t, cond, **pos_kw)

    def _dino_project(self, guided_v, pos_v, lock_strength):
        """
        Linear velocity blend toward the DINO-only signal.

        lock_strength=0 → guided_v unchanged
        lock_strength=1 → pure pos_v (full DINO trajectory)
        """
        return (1.0 - lock_strength) * guided_v + lock_strength * pos_v

    @staticmethod
    def _alignment_stats(guided_v, pos_v):
        """Alignment statistics between guided and positive-only velocity."""
        g_raw = guided_v.feats if hasattr(guided_v, 'feats') else guided_v
        p_raw = pos_v.feats if hasattr(pos_v, 'feats') else pos_v

        g_flat = g_raw.reshape(-1).float().unsqueeze(0)
        p_flat = p_raw.reshape(-1).float().unsqueeze(0)

        g_norm = g_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        p_norm = p_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

        cos = (g_flat * p_flat).sum(dim=1) / (g_norm.squeeze() * p_norm.squeeze())
        cos_mean = cos.mean().item()
        angle = np.degrees(np.arccos(np.clip(cos_mean, -1.0, 1.0)))
        mag_ratio = (p_norm.squeeze() / g_norm.squeeze()).mean().item()
        drift = (g_flat - p_flat).norm(dim=1).mean().item() / g_norm.squeeze().mean().item()

        proj_c = (g_flat * p_flat).sum(dim=1, keepdim=True) / (p_flat * p_flat).sum(dim=1, keepdim=True).clamp(min=1e-8)
        perp = g_flat - proj_c * p_flat
        perp_ratio = perp.norm(dim=1).mean().item() / g_norm.squeeze().mean().item()

        return {"cos_sim": cos_mean, "mag_ratio": mag_ratio,
                "angle_deg": angle, "drift": drift, "perp_ratio": perp_ratio}

    def _dino_lock_step(self, model, x_t, t, t_prev, cond,
                         lock_strength, step_idx, total_steps,
                         substeps=1, v_ema=None, ema_alpha=0.85, verbose=True, **kwargs):
        """
        One step with DINO lock + velocity EMA smoothing.

        v_ema smoothing reduces velocity discontinuities at phase
        transitions: v_final = α·v_current + (1-α)·v_ema_prev.
        """
        guided_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pos_v = self._get_pos_only_v(model, x_t, t, cond, **kwargs)

        stats = self._alignment_stats(guided_v, pos_v)
        
        if verbose:
            phase = "FOUND" if lock_strength >= 0.9 else ("RAMP" if step_idx >= int(total_steps * 0.4) and step_idx < int(total_steps * 0.7) else "GUARD")
            sub_tag = f" x{substeps}" if substeps > 1 else ""
            ema_tag = " +ema" if v_ema is not None else ""
            print(f"  [DinoLock {step_idx+1:>3}/{total_steps}] "
                  f"cos={stats['cos_sim']:+.4f}  "
                  f"angle={stats['angle_deg']:5.1f}°  "
                  f"perp={stats['perp_ratio']:.3f}  "
                  f"drift={stats['drift']:.4f}  "
                  f"lock={lock_strength:.3f} ({phase}{sub_tag}{ema_tag})")

        if lock_strength <= 0.0:
            pred_v = guided_v
        else:
            pred_v = self._dino_project(guided_v, pos_v, lock_strength)

        if v_ema is not None:
            pred_v = ema_alpha * pred_v + (1.0 - ema_alpha) * v_ema
        new_v_ema = pred_v

        if substeps > 1 and lock_strength >= 0.9:
            dt_total = t - t_prev
            dt_sub = dt_total / substeps
            current = x_t
            t_cur = t
            for _s in range(substeps):
                v_sub = self._inference_model(model, current, t_cur, cond, **kwargs)
                current = current - dt_sub * v_sub
                t_cur = t_cur - dt_sub
            pred_x_prev = current
        else:
            pred_x_prev = x_t - (t - t_prev) * pred_v

        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0,
                       "stats": stats, "lock_strength": lock_strength,
                       "v_ema": new_v_ema})

    @staticmethod
    def _compute_lock_strength(step_idx, total_steps, base_strength, foundation_cap = 0.92):
        """
        DINO Foundation schedule:
        - Steps  0 – 40 %:  0.92  (near-full DINO, 8 % CFG on-distribution).
        - Steps 40 – 70 %:  cosine ramp 0.92 → base_strength.
        - Steps 70 – 100 %: base_strength (residual guardrail).
        """
        #FOUNDATION_CAP = 0.92
        foundation_end = int(total_steps * 0.4)
        ramp_end = int(total_steps * 0.7)
        if step_idx < foundation_end:
            return foundation_cap
        if step_idx >= ramp_end:
            return base_strength
        progress = (step_idx - foundation_end) / max(1, ramp_end - foundation_end)
        blend = 0.5 * (1.0 - np.cos(np.pi * progress))
        return float(foundation_cap + (base_strength - foundation_cap) * blend)

    # ------------------------------------------------------------------ #
    #  sample() override                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond=None,
        neg_cond=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,        
        **kwargs
    ):
        # Strip keys that must not reach the model
        kwargs.pop("rk4_cond_lock_strength", None)
        kwargs.pop("rk4_cond_lock_end_strength", None)
        kwargs.pop("debug_dino_alignment", None)
        kwargs.pop("debug_dino_interval", None)

        # Multiview path: cond/neg_cond not provided (per-view conds are in kwargs)
        if cond is None:
            return super().sample(model, noise,
                                  steps=steps, rescale_t=rescale_t, verbose=verbose,
                                  tqdm_desc=tqdm_desc,
                                  guidance_strength=guidance_strength,
                                  guidance_interval=guidance_interval,
                                  **kwargs)

        if dino_lock <= 0.0:
            return super().sample(model, noise, cond, 
                                  steps = steps, 
                                  rescale_t = rescale_t, 
                                  verbose = verbose,
                                  tqdm_desc=tqdm_desc,
                                  neg_cond=neg_cond,
                                  guidance_strength=guidance_strength,
                                  guidance_interval=guidance_interval,
                                  **kwargs)

        # ----- DINO-locked sampling loop -----
        if verbose:
            print(f"\n{'='*72}")
            print(f"  DINO Foundation  |  guardrail={dino_lock:.2f}  |  steps={steps}")
            print(f"  Schedule: 0-40% DINO@0.92 (shape), 40-70% ramp→{dino_lock:.2f}, 70-100% guardrail")
            print(f"  Mode: foundation-first + velocity EMA smoothing")
            if dino_substeps > 1:
                print(f"  Substeps: {dino_substeps}x during foundation phase")
            print(f"{'='*72}")

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

        merged_kwargs = dict(kwargs)
        merged_kwargs["neg_cond"] = neg_cond
        merged_kwargs["guidance_strength"] = guidance_strength
        merged_kwargs["guidance_interval"] = guidance_interval

        all_stats = []
        v_ema = None
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc=tqdm_desc)):
            s = self._compute_lock_strength(i, steps, dino_lock, dino_foundation_cap)
            n_sub = dino_substeps if (s >= 0.9 and dino_substeps > 1) else 1
            out = self._dino_lock_step(model, sample, t, t_prev, cond,
                                        lock_strength=s,
                                        step_idx=i, total_steps=steps,
                                        substeps=n_sub,
                                        v_ema=v_ema,
                                        verbose=verbose,
                                        **merged_kwargs)
            sample = out.pred_x_prev
            v_ema = out.v_ema
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            all_stats.append(out.stats)

        avg_cos = np.mean([s["cos_sim"] for s in all_stats])
        avg_drift = np.mean([s["drift"] for s in all_stats])
        avg_perp = np.mean([s["perp_ratio"] for s in all_stats])
        final_cos = all_stats[-1]["cos_sim"]
        final_angle = all_stats[-1]["angle_deg"]
        final_perp = all_stats[-1]["perp_ratio"]
        
        if verbose:
            print(f"\n{'─'*72}")
            print(f"  DINO Foundation Summary")
            print(f"  avg cos_sim={avg_cos:+.4f}  avg drift={avg_drift:.4f}  avg perp={avg_perp:.4f}")
            print(f"  final cos_sim={final_cos:+.4f}  final angle={final_angle:.1f}°  final perp={final_perp:.4f}")
            print(f"{'─'*72}\n")

        ret.samples = sample
        return ret


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """Euler sampling with CFG, guidance interval, and optional DINO lock."""
    pass


# LE REPÈRE DE LA GRILLE TRELLIS.2, ET IL N'EST PAS CELUI QUE CE CODE CROYAIT.
#
# Ce mélangeur vient du nœud ComfyUI (TRELLIS.2 lui-même n'a AUCUNE voie
# multi-vue : `run()` prend une image, point). Il supposait la convention
# d'un tenseur d'images (profondeur, hauteur, largeur) : colonne 1 = avant
# arrière, colonne 3 = gauche droite. Or les coordonnées creuses de
# TRELLIS.2 sont (x, y, z) avec Z VERS LE HAUT, et `remaillage.vers_trimesh`
# les pose en (x, z, -y) pour sortir un maillage Y-up dont la face regarde
# +Z, c'est-à-dire -y dans la grille. Mesuré le 4 septembre 2026 sur le
# buste du banc (vignettes du moteur, extents des colonnes : la colonne 3
# occupe toute la grille, c'est la hauteur d'un buste) :
#
#     colonne 1 (x) : gauche <-> droite, « droite » du produit vers +x
#                     (la caméra déplacée à droite de l'image de face)
#     colonne 2 (y) : avant <-> arrière, la face vers -y, le dos vers +y
#     colonne 3 (z) : le haut
#
# Le code d'origine mélangeait donc avant/arrière le long de la LARGEUR et
# gauche/droite le long de la HAUTEUR : la vue droite peignait le sommet
# du crâne, la vue gauche le socle, et face/dos se partageaient les deux
# moitiés gauche et droite. Chaque voxel recevait un mélange de vues qui
# ne le regardaient pas ; d'où des structures denses et bruitées (5,1 M de
# faces de forme contre 1,45 M en mono-vue au même palier) et un dos qui
# ne devait rien à la photo du dos. `front_axis` est conservé pour la
# signature ; la seule convention qui existe est celle-ci.
_AXE_LARGEUR, _AXE_PROFONDEUR, _AXE_HAUTEUR = 1, 2, 3


def _scores_des_vues(x_lr, y_fb, views):
    """Un score par vue et par position : +1 au pôle qui regarde la vue,
    -1 au pôle opposé, 0 sur l'équateur. x_lr et y_fb sont dans [-1, 1]."""
    tables = {
        'front': -y_fb,
        'back': y_fb,
        'right': x_lr,
        'left': -x_lr,
    }
    return [tables[v] if v in tables else torch.full_like(x_lr, -10.0)
            for v in views]


class FlowEulerMultiViewSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with multi-view blending.
    """
    def __init__(self, sigma_min: float, resolution: int):
        super().__init__(sigma_min)
        self.resolution = resolution
    
    def _compute_view_weights_sparse(self, coords, views, front_axis='z', blend_temperature=2.0, grille=None) -> torch.Tensor:
        """
        Compute blending weights for sparse voxels.
        """
        # Normalize coords to [-1, 1] range (roughly)
        # L ECHELLE VIENT DE LA GRILLE DES COORDONNEES, PAS DU DIT.
        # `self.resolution` est celle du modele de flux : 32 pour le DiT
        # 512, 64 pour le 1024. Les coordonnees, elles, sortent TOUJOURS
        # de la structure creuse en 32, quel que soit le palier. Diviser
        # par 64 des positions qui vont jusqu a 31 les envoyait toutes
        # dans le negatif : `dos` (note -z) et `gauche` (note -x)
        # gagnaient le vote PARTOUT, y compris sur la face. Mesure au
        # banc, palier Standard : « coords z[3..28] x[0..31] /
        # diviseur 64 ». Au palier draft le diviseur valait deja 32 et
        # tout allait bien — c est pourquoi le defaut ne se voyait qu au
        # dessus de Standard.
        #
        # On deduit donc la grille des coordonnees elles-memes, arrondie
        # a la puissance de deux superieure. Elle rend 32 la ou le code
        # d origine rendait deja 32 : les paliers qui marchaient ne
        # bougent pas d un bit.
        # LA GRILLE VIENT DE L'APPELANT quand il la connaît (32 pour la
        # structure creuse, `hr_resolution // 16` pour l'étage haut de la
        # cascade : 96 à 1536, 64 à 1024, moins quand les jetons débordent).
        # À défaut, on la déduit de la plus grande coordonnée, arrondie au
        # multiple de 8 supérieur : TRELLIS.2 fait remplir la boîte sur
        # l'axe le plus long, donc max + 1 vaut la grille à un arrondi près
        # (relecture du 4 septembre 2026 : « puissance de deux supérieure »
        # rendait 128 pour 96, et décalait l'équateur d'un quart de grille).
        grille = grille or getattr(self, "_grille", None)
        if not grille:
            etendue = int(coords[:, 1:].max().item()) + 1
            grille = max(8, (etendue + 7) // 8 * 8)
        # Le centre d'une cellule, pas son coin : sur une grille de 32, la
        # cellule 0 vaut -0,97 et la cellule 31 +0,97, symétriques.
        x_lr = ((coords[:, _AXE_LARGEUR].float() + 0.5) / grille) * 2 - 1.0
        y_fb = ((coords[:, _AXE_PROFONDEUR].float() + 0.5) / grille) * 2 - 1.0
        scores = torch.stack(_scores_des_vues(x_lr, y_fb, views), dim=1) # (N, num_views)
        weights = torch.softmax(scores * blend_temperature, dim=1)
        return weights

    def _compute_view_weights_dense(self, shape, device, views, front_axis='z', blend_temperature=2.0) -> torch.Tensor:
        """
        Compute blending weights for dense grid (B, C, D, H, W).
        Returns weights of shape (1, 1, D, H, W, NumViews) for easy broadcasting (actually we want (1, 1, D, H, W) per view)
        """
        # shape is (B, C, D, H, W)
        D, H, W = shape[2], shape[3], shape[4]
        
        # Create meshgrid in [-1, 1]
        # We assume D is Z axis, W is X axis (usually D, H, W = Z, Y, X in 3D tensors?)
        # Let's verify standard: (Batch, Channel, Depth, Height, Width) -> (B, C, Z, Y, X)
        
        # Centres de cellule, comme pour la structure creuse.
        centres = lambda n: (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n * 2 - 1
        dz, dy, dx = centres(D), centres(H), centres(W)
        
        # Les trois dimensions spatiales sont (x, y, z) dans l'ordre des
        # coordonnées creuses : première = largeur, deuxième = profondeur,
        # troisième = hauteur (voir le repère au-dessus de la classe).
        grid_x, grid_y, _grid_z = torch.meshgrid(dz, dy, dx, indexing='ij')
        scores = torch.stack(_scores_des_vues(grid_x, grid_y, views), dim=0)  # (NumViews, D, H, W)
        
        # Softmax over views dimension (0)
        weights = torch.softmax(scores * blend_temperature, dim=0)
        
        # Reshape for broadcasting: (NumViews, 1, 1, D, H, W) -> No wait, loop is over views.
        # We want to return something we can index like weights[i] -> (1, 1, D, H, W)
        
        # Current shape: (NumViews, D, H, W)
        return weights

    @staticmethod
    def _memes_conditions(a, b) -> bool:
        """Deux conditionnements identiques ? (tenseurs, dicts de tenseurs,
        tenseurs creux). Le moindre doute vaut « non »."""
        try:
            if isinstance(a, dict) and isinstance(b, dict):
                return a.keys() == b.keys() and all(
                    FlowEulerMultiViewSampler._memes_conditions(a[k], b[k]) for k in a)
            if hasattr(a, 'feats') and hasattr(b, 'feats'):
                return (a.feats.shape == b.feats.shape and torch.equal(a.feats, b.feats)
                        and torch.equal(a.coords, b.coords))
            if torch.is_tensor(a) and torch.is_tensor(b):
                return a.shape == b.shape and torch.equal(a, b)
        except Exception:                                     # noqa: BLE001
            pass
        return False

    def _predictions_par_vue(self, model, x_t, t, conds, views, **kwargs):
        """Une prédiction du modèle par vue, avec le guidage sans classifieur
        de `ClassifierFreeGuidanceSamplerMixin` et l'intervalle de
        `GuidanceIntervalSamplerMixin`, MAIS LE PASSAGE NÉGATIF N'EST CALCULÉ
        QU'UNE FOIS.

        Le guidage compare, pour chaque vue, la prédiction conditionnée par
        sa photo à la prédiction sans photo. Cette seconde prédiction ne
        dépend pas de la vue : `get_cond` la construit en zéros de la même
        forme pour chacune. Le code d'origine la recalculait pourtant pour
        chaque vue : quatre vues, huit passages du modèle par pas. Ici cinq,
        et le résultat est le même au bit près (banc du 4 septembre 2026,
        Standard sur la 4060 : la forme et la texture pesaient 300 et
        310 s). Si les négatifs différaient d'une vue à l'autre, chacune
        garde le sien.
        """
        force = kwargs.pop('guidance_strength', 1.0)
        intervalle = kwargs.pop('guidance_interval', None)
        rescale = kwargs.pop('guidance_rescale', 0.0)
        base = FlowEulerSampler._inference_model

        def separe(c):
            if isinstance(c, dict) and 'cond' in c and 'neg_cond' in c:
                return c['cond'], c['neg_cond']
            return c, None

        paires = [separe(conds[v]) for v in views]
        dans_intervalle = intervalle is None or intervalle[0] <= t <= intervalle[1]
        actif = dans_intervalle and force not in (0, 1)

        if not actif:
            if force == 0 and dans_intervalle:
                for c, n in paires:
                    yield base(self, model, x_t, t, n if n is not None else c, **kwargs)
                return
            for c, _ in paires:
                yield base(self, model, x_t, t, c, **kwargs)
            return

        # Le négatif commun ne se calcule qu'une fois, entre les vues qui en
        # ont un et qui ont le même ; une vue sans négatif garde son positif
        # seul, sans éteindre le guidage des autres.
        negatifs = [n for _, n in paires if n is not None]
        partage = bool(negatifs) and all(
            self._memes_conditions(negatifs[0], n) for n in negatifs[1:])
        neg_commun = base(self, model, x_t, t, negatifs[0], **kwargs) if partage else None

        for c, n in paires:
            pred_pos = base(self, model, x_t, t, c, **kwargs)
            if n is None:
                yield pred_pos
                continue
            pred_neg = neg_commun if partage else base(self, model, x_t, t, n, **kwargs)
            pred = force * pred_pos + (1 - force) * pred_neg
            if rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = rescale * x_0_rescaled + (1 - rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)
                del x_0_pos, x_0_cfg, x_0_rescaled, x_0
            del pred_pos, pred_neg
            yield pred

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        conds: Dict[str, Any], # Changed: expects dict of {view: cond}
        views: List[str],      # Changed: list of view keys corresponding to conds
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        **kwargs
    ):
        """
        Sample with multi-view blending.
        """
        is_sparse = hasattr(x_t, 'coords')
        
        if is_sparse:
            # 1. Compute per-voxel weights based on current sparse coords
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
            # weights: (N, NumViews)
        else:
            # Dense tensor (B, C, D, H, W)
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            # weights: (NumViews, D, H, W)
        
        # 2. Run model for each view and blend predictions
        pred_v_accum = 0

        # UNE PREDICTION A LA FOIS. Le generateur ne garde vivantes que la
        # prediction negative commune et celle de la vue en cours : garder
        # les quatre d'un coup coutait ~0,5 Go sur la texture en Standard
        # (banc du samourai, 4 septembre 2026).
        for i, pred_v_view in enumerate(
                self._predictions_par_vue(model, x_t, t, conds, views, **kwargs)):

            # Weighted accumulation
            if is_sparse:
                # weights[:, i] is (N,), pred_v_view might be SparseTensor or Tensor (N, C)
                w = weights[:, i].unsqueeze(1)
                
                v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                pred_v_accum += v_feats * w
            else:
                # Dense
                # weights[i] is (D, H, W). pred_v_view is (B, C, D, H, W)
                w = weights[i].unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
                pred_v_accum += pred_v_view * w
                
        if is_sparse:
            # Re-wrap accumulated features into a SparseTensor matching x_t
            # pred_v_accum is (N, C) tensor now
            pred_v = x_t.replace(feats=pred_v_accum)
        else:
            pred_v = pred_v_accum
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)

        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        conds: Dict[str, Any], # {view: cond}
        views: List[str],      # ['front', 'back', ...]
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling MultiView",
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        grille: int | None = None,
        **kwargs
    ):
        # La grille du mélange, posée sur l'instance pour que TOUS les
        # échantillonneurs multi-vue (Euler, RK4, RK5, Heun) la lisent sans
        # changer de signature, et sans qu'elle atteigne jamais le modèle.
        self._grille = grille
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc):
            out = self.sample_once(
                model, sample, t, t_prev, 
                conds=conds, 
                views=views,
                front_axis=front_axis, 
                blend_temperature=blend_temperature, 
                **kwargs
            )
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerMultiViewGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerMultiViewSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with multi-view blending, CFG, and guidance interval.
    """
    pass
    
# RK4 and RK5 Samplers

class FlowRK4Sampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using the 4th-order Runge-Kutta method.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # RK4 intermediate slopes
        k1 = get_v(x_t, t)
        k2 = get_v(x_t + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = get_v(x_t + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = get_v(x_t + dt * k3, t + dt)
        
        # RK4 integration
        pred_x_prev = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # We need to return pred_x_0 as well to satisfy the pipeline's logging/tracking
        # We compute x_start_eps based on the k1 velocity (equivalent to the Euler estimation of x_0)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK5Sampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Butcher's 5th-order Runge-Kutta method.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # Intermediate time step fractions for Butcher's RK5
        c2, c3, c4, c5, c6 = 1/4, 1/4, 1/2, 3/4, 1.0
        
        k1 = get_v(x_t, t)
        k2 = get_v(x_t + dt * (1/4 * k1), t + dt * c2)
        k3 = get_v(x_t + dt * (1/8 * k1 + 1/8 * k2), t + dt * c3)
        k4 = get_v(x_t + dt * (-1/2 * k2 + 1.0 * k3), t + dt * c4)
        k5 = get_v(x_t + dt * (3/16 * k1 + 9/16 * k4), t + dt * c5)
        k6 = get_v(x_t + dt * (-3/7 * k1 + 2/7 * k2 + 12/7 * k3 - 12/7 * k4 + 8/7 * k5), t + dt * c6)
        
        # Final RK5 Integration
        pred_x_prev = x_t + dt * (7/90 * k1 + 32/90 * k3 + 12/90 * k4 + 32/90 * k5 + 7/90 * k6)
        
        # Estimate x_0 based on k1 for tracking
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


# --- Classifier Free Guidance (CFG) Wrappers ---
class FlowRK4CfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowRK4Sampler):
    """RK4 sampling with classifier-free guidance."""
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0, guidance_strength: float = 3.0, verbose: bool = True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)

class FlowRK5CfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowRK5Sampler):
    """RK5 sampling with classifier-free guidance."""
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0, guidance_strength: float = 3.0, verbose: bool = True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)
        
class FlowRK4GuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK4Sampler):
    """RK4 with CFG, Guidance Intervals, and optional DINO lock."""
    pass

class FlowRK5GuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK5Sampler):
    """RK5 with CFG, Guidance Intervals, and optional DINO lock."""
    pass        
    
# RK4 and RK5 for MultiView

class FlowRK4MultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using 4th-order Runge-Kutta."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        # Calculate spatial blending weights ONCE for the current step
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        # Helper function to compute the blended velocity for a given intermediate x and t
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # RK4 Evaluations
        k1 = get_blended_v(x_t, t)
        k2 = get_blended_v(x_t + k1 * (0.5 * dt), t + 0.5 * dt)
        k3 = get_blended_v(x_t + k2 * (0.5 * dt), t + 0.5 * dt)
        k4 = get_blended_v(x_t + k3 * dt, t + dt)
        
        pred_x_prev = x_t + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6.0)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK5MultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using Butcher's 5th-order Runge-Kutta."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # Butcher Tableau Intermediate steps
        c2, c3, c4, c5, c6 = 1/4, 1/4, 1/2, 3/4, 1.0
        
        k1 = get_blended_v(x_t, t)
        k2 = get_blended_v(x_t + k1 * (dt * 1/4), t + dt * c2)
        k3 = get_blended_v(x_t + (k1 * 1/8 + k2 * 1/8) * dt, t + dt * c3)
        k4 = get_blended_v(x_t + (k2 * -1/2 + k3 * 1.0) * dt, t + dt * c4)
        k5 = get_blended_v(x_t + (k1 * 3/16 + k4 * 9/16) * dt, t + dt * c5)
        k6 = get_blended_v(x_t + (k1 * -3/7 + k2 * 2/7 + k3 * 12/7 + k4 * -12/7 + k5 * 8/7) * dt, t + dt * c6)
        
        pred_x_prev = x_t + (k1 * 7/90 + k3 * 32/90 + k4 * 12/90 + k5 * 32/90 + k6 * 7/90) * dt
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK4MultiViewGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK4MultiViewSampler):
    pass

class FlowRK5MultiViewGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK5MultiViewSampler):
    pass

# Heun (RK2)

class FlowHeunSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Heun's Method (2nd-order Runge-Kutta).
    Requires 2 NFEs per step.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # Step 1: Predictor (Euler step)
        k1 = get_v(x_t, t)
        x_temp = x_t + k1 * dt
        
        # Step 2: Corrector
        k2 = get_v(x_temp, t + dt)
        
        # Average the two velocities for the final step
        pred_x_prev = x_t + 0.5 * dt * (k1 + k2)
        
        # Estimate x_0 based on k1 for tracking/logging
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

# --- CFG Wrapper for Heun ---
class FlowHeunGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowHeunSampler):
    """Heun sampling with CFG, Guidance Intervals, and optional DINO lock."""
    pass
    
class FlowHeunMultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using Heun's method (2nd-order Runge-Kutta)."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        # Calculate spatial blending weights ONCE for the current step
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        # Helper function to compute the blended velocity for a given intermediate x and t
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # Heun's Method (RK2) Evaluations
        # Step 1: Predictor (Euler step)
        k1 = get_blended_v(x_t, t)
        x_temp = x_t + k1 * dt
        
        # Step 2: Corrector
        k2 = get_blended_v(x_temp, t + dt)
        
        # Combine
        pred_x_prev = x_t + 0.5 * dt * (k1 + k2)
        
        # Estimate x_0 based on k1 for tracking
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

# --- CFG Wrapper for Heun Multi-View ---
class FlowHeunMultiViewGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowHeunMultiViewSampler):
    pass    