"""pi0.5: a vision-language-action policy with a flow-matching action expert.

Inference proceeds in two phases:

1. **Prefix prefill.** Camera images go through SigLIP and are projected into
   the language model's width; the prompt — task text *plus the discretized
   robot state* — is embedded from the token table. The 2B Gemma runs once over
   this prefix with full bidirectional attention, and its per-layer keys/values
   are cached.

2. **Flow-matching denoise.** A `(chunk_size, max_action_dim)` Gaussian sample
   is integrated from `t = 1` to `t = 0` by the 300M action expert. Each Euler
   step re-runs only the expert, attending into the cached prefix. The timestep
   conditions the expert through adaptive RMSNorm rather than through a token.

Unlike pi0, pi0.5 has no `state_proj`: proprioception enters as text (256-bin
discretization) in the prompt, which is why the state must already be
normalized to [-1, 1] before the prompt is built.
"""

from __future__ import annotations

import logging
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from mrs.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from mrs.policies.common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from mrs.policies.common.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    prepare_attention_masks_4d,
    resize_with_pad,
)
from mrs.policies.pi_gemma import (
    GemmaRotaryEmbedding,
    KVCache,
    MultiModalProjector,
    PiGemmaStack,
    SiglipVisionTower,
    apply_rotary_pos_emb,
    gated_residual,
    get_gemma_config,
    joint_attention,
)
from mrs.policies.pretrained import PreTrainedPolicy, _report_load
from mrs.policies.pi05.configuration_pi05 import PI05Config

logger = logging.getLogger(__name__)

SIGLIP_WIDTH = 1152
PALIGEMMA_VOCAB_SIZE = 257152


class PaliGemmaBackbone(nn.Module):
    """The vision-language half: SigLIP eye, projector, and the 2B Gemma stack.

    `lm_head` doubles as the input embedding table — the published checkpoints
    tie them and ship only `lm_head.weight`.
    """

    def __init__(self, config, image_size: int):
        super().__init__()
        self.model = nn.Module()
        self.model.vision_tower = SiglipVisionTower(image_size=image_size)
        self.model.multi_modal_projector = MultiModalProjector(SIGLIP_WIDTH, config.width)
        self.model.language_model = PiGemmaStack(config, cond_dim=None)
        self.lm_head = nn.Linear(config.width, PALIGEMMA_VOCAB_SIZE, bias=False)


class ActionExpertBackbone(nn.Module):
    """The 300M Gemma stack that carries the action tokens."""

    def __init__(self, config):
        super().__init__()
        # Conditioned on the timestep MLP output, which has the expert's width.
        self.model = PiGemmaStack(config, cond_dim=config.width)


class PaliGemmaWithExpert(nn.Module):
    """Both Gemma stacks, plus the layer-wise joint attention that couples them."""

    def __init__(self, vlm_config, expert_config, image_size: int):
        super().__init__()
        self.vlm_config = vlm_config
        self.expert_config = expert_config

        self.paligemma = PaliGemmaBackbone(vlm_config, image_size=image_size)
        self.gemma_expert = ActionExpertBackbone(expert_config)

        self.rotary_emb = GemmaRotaryEmbedding(vlm_config.head_dim, base=vlm_config.rope_theta)
        self.scaling = vlm_config.head_dim**-0.5

    # ---- embeddings -----------------------------------------------------
    def embed_image(self, image: Tensor) -> Tensor:
        """SigLIP patch features projected into the language model's width."""
        features = self.paligemma.model.vision_tower(image)
        features = self.paligemma.model.multi_modal_projector(features)
        return features / (self.vlm_config.width**0.5)

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        return F.embedding(tokens, self.paligemma.lm_head.weight)

    # ---- forward variants ----------------------------------------------
    def prefill_prefix(
        self, prefix_embs: Tensor, attention_mask_4d: Tensor, position_ids: Tensor
    ) -> KVCache:
        """Run the 2B stack over the prefix and cache every layer's keys/values.

        The prefix hidden states themselves are discarded — only the cache is
        consumed downstream — so the final norm is skipped.
        """
        stack = self.paligemma.model.language_model
        cache = KVCache(len(stack.layers))

        hidden = prefix_embs.to(stack.layers[0].self_attn.q_proj.weight.dtype)
        cos, sin = self.rotary_emb(position_ids, hidden.dtype)

        for layer_idx, layer in enumerate(stack.layers):
            normed, gate = layer.input_layernorm(hidden)
            q, k, v = layer.self_attn.project_qkv(normed)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            cache.set(layer_idx, k, v)

            attn = joint_attention(q, k, v, attention_mask_4d, self.scaling)
            attn = attn.reshape(hidden.shape[0], hidden.shape[1], -1)
            hidden = self._apply_output_blocks(layer, hidden, attn, gate, adarms_cond=None)

        return cache

    def forward_expert_with_cache(
        self,
        suffix_embs: Tensor,
        cache: KVCache,
        attention_mask_4d: Tensor,
        position_ids: Tensor,
        adarms_cond: Tensor,
    ) -> Tensor:
        """Run the action expert, attending into the cached prefix."""
        stack = self.gemma_expert.model
        hidden = suffix_embs.to(stack.layers[0].self_attn.q_proj.weight.dtype)
        cos, sin = self.rotary_emb(position_ids, hidden.dtype)

        for layer_idx, layer in enumerate(stack.layers):
            normed, gate = layer.input_layernorm(hidden, cond=adarms_cond)
            q, k, v = layer.self_attn.project_qkv(normed)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            prefix_k, prefix_v = cache.get(layer_idx)
            k = torch.cat([prefix_k.to(k.dtype), k], dim=2)
            v = torch.cat([prefix_v.to(v.dtype), v], dim=2)

            attn = joint_attention(q, k, v, attention_mask_4d, self.scaling)
            attn = attn.reshape(hidden.shape[0], hidden.shape[1], -1)
            hidden = self._apply_output_blocks(layer, hidden, attn, gate, adarms_cond=adarms_cond)

        hidden, _ = stack.norm(hidden, adarms_cond)
        return hidden

    def forward_joint(
        self,
        prefix_embs: Tensor,
        suffix_embs: Tensor,
        attention_mask_4d: Tensor,
        position_ids: Tensor,
        adarms_cond: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run both stacks in lockstep over one shared attention (training path)."""
        prefix_layers = self.paligemma.model.language_model.layers
        expert_layers = self.gemma_expert.model.layers

        hidden = [
            prefix_embs.to(prefix_layers[0].self_attn.q_proj.weight.dtype),
            suffix_embs.to(expert_layers[0].self_attn.q_proj.weight.dtype),
        ]
        conds = [None, adarms_cond]
        cos, sin = self.rotary_emb(position_ids, hidden[0].dtype)

        for prefix_layer, expert_layer in zip(prefix_layers, expert_layers, strict=True):
            layers = (prefix_layer, expert_layer)

            queries, keys, values, gates = [], [], [], []
            for i, layer in enumerate(layers):
                normed, gate = layer.input_layernorm(hidden[i], cond=conds[i])
                q, k, v = layer.self_attn.project_qkv(normed)
                queries.append(q)
                keys.append(k)
                values.append(v)
                gates.append(gate)

            q = torch.cat(queries, dim=2)
            k = torch.cat(keys, dim=2)
            v = torch.cat(values, dim=2)
            q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))

            attn = joint_attention(q, k, v, attention_mask_4d, self.scaling)
            attn = attn.reshape(attn.shape[0], attn.shape[1], -1)

            start = 0
            for i, layer in enumerate(layers):
                end = start + hidden[i].shape[1]
                hidden[i] = self._apply_output_blocks(
                    layer, hidden[i], attn[:, start:end], gates[i], adarms_cond=conds[i]
                )
                start = end

        prefix_out, _ = self.paligemma.model.language_model.norm(hidden[0], conds[0])
        suffix_out, _ = self.gemma_expert.model.norm(hidden[1], conds[1])
        return prefix_out, suffix_out

    @staticmethod
    def _apply_output_blocks(layer, hidden: Tensor, attn: Tensor, gate, adarms_cond) -> Tensor:
        """Attention output projection, MLP, and the two gated residuals."""
        attn = attn.to(layer.self_attn.o_proj.weight.dtype)
        out = layer.self_attn.o_proj(attn)
        out = gated_residual(hidden, out.to(hidden.dtype), gate)

        residual = out
        out, gate2 = layer.post_attention_layernorm(out, cond=adarms_cond)
        out = layer.mlp(out)
        return gated_residual(residual, out.to(residual.dtype), gate2)


class PI05Model(nn.Module):
    """The pi0.5 network: backbones plus the action/time projections."""

    def __init__(self, config: PI05Config):
        super().__init__()
        self.config = config

        vlm_config = get_gemma_config(config.paligemma_variant)
        expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpert(
            vlm_config, expert_config, image_size=config.image_resolution[0]
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, expert_config.width)
        self.action_out_proj = nn.Linear(expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(expert_config.width, expert_config.width)
        self.time_mlp_out = nn.Linear(expert_config.width, expert_config.width)

        self.apply_precision_policy(config.dtype)

    def apply_precision_policy(self, precision: str) -> None:
        """Set each parameter's runtime dtype.

        Weights are *not* kept at whatever dtype the checkpoint happened to be
        saved in — `load_state_dict` casts incoming tensors into the dtypes
        established here, which is how the reference implementation behaves and
        what its published numbers correspond to.

        Under `bfloat16`, the matmul-heavy parameters go to bfloat16 while the
        vision path, every RMSNorm, and the action/time projections stay in
        float32. Keeping the norms in float32 matters for accuracy; keeping the
        vision tower homogeneous also avoids mixed-dtype kernels that some
        backends (notably MPS) reject outright.
        """
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Unsupported precision {precision!r}; expected 'bfloat16' or 'float32'.")

        self.paligemma_with_expert.to(dtype=torch.bfloat16)

        keep_float32 = (
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, param in self.paligemma_with_expert.named_parameters():
            if any(selector in name for selector in keep_float32):
                param.data = param.data.to(torch.float32)

        for module in (self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out):
            module.to(dtype=torch.float32)

    # ---- prefix / suffix construction -----------------------------------
    def embed_prefix(
        self, images: list[Tensor], img_masks: list[Tensor], tokens: Tensor, token_masks: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Concatenate image patch tokens and prompt tokens into one block.

        Every prefix token gets block id 0, so the prefix attends to itself
        bidirectionally (subject to padding).
        """
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            batch_size, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        embs.append(lang_emb.to(embs[0].dtype) if embs else lang_emb)
        pad_masks.append(token_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        """Project the noisy action chunk and build the timestep conditioning."""
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(timestep.dtype)

        action_emb = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))

        time_emb = F.silu(self.time_mlp_in(time_emb))
        adarms_cond = F.silu(self.time_mlp_out(time_emb))

        batch_size, chunk_len = action_emb.shape[:2]
        pad_masks = torch.ones(batch_size, chunk_len, dtype=torch.bool, device=timestep.device)

        # Block id 1: action tokens see the prefix and each other, but the
        # prefix cannot see them.
        att_masks = torch.tensor(
            [1] + [0] * (chunk_len - 1), dtype=torch.bool, device=action_emb.device
        )
        att_masks = att_masks[None, :].expand(batch_size, chunk_len)

        return action_emb, pad_masks, att_masks, adarms_cond

    # ---- training -------------------------------------------------------
    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        token_masks: Tensor,
        actions: Tensor,
        noise: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Per-element flow-matching regression loss."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, tokens, token_masks)
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        _, suffix_out = self.paligemma_with_expert.forward_joint(
            prefix_embs,
            suffix_embs,
            prepare_attention_masks_4d(att_2d),
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :].to(torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    # ---- inference ------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        token_masks: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Integrate the flow to produce one action chunk."""
        num_steps = num_steps or self.config.num_inference_steps
        batch_size = tokens.shape[0]
        device = tokens.device

        if noise is None:
            noise = sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, tokens, token_masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_position_ids = torch.cumsum(prefix_pad, dim=1) - 1

        cache = self.paligemma_with_expert.prefill_prefix(
            prefix_embs, prepare_attention_masks_4d(prefix_att_2d), prefix_position_ids
        )

        return euler_integrate(
            lambda x_t, t: self._denoise_step(prefix_pad, cache, x_t, t),
            noise,
            num_steps,
        )

    def _denoise_step(self, prefix_pad: Tensor, cache: KVCache, x_t: Tensor, timestep: Tensor) -> Tensor:
        """One Euler step: predict the velocity field at `(x_t, timestep)`."""
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(x_t, timestep)

        batch_size, prefix_len = prefix_pad.shape
        suffix_len = suffix_pad.shape[1]

        # Action tokens attend to every non-padded prefix token, plus to the
        # whole action block.
        prefix_2d = prefix_pad[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d = torch.cat([prefix_2d, suffix_2d], dim=2)

        prefix_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1

        suffix_out = self.paligemma_with_expert.forward_expert_with_cache(
            suffix_embs,
            cache,
            prepare_attention_masks_4d(full_att_2d),
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :].to(torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """pi0.5 wrapped in the policy interface: normalized batch in, action out."""

    config_class = PI05Config
    name = "pi05"

    def __init__(self, config: PI05Config):
        super().__init__(config)
        config.validate_features()
        self.model = PI05Model(config)
        self.reset()

    # ---- action selection ----------------------------------------------
    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    @property
    def pending_actions(self) -> int:
        """How many actions remain from the last generated chunk."""
        return len(self._action_queue)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Pop the next action, generating a fresh chunk when the queue drains."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # (batch, steps, dim) -> steps entries of (batch, dim)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        images, img_masks = self._prepare_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.sample_actions(images, img_masks, tokens, token_masks)

        # Strip the padding dimensions back off.
        action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :action_dim]

    # ---- training -------------------------------------------------------
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        images, img_masks = self._prepare_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        noise = sample_noise(actions.shape, actions.device)
        time = sample_time_beta(
            actions.shape[0],
            actions.device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )

        losses = self.model.forward(images, img_masks, tokens, token_masks, actions, noise, time)
        losses = losses[:, :, : self.config.output_features[ACTION].shape[0]]

        loss = losses.mean()
        return loss, {"loss": loss.item(), "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().tolist()}

    # ---- image handling --------------------------------------------------
    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Resize, rescale to [-1, 1], and pad out the unused camera slots.

        Camera slots the checkpoint declares but the batch does not provide are
        filled with an all -1 image and a zeroed mask, matching how the model
        was trained on embodiments with fewer than three cameras.
        """
        device = next(self.parameters()).device
        images, img_masks = [], []

        present = [key for key in self.config.image_features if key in batch]
        missing = [key for key in self.config.image_features if key not in batch]
        if not present:
            raise ValueError(
                f"No image features found in batch. Expected any of {list(self.config.image_features)}, "
                f"got {sorted(batch)}."
            )

        for key in present:
            img = batch[key].to(device=device, dtype=torch.float32)
            if img.shape[-2:] != tuple(self.config.image_resolution):
                img = resize_with_pad(img, *self.config.image_resolution)
            # SigLIP expects [-1, 1]; the batch arrives in [0, 1].
            images.append(img * 2.0 - 1.0)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))

        for _ in missing:
            images.append(torch.full_like(images[0], -1.0))
            img_masks.append(torch.zeros_like(img_masks[0]))

        return images, img_masks

    # ---- weight loading --------------------------------------------------
    def load_pretrained_state_dict(self, state_dict: dict[str, Tensor], *, strict: bool = True) -> None:
        """Load a published pi0.5 checkpoint, reconciling naming differences.

        Published checkpoints differ from this module tree in three ways:
        the `model.` prefix is sometimes absent, the pi0-era `action_time_mlp_*`
        names were renamed to `time_mlp_*`, and the expert's unused `lm_head` is
        stored but not instantiated here.
        """
        remapped: dict[str, Tensor] = {}
        dropped: list[str] = []

        for key, value in state_dict.items():
            new_key = key if key.startswith("model.") else f"model.{key}"

            new_key = new_key.replace("action_time_mlp_in.", "time_mlp_in.")
            new_key = new_key.replace("action_time_mlp_out.", "time_mlp_out.")

            # pi0 carried a projection for continuous state; pi0.5 tokenizes it.
            if ".state_proj." in new_key or new_key.startswith("model.state_proj."):
                dropped.append(key)
                continue

            # The expert's output vocabulary head is never used for control.
            if new_key.endswith("paligemma_with_expert.gemma_expert.lm_head.weight"):
                dropped.append(key)
                continue

            remapped[new_key] = value

        # `load_state_dict` casts each incoming tensor into the dtype the
        # parameter already has, so the precision policy set at construction
        # wins over the dtypes the checkpoint file was serialized with.
        missing, unexpected = self.load_state_dict(remapped, strict=False)

        if dropped:
            logger.info("Ignored %d checkpoint key(s) unused at inference: %s", len(dropped), dropped[:3])
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Checkpoint does not match the model.\n  missing: {missing[:10]}\n"
                f"  unexpected: {unexpected[:10]}"
            )
        _report_load(missing, unexpected)
