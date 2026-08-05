import math

import torch

# import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin

from .attention import attention


# --- TEMPORARY DEBUG INSTRUMENTATION (round 2) ---
# Round 1 confirmed self_attn (including attention() itself) is NOT at
# fault, and found + fixed a missing dtype cast in WanT2VCrossAttention.
# That fix did not resolve the crash, so it wasn't the whole picture (or
# wasn't it at all) — extending checkpoints further this time: inside
# cross_attn's own internals, through ffn, and past the end of the block,
# rather than guessing again.
def _dbg_sync(label):
    if 0:
        print(f"[DEBUG SYNC] about to sync after: {label}", flush=True)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        print(f"[DEBUG SYNC] survived: {label}", flush=True)


# --- END TEMPORARY DEBUG INSTRUMENTATION ---


__all__ = ["WanModel"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    orig_device = position.device
    # MPS does not support float64 tensors at all, so do this tiny
    # computation on CPU (in float64, for precision) and move the
    # result back to the original device afterward.
    if orig_device.type == "mps":
        position = position.detach().to("cpu")
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if orig_device.type == "mps":
        # x is still float64 here (from the CPU computation above) — MPS
        # can't hold float64 tensors at all, so the dtype cast has to happen
        # in the same .to() call as the device move, not after it.
        x = x.to(dtype=torch.float32, device=orig_device)
    return x


# @amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast("cuda", enabled=False)
def rope_apply(
    x,
    grid_sizes,
    freqs,
    context_window_size=0,
    num_token_list=[],
    num_frame_list=[],
    grid_size_list=[],
):
    n, c = x.size(2), x.size(3) // 2
    bs = x.size(0)
    orig_dtype = x.dtype  # rope math below runs in float32 for the complex
    # rotation; restore this dtype before returning so q/k match v's dtype.

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    f, h, w = grid_sizes.tolist()

    if context_window_size == 0 and len(num_frame_list) > 0:
        seq_len = x.shape[1]
        num_frame = f - sum(num_frame_list)

        # precompute multipliers
        x = torch.view_as_complex(x.to(torch.float32).reshape(bs, seq_len, n, -1, 2))

        latent_seq_len = num_frame * h * w
        freqs_i = torch.cat(
            [
                freqs[0][:num_frame]
                .view(num_frame, 1, 1, -1)
                .expand(num_frame, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(num_frame, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(num_frame, h, w, -1),
            ],
            dim=-1,
        ).reshape(latent_seq_len, 1, -1)

        freqs_context_list = []
        for ii, num_frame in enumerate(num_frame_list):
            if ii == 0:
                start = 1024
            else:
                start = 1024 + sum(num_frame_list[:ii])

            freqs_temp = torch.cat(
                [
                    freqs[0][start : start + num_frame]
                    .view(num_frame, 1, 1, -1)
                    .expand(num_frame, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(num_frame, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(num_frame, h, w, -1),
                ],
                dim=-1,
            ).reshape(num_token_list[ii], 1, -1)
            freqs_context_list.append(freqs_temp)
        freqs_context = torch.cat(freqs_context_list, dim=0)
        freqs_i = torch.cat([freqs_i, freqs_context], dim=0)
        # apply rotary embedding
        x = torch.view_as_real(x * freqs_i).flatten(3)
    elif context_window_size != 0:
        seq_len = x.shape[1]
        # precompute multipliers
        x = torch.view_as_complex(x.to(torch.float32).reshape(bs, seq_len, n, -1, 2))

        num_latent_frame = f - sum(num_frame_list)
        latent_seq_len = num_latent_frame * h * w

        freqs_list = []
        for i, num_frame in enumerate(num_frame_list):
            if i == 0:
                start = 0
            else:
                start = sum(num_frame_list[:i])

            _, c_h, c_w = grid_size_list[i]
            end = start + num_frame
            freqs_tmp = torch.cat(
                [
                    freqs[0][start:end]
                    .view(num_frame, 1, 1, -1)
                    .expand(num_frame, c_h, c_w, -1),
                    freqs[1][:c_h].view(1, c_h, 1, -1).expand(num_frame, c_h, c_w, -1),
                    freqs[2][:c_w].view(1, 1, c_w, -1).expand(num_frame, c_h, c_w, -1),
                ],
                dim=-1,
            ).reshape(num_frame * c_h * c_w, 1, -1)

            freqs_list.append(freqs_tmp)

        freqs_i = torch.cat(
            [
                freqs[0][sum(num_frame_list) : f]
                .view(num_latent_frame, 1, 1, -1)
                .expand(num_latent_frame, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(num_latent_frame, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(num_latent_frame, h, w, -1),
            ],
            dim=-1,
        ).reshape(latent_seq_len, 1, -1)
        freqs_list.append(freqs_i)

        freqs_i = torch.cat(freqs_list, dim=0)

        # apply rotary embedding
        x = torch.view_as_real(x * freqs_i).flatten(3)
    else:
        # Standard rope apply (transformer_1)
        seq_len = f * h * w
        x = torch.view_as_complex(x.to(torch.float32).reshape(bs, seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x = torch.view_as_real(x * freqs_i).flatten(3)

    return x.to(orig_dtype)


def fast_rms_norm(x, weight, eps):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    # NOTE: this was `x.type_as(x)`, a no-op that silently left the output in
    # float32 instead of casting back to the model's working dtype (bf16/fp16).
    # Under the old attention() that force-cast q/k/v to a hardcoded bf16 no
    # matter what, this went unnoticed; with that force-cast removed, q/k
    # (which pass through this norm) and v (which doesn't) would otherwise
    # end up in different dtypes and fail the SDPA dtype check.
    x = x.type_as(weight) * weight
    return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return fast_rms_norm(x, self.weight, self.eps)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # PyTorch's MPS backend has long-standing, documented bugs running
        # LayerNorm directly on float16 input (e.g. pytorch/pytorch#96113,
        # huggingface/diffusers#2521) — internally its variance computation
        # can end up in a different dtype than its output buffer, which can
        # surface as a native Metal assertion failure. Because MPS dispatches
        # GPU work asynchronously, that crash can appear to originate from a
        # completely different, later op that happened to force a sync —
        # which is exactly what we saw here (the same crash from three
        # different attention implementations). fast_rms_norm already
        # upcasts to float32 for exactly this kind of reason; this class
        # never did, so it's forced explicitly here instead of relying on
        # nn.LayerNorm's default behavior on whatever dtype x arrives in.
        orig_dtype = x.dtype
        out = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.to(orig_dtype)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        grid_sizes,
        freqs,
        block_mask=None,
        context_window_size=0,
        num_token_list=[],
        num_frame_list=[],
        grid_size_list=[],
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        x = x.to(self.q.weight.dtype)
        q, k, v = qkv_fn(x)

        q = rope_apply(
            q,
            grid_sizes,
            freqs,
            context_window_size,
            num_token_list,
            num_frame_list,
            grid_size_list,
        )
        k = rope_apply(
            k,
            grid_sizes,
            freqs,
            context_window_size,
            num_token_list,
            num_frame_list,
            grid_size_list,
        )
        x = attention(q=q, k=k, v=v, window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # x arrives as float32 here (mul_add() upstream always upcasts to
        # float32 unconditionally), but self.q/self.k/self.v are fp16/bf16
        # weights. WanSelfAttention.forward casts x to the weight's dtype
        # before its own Linear calls; this class was missing that same
        # cast, so self.q(x) ran a Linear with mismatched float32 input
        # against fp16 weights — a real dtype mismatch feeding a matmul,
        # which is what was actually causing the native MPS assertion
        # failure (not attention itself, which was fine all along).
        x = x.to(self.q.weight.dtype)
        context = context.to(self.k.weight.dtype)
        _dbg_sync("cross_attn: after x/context dtype cast")

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        _dbg_sync("cross_attn: after norm_q(q_proj(x))")
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        _dbg_sync("cross_attn: after norm_k(k_proj(context))")
        v = self.v(context).view(b, -1, n, d)
        _dbg_sync("cross_attn: after v_proj(context)")

        # compute attention
        x = attention(q, k, v)
        _dbg_sync("cross_attn: after attention()")

        # output
        x = x.flatten(2)
        x = self.o(x)
        _dbg_sync("cross_attn: after output proj self.o(x)")
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # See WanT2VCrossAttention.forward for why this cast is needed.
        x = x.to(self.q.weight.dtype)
        context = context.to(self.k.weight.dtype)
        context_img = context_img.to(self.k_img.weight.dtype)

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = attention(q, k_img, v_img)
        # compute attention
        x = attention(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


def mul_add(x, y, z):
    return x.float() + y.float() * z.float()


def mul_add_add(x, y, z):
    return x.float() * (1 + y) + z


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        context,
        block_mask=None,
        context_window_size=0,
        num_token_list=[],
        num_frame_list=[],
        grid_size_list=[],
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # with amp.autocast("cuda", dtype=torch.float32):
        e = (self.modulation.unsqueeze(2) + e.transpose(1, 2)).chunk(6, dim=1)
        e = [e_i.transpose(1, 2) for e_i in e]  # [B, F, 1, C] * 6
        expand_rate = x.shape[1] // e[0].shape[1]
        assert x.shape[1] % e[0].shape[1] == 0
        _dbg_sync("block: after modulation chunk")

        # self-attention
        y = self.self_attn(
            mul_add_add(
                self.norm1(x).unflatten(1, (-1, expand_rate)), e[1], e[0]
            ).flatten(1, 2),
            grid_sizes,
            freqs,
            block_mask,
            context_window_size,
            num_token_list,
            num_frame_list,
            grid_size_list,
        )
        _dbg_sync("block: after self_attn returned")
        # with amp.autocast("cuda", dtype=torch.float32):
        x = mul_add(
            x.unflatten(1, (-1, expand_rate)),
            y.unflatten(1, (-1, expand_rate)),
            e[2],
        ).flatten(1, 2)
        _dbg_sync("block: after post-self-attn mul_add")

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(self.norm3(x), context)
            _dbg_sync("block: after cross_attn")
            ffn_input = mul_add_add(
                self.norm2(x).unflatten(1, (-1, expand_rate)), e[4], e[3]
            ).flatten(1, 2)
            # Same bug class as WanT2VCrossAttention: mul_add_add() always
            # returns float32, but self.ffn's first Linear has fp16/bf16
            # weights. Cast explicitly rather than relying on an implicit
            # (and apparently, on this MPS build, crash-prone) promotion.
            ffn_input = ffn_input.to(self.ffn[0].weight.dtype)
            _dbg_sync("block: after ffn input dtype cast")
            y = self.ffn(ffn_input)
            _dbg_sync("block: after ffn")
            # with amp.autocast("cuda", dtype=torch.float32):
            x = mul_add(
                x.unflatten(1, (-1, expand_rate)),
                y.unflatten(1, (-1, expand_rate)),
                e[5],
            ).flatten(1, 2)
            _dbg_sync("block: after post-ffn mul_add")
            return x

        x = cross_attn_ffn(x, context, e)
        _dbg_sync("block: after cross_attn_ffn returned")
        return x.to(self.modulation.dtype)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # with amp.autocast("cuda", dtype=torch.float32):
        e = (self.modulation.unsqueeze(2) + e.unsqueeze(1)).chunk(2, dim=1)
        e = [e_i.transpose(1, 2) for e_i in e]  # [B, F, 1, C] * 2
        expand_rate = x.shape[1] // e[0].shape[1]
        assert x.shape[1] % e[0].shape[1] == 0
        # Same bug pattern as WanT2VCrossAttention / the FFN input: mul_add_add()
        # always returns float32, but self.head is an fp16/bf16-weighted Linear.
        # Cast explicitly before it.
        head_input = mul_add_add(
            self.norm(x).unflatten(1, (-1, expand_rate)), e[1], e[0]
        ).flatten(1, 2)
        head_input = head_input.to(self.head.weight.dtype)
        x = self.head(head_input)

        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = 1
        self.enable_teacache = False

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        with torch.device("cpu"):
            self.freqs = torch.cat(
                [
                    rope_params(2048, d - 4 * (d // 6)),
                    rope_params(2048, 2 * (d // 6)),
                    rope_params(2048, 2 * (d // 6)),
                ],
                dim=1,
            )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = False

        self.cpu_offloading = False

        # initialize weights
        self.init_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def zero_init_i2v_cross_attn(self):
        print("zero init i2v cross attn")
        for i in range(self.num_layers):
            self.blocks[i].cross_attn.v_img.weight.data.zero_()
            self.blocks[i].cross_attn.v_img.bias.data.zero_()

    def forward(
        self,
        x,
        t,
        context,
        clip_fea=None,
        y=None,
        block_mask=None,
        context_window_size=0,
        block_offload: bool = False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor or List[Tensor]):
                Input video tensor [C_in, F, H, W] or list of tensors for context support.
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            block_mask (Tensor, *optional*):
                Attention block mask
            context_window_size (int, *optional*):
                Window size for context support

        Returns:
            Tensor:
                Denoised video tensor with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        if block_offload:
            self.patch_embedding.to("mps")
            self.text_embedding.to("mps")
            self.time_embedding.to("mps")
            self.time_projection.to("mps")
            if hasattr(self, "img_emb"):
                self.img_emb.to("mps")
            self.freqs = self.freqs.to("mps")

            if isinstance(x, torch.Tensor):
                x = x.to("mps")
            elif isinstance(x, list):
                x = [item.to("mps") for item in x]

            t = t.to("mps")
            context = context.to("mps")
            if clip_fea is not None:
                clip_fea = clip_fea.to("mps")
            if y is not None:
                y = y.to("mps")
            if block_mask is not None:
                block_mask = block_mask.to("mps")

        # params
        device = self.patch_embedding.weight.device
        weight_dtype = self.patch_embedding.weight.dtype
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Ensure freqs is always on the correct device if it wasn't moved in block_offload
        if (
            not block_offload
            and self.freqs.device != x.device
            and isinstance(x, torch.Tensor)
        ):
            self.freqs = self.freqs.to(x.device)
        elif (
            not block_offload
            and isinstance(x, list)
            and len(x) > 0
            and self.freqs.device != x[0].device
        ):
            self.freqs = self.freqs.to(x[0].device)

        if isinstance(x, torch.Tensor):
            # Cast input to the patch_embedding weight's dtype (e.g. bf16)
            # before running the conv. Without this, MPS (which has no
            # implicit autocast for conv3d the way CUDA does) will throw
            # "Input type (float) and bias type (BFloat16) should be the same".
            x = x.to(weight_dtype)
            if y is not None:
                if y.device != x.device:
                    y = y.to(x.device)
                y = y.to(weight_dtype)
                x = torch.cat([x, y], dim=1)

            # embeddings
            x = self.patch_embedding(x)
            _dbg_sync("model: after patch_embedding")
            grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
            x = x.flatten(2).transpose(1, 2)

            grid_size_list = []
            num_frame_list = []
            num_token_list = []
        else:
            # list of tensors path (transformer_2)
            x = [item.to(weight_dtype) for item in x]
            x = [self.patch_embedding(item) for item in x]
            grid_size_list = [
                torch.tensor(item.shape[2:], dtype=torch.long) for item in x[1:]
            ]
            num_frame_list = [item.shape[2] for item in x[1:]]

            grid_sizes = torch.tensor(x[0].shape[2:], dtype=torch.long)
            grid_sizes[0] = grid_sizes[0] + sum(num_frame_list)

            x = [item.flatten(2).transpose(1, 2) for item in x]
            num_token_list = [item.shape[1] for item in x[1:]]

            x = torch.cat(x, dim=1)

        # time embeddings
        # with amp.autocast("cuda", dtype=torch.float32):
        b, f = t.shape

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(weight_dtype)
        )  # b, dim
        _dbg_sync("model: after time_embedding")
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim
        _dbg_sync("model: after time_projection")
        # On CUDA, a `torch.amp.autocast("cuda", dtype=torch.float32)` context
        # (now commented out below/above) used to force this computation up
        # to float32 regardless of the model's bf16 weights. MPS has no
        # equivalent implicit upcast here, so time_embedding/time_projection
        # now genuinely run in bf16 and produce bf16 output. Cast explicitly
        # instead of asserting float32, since downstream code (mul_add,
        # mul_add_add) already mixes precision safely via .float().
        e = e.view(b, f, -1).float()
        e0 = e0.view(b, f, 6, self.dim).float()

        # context
        context = self.text_embedding(context)
        _dbg_sync("model: after text_embedding")

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if block_offload:
            self.patch_embedding.to("cpu")
            self.text_embedding.to("cpu")
            self.time_embedding.to("cpu")
            self.time_projection.to("cpu")
            if hasattr(self, "img_emb"):
                self.img_emb.to("cpu")
            torch.mps.empty_cache()

        # arguments
        kwargs = dict(
            e=e0,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            grid_size_list=grid_size_list,
            num_frame_list=num_frame_list,
            num_token_list=num_token_list,
            context_window_size=context_window_size,
            block_mask=block_mask,
        )

        _dbg_sync("model: about to enter block loop")
        if block_offload:
            for i, block in enumerate(self.blocks):
                block.to("mps")

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = self._gradient_checkpointing_func(
                        block,
                        x,
                        **kwargs,
                    )
                else:
                    x = block(x, **kwargs)
                _dbg_sync(f"model: after block {i} (block_offload path)")

                block.to("cpu")
                torch.mps.empty_cache()
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, **kwargs)
                _dbg_sync(f"model: after block {i}")

        if block_offload:
            self.head.to("mps")

        x = self.head(x, e)
        _dbg_sync("model: after head")

        if block_offload:
            self.head.to("cpu")
            torch.mps.empty_cache()

        if len(num_token_list) > 0:
            num_context_token = sum(num_token_list)
            x = x[:, :-num_context_token]
            grid_sizes[0] = grid_sizes[0] - sum(num_frame_list)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        _dbg_sync("model: after unpatchify")
        return x.float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])

        return x

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
