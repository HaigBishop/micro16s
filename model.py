"""
This file contains the Micro16S model and related functions.

Contents:
 - Micro16S model (+ all methods for model)
 - embedding_triplet_loss
 - embedding_pair_loss
 - run_inference
 - load_micro16s_model
"""

# Imports
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import globals_config as gb
from logging_utils import write_triplet_loss_log, write_pair_loss_log, print_pair_loss_stats, print_triplet_loss_stats
from utils import synchronize_if_cuda




class MaskAwareConvModule(nn.Module):
    """
    Conv module designed for strict pad-invariance. For use in Convformer as well as an optional conv stem before the first conformer layer.

    Key idea:
    - Normalise per token (optional)
    - Before conv: force padded token vectors to EXACT zeros
      so padding is literally "absence of signal"
    - Use only constant zero padding for the convolution
    - Count-based renormalisation on the depthwise conv output so valid
      tokens are independent of how much padding is nearby
    - After conv: force padded outputs back to zero
    """

    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd so we can do 'same' padding cleanly")

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(dropout)
        self.use_norm = use_norm

        # Light, effective pattern: pointwise -> GLU -> depthwise conv -> pointwise
        # This is a common Conformer trick for local mixing.
        if self.use_norm:
            self.ln = nn.LayerNorm(d_model)
        else:
            self.ln = None

        self.pw_in = nn.Linear(d_model, 2 * d_model)   # for GLU gating
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,   # constant zero padding
            groups=d_model,             # depthwise
            bias=False                  # bias can create non-zero outputs on pure padding
        )
        self.pw_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None):
        """
        x: (B, L, C)
        key_padding_mask: (B, L) with True where padded, like your attention uses
        """
        # Normalise per token (safe, does not mix tokens)
        if self.use_norm:
            y = self.ln(x)
        else:
            y = x

        if key_padding_mask is not None:
            # Force padded token vectors to zero
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # Pointwise + GLU (gated local feature selection)
        y = self.pw_in(y)
        a, b = y.chunk(2, dim=-1)
        y = a * torch.sigmoid(b)

        # Re-zero before any token-mixing op
        if key_padding_mask is not None:
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # Depthwise conv expects (B, C, L)
        y = y.transpose(1, 2)

        # --- Count-based renormalisation for depthwise conv ---
        # Renormalise the depthwise conv output by the number of valid positions
        # covered by the kernel, so outputs for valid tokens are independent of
        # how much padding is nearby. When there is no padding in the window,
        # valid_count == K and scale == 1, so behaviour is unchanged.
        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).to(dtype=y.dtype)  # (B, L), 1=valid 0=pad
            valid_mask_3d = valid_mask.unsqueeze(1)              # (B, 1, L)
            # Ensure padded positions are exactly zero going into the depthwise conv
            y = y * valid_mask_3d
            # Count valid positions under each kernel window via cheap 1D conv
            K = self.kernel_size
            ones = torch.ones(1, 1, K, device=y.device, dtype=y.dtype)
            valid_count = F.conv1d(valid_mask_3d, ones, padding=K // 2)  # (B, 1, L)

        y = self.dw_conv(y)

        if key_padding_mask is not None:
            # Scale: K / clamp(valid_count, min=1) so fully-valid windows get scale=1
            scale = float(K) / valid_count.clamp(min=1.0)
            y = y * scale

        y = y.transpose(1, 2)

        # Pointwise projection back to model dim
        y = self.pw_out(y)
        y = self.dropout(y)

        if key_padding_mask is not None:
            # Keep padded outputs pinned to zero so they never leak into residual streams
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return y


class CustomMultiheadAttention(nn.Module):
    """
    Custom multi-head attention module with Rotary Position Embeddings (RoPE).
    
    RoPE encodes positions directly in the attention mechanism by rotating Q and K,
    enabling relative position awareness without additive positional embeddings.

    Built for self-attention only.
    
    Architecture:
        Q, K, V = linear_proj(input)
        Q, K = apply_rotary_embeddings(Q, K)
        attention = softmax(Q @ K.T / sqrt(d_k)) @ V
        output = linear_proj(attention)
    
    Args:
        embed_dim: Total dimension of the model (d_model)
        num_heads: Number of parallel attention heads
        attn_dropout: Dropout probability on attention weights (default: 0.0)
        batch_first: If True, input/output tensors are (batch, seq, feature)
        max_seq_len: Maximum sequence length for precomputing RoPE frequencies
        rope_base: Base for RoPE frequency computation (default: 10000.0)
    
    Note: Unlike nn.MultiheadAttention, this implementation:
        - Always uses batch_first=True internally for clarity
        - Does not support kdim/vdim different from embed_dim
        - Does not support add_bias_kv or add_zero_attn
        - These limitations are intentional for simplicity and our use case
    """
    
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, batch_first=True,
                 max_seq_len=None, rope_base=10000.0):
        super().__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = attn_dropout
        self.batch_first = batch_first
        
        # Validate head dimension is valid
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        # RoPE requires even head_dim (rotates pairs of dimensions)
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert max_seq_len is not None, "max_seq_len required for RoPE frequency precomputation"
        
        # Precompute RoPE cos/sin tables and register as buffers
        # Buffers are saved with model state and move with .to(device)
        rope_cos, rope_sin = self._precompute_rope_freqs(
            self.head_dim, max_seq_len, base=rope_base
        )
        self.register_buffer('rope_cos', rope_cos)
        self.register_buffer('rope_sin', rope_sin)
        
        # Scaling factor for dot-product attention (1/sqrt(d_k))
        # This prevents softmax from having extremely small gradients
        self.scale = self.head_dim ** -0.5
        
        # Combined Q, K, V projection for efficiency
        # Instead of 3 separate (embed_dim, embed_dim) matrices, we use one (3*embed_dim, embed_dim)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        
        # Output projection: maps concatenated heads back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout applied to attention weights (after softmax, before matmul with V)
        # This uses attn_dropout (ATT_DROPOUT_PROP) which is separate from the
        # general dropout (DROPOUT_PROP) used in residual/FFN paths.
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # Initialize parameters with Xavier uniform (matches PyTorch default)
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def _precompute_rope_freqs(self, head_dim, max_seq_len, base=10000.0):
        """
        Precompute cosine and sine tables for Rotary Position Embeddings.
        
        RoPE encodes position by rotating pairs of dimensions. Each dimension pair
        (2i, 2i+1) is rotated by an angle θ = position * freq_i, where higher
        dimension indices get lower frequencies (longer wavelengths).
        
        This enables the model to learn relative positions: the dot product
        between two rotated vectors depends on the difference of their positions.
        
        Args:
            head_dim: Dimension of each attention head (must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for inverse frequency computation (default: 10000.0)
        
        Returns:
            rope_cos: (max_seq_len, head_dim) cosine values
            rope_sin: (max_seq_len, head_dim) sine values
        """
        # Compute inverse frequencies for each dimension pair
        # inv_freq[i] = 1 / (base^(2i/head_dim)) for i in [0, head_dim/2)
        # Higher dimensions get smaller inv_freq -> lower frequency -> longer wavelength
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).float()
        
        # Compute angles: θ[pos, i] = pos * inv_freq[i]
        # Shape: (max_seq_len, head_dim/2)
        angles = torch.outer(positions, inv_freq)
        
        # Repeat each angle for the pair of dimensions it applies to
        # [θ_0, θ_1, ...] -> [θ_0, θ_0, θ_1, θ_1, ...]
        # Shape: (max_seq_len, head_dim)
        angles = angles.repeat_interleave(2, dim=-1)
        
        return torch.cos(angles), torch.sin(angles)
    
    def _apply_rope(self, x):
        """
        Apply Rotary Position Embeddings to input tensor.
        
        For each position and each pair of dimensions (2i, 2i+1), applies:
            [x_2i']     = [cos(θ)  -sin(θ)] [x_2i  ]
            [x_2i+1']   = [sin(θ)   cos(θ)] [x_2i+1]
        
        This 2D rotation preserves vector magnitude while encoding position.
        
        Args:
            x: (batch, n_heads, seq_len, head_dim) tensor to rotate
        
        Returns:
            Rotated tensor of same shape with positions encoded
        """
        seq_len = x.shape[2]
        
        # Slice precomputed tables to actual sequence length
        # Shape: (seq_len, head_dim)
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        
        # Reshape for broadcasting: (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Split into even and odd dimension indices (the pairs to rotate)
        # Shape: (batch, heads, seq, head_dim/2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Get cos/sin for each dimension pair
        # Shape: (1, 1, seq, head_dim/2)
        cos = cos[..., 0::2]
        sin = sin[..., 0::2]
        
        # Apply 2D rotation matrix to each pair
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        
        # Interleave rotated values back together
        # Stack: (batch, heads, seq, head_dim/2, 2)
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        
        # Flatten last two dims: (batch, heads, seq, head_dim)
        x_rot = x_rot.flatten(-2)
        
        return x_rot
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        """
        Forward pass for multi-head attention.

        This implementation uses PyTorch SDPA (scaled_dot_product_attention) which is typically
        backed by FlashAttention or a memory-efficient kernel on modern NVIDIA GPUs.

        For self-attention, query == key == value.

        Args:
            query: Query tensor, shape (batch, seq_len, embed_dim) if batch_first
            key: Key tensor (same shape as query for self-attention)
            value: Value tensor (same shape as query for self-attention)
            key_padding_mask: Boolean mask where True = ignore position
                Shape: (batch, key_seq_len)
            need_weights: If True, return attention weights (ignored, for API compat)
            attn_mask:
                - If boolean: True = mask out attention
                - If float: additive mask applied to attention logits (eg 0 for keep, -inf for mask)
                Shape must be broadcastable to (batch, heads, query_len, key_len)

        Returns:
            output: Attention output, shape (batch, seq_len, embed_dim)
            attn_weights: None (we do not return weights to save memory)
        """
        # ---- Step 0: Convert to batch-first format if needed ----
        if not self.batch_first:
            query = query.transpose(0, 1)
            key   = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Get dimensions
        batch_size, q_len, _ = query.shape
        _, k_len, _ = key.shape

        # ---- Step 1: Project Q, K, V ----
        # Combined projection: (batch, seq, embed) -> (batch, seq, 3*embed)
        # Note: this matches your original behaviour for self-attention
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)

        # Split into Q, K, V: each (batch, seq, embed)
        q, k, v = qkv.chunk(3, dim=-1)

        # ---- Step 2: Reshape for multi-head attention ----
        # (batch, seq, embed) -> (batch, heads, seq, head_dim)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- Step 2.5: Apply RoPE to Q and K ----
        # RoPE encodes positions by rotating Q and K vectors, enabling relative position awareness
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # ---- Step 3: Build a single SDPA-compatible mask ----
        # SDPA takes ONE mask tensor:
        #   - boolean mask: True = masked out
        #   - float mask: additive logits mask (eg 0 for keep, -inf for masked)
        #
        # We combine:
        #   A) key_padding_mask: (batch, key_len) True = ignore
        #   B) attn_mask: optional, boolean or float, broadcastable to (batch, heads, q_len, k_len)

        sdp_mask = attn_mask

        # Key padding mask applies to keys, so shape needs to broadcast onto (..., k_len)
        if key_padding_mask is not None:
            # Expand: (batch, key_len) -> (batch, 1, 1, key_len)
            pad_mask_bool = key_padding_mask[:, None, None, :]

            if sdp_mask is None:
                # If nothing else exists, we can use the boolean padding mask directly
                sdp_mask = pad_mask_bool
            else:
                # Combine depending on mask type
                if sdp_mask.dtype == torch.bool:
                    # Both boolean: True masks out attention
                    sdp_mask = sdp_mask | pad_mask_bool
                else:
                    # Additive mask: add -inf where padding is True
                    neg_inf = torch.tensor(float('-inf'), device=sdp_mask.device, dtype=sdp_mask.dtype)
                    pad_mask_add = torch.zeros_like(pad_mask_bool, dtype=sdp_mask.dtype)
                    pad_mask_add = pad_mask_add.masked_fill(pad_mask_bool, neg_inf)
                    if torch.isnan(pad_mask_add).any().item(): raise RuntimeError("NaNs in pad_mask_add (pad_mask * -inf)") # Note: This is a temporary checkfor in development
                    sdp_mask = sdp_mask + pad_mask_add
                    if torch.isnan(sdp_mask).any().item(): raise RuntimeError("NaNs in sdp_mask after adding pad mask") # Note: This is a temporary checkfor in development

        # ---- Step 4: SDPA attention (fast and memory-efficient) ----
        # SDPA internally applies scaling by 1/sqrt(head_dim), matching your old self.scale
        # if self.scale == 1/sqrt(head_dim) as per standard attention.
        dropout_p = self.attn_dropout.p if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdp_mask,
            dropout_p=dropout_p,
            is_causal=False
        )

        # ---- Step 5: Reshape and project output ----
        # (batch, heads, seq, head_dim) -> (batch, seq, embed)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)

        # Final output projection
        output = self.out_proj(attn_output)

        # ---- Step 6: Optional cleanup of padded query positions ----
        # In self-attention, query_len == key_len, so we can safely zero padded rows.
        # This prevents padded tokens from carrying junk activations downstream.
        if key_padding_mask is not None and key_padding_mask.shape[1] == q_len:
            output = output.masked_fill(key_padding_mask[:, :, None], 0.0)

        # Convert back from batch-first if needed
        if not self.batch_first:
            output = output.transpose(0, 1)

        # Return output and None for weights (API compatibility)
        return output, None


class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer using RoPE and optional convolution (Convformer).
    
    Uses post-norm architecture (the PyTorch default) with CustomMultiheadAttention
    for RoPE-based positional encoding. Optionally includes a mask-aware depthwise
    convolution sublayer between attention and feedforward for local pattern mixing.
    
    Architecture (Post-Norm):
        x -> self_attn -> dropout -> add(residual) -> layernorm ->
        [-> conv -> dropout -> add(residual) -> layernorm ->]   (if use_conv)
        -> feedforward -> dropout -> add(residual) -> layernorm -> output
    
    Args:
        d_model: Dimension of the model (embed_dim)
        nhead: Number of attention heads
        dim_feedforward: Dimension of the feedforward hidden layer (default: 2048)
        dropout: Dropout probability for residual/FFN/conv paths (default: 0.1)
        attn_dropout: Dropout probability for attention weights only (default: 0.0)
        batch_first: If True, input is (batch, seq, feature)
        max_seq_len: Maximum sequence length for RoPE frequency precomputation
        rope_base: Base for RoPE frequency computation (default: 10000.0)
        use_conv: If True, enable the mask-aware convolution module (default: False)
        conv_kernel_size: Kernel size for the convolution module (default: 7)
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attn_dropout=0.0, batch_first=True,
                 max_seq_len=None, rope_base=10000.0, use_conv=False, conv_kernel_size=7):
        super().__init__()
        
        # Store configuration for potential introspection
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.use_conv = use_conv
        
        # ---- Self-attention sublayer (with RoPE) ----
        self.self_attn = CustomMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            attn_dropout=attn_dropout,
            batch_first=batch_first,
            max_seq_len=max_seq_len,
            rope_base=rope_base
        )

        # ---- Convolution sublayer (optional) ----
        # Mask-aware conv module for strict pad-invariance (no downsampling)
        # We skip internal LN here because this layer applies norm_conv after the residual add.
        if self.use_conv:
            self.conv = MaskAwareConvModule(d_model=d_model, kernel_size=conv_kernel_size, dropout=dropout, use_norm=False)
            self.norm_conv = nn.LayerNorm(d_model)
        else:
            self.conv = None
            self.norm_conv = None
        
        # ---- Feedforward sublayer ----
        # Two linear layers with GELU activation in between
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # ---- Layer normalization ----
        # Post-norm: applied after residual connection
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ---- Dropout layers ----
        # Separate dropout instances for each sublayer (matches PyTorch)
        self.dropout = nn.Dropout(dropout)   # For feedforward activation
        self.dropout1 = nn.Dropout(dropout)  # After self-attention
        self.dropout2 = nn.Dropout(dropout)  # After feedforward
        
        # Activation function
        self.activation = F.gelu
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """
        Forward pass for encoder layer.
        
        Args:
            src: Input tensor, shape (batch, seq_len, d_model) if batch_first
            src_mask: Attention mask for the sequence (rarely used)
            src_key_padding_mask: Boolean mask where True = padding
                Shape: (batch, seq_len)
            is_causal: Ignored, included for API compatibility
        
        Returns:
            Output tensor of same shape as input
        """
        # ---- Self-attention sublayer with residual ----
        # Compute self-attention
        attn_output, _ = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask
        )
        # Residual connection + dropout + layer norm
        src = self.norm1(src + self.dropout1(attn_output))
        
        # ---- Convolution sublayer with residual (optional) ----
        # Apply mask-aware convolution after attention
        if self.use_conv:
            conv_output = self.conv(src, src_key_padding_mask)
            src = self.norm_conv(src + conv_output)
        
        # ---- Feedforward sublayer with residual ----
        # Two-layer MLP with GELU activation
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Residual connection + dropout + layer norm
        src = self.norm2(src + self.dropout2(ff_output))
        
        return src


class CustomTransformerEncoder(nn.Module):
    """
    Custom transformer encoder that mirrors nn.TransformerEncoder.
    
    Stacks multiple CustomTransformerEncoderLayer instances. Each layer
    is an independent copy with its own parameters.
    
    Args:
        encoder_layer: An instance of CustomTransformerEncoderLayer (used as template)
        num_layers: Number of encoder layers to stack
        norm: Optional final layer normalization (not used by default in our model)
    """
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        
        # Create independent copies of the encoder layer
        # deepcopy ensures each layer has its own parameters
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
        # Optional final normalization (PyTorch allows this but we don't use it)
        self.norm = norm
    
    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        """
        Forward pass through all encoder layers.
        
        Args:
            src: Input tensor, shape (batch, seq_len, d_model)
            mask: Attention mask (passed to each layer)
            src_key_padding_mask: Padding mask (passed to each layer)
            is_causal: Ignored, for API compatibility
        
        Returns:
            Output tensor after all encoder layers
        """
        output = src
        
        # Pass through each encoder layer sequentially
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )
        
        # Apply final normalization if provided
        if self.norm is not None:
            output = self.norm(output)
        
        return output



class Micro16S(nn.Module):
    """
    A Micro16S Transformer (FMR) model.
    
    This model uses a transformer encoder with RoPE to act directly on DNA sequences.

    1.  The model natively takes 3-bit representations of 16S sequences.
    2.  It uses an embedding lookup to project nucleotide tokens to d_model.
    3.  An optional conv stem applies local mixing before the first attention layer.
    4.  A stack of custom Transformer Encoder layers (with RoPE) applies self-attention.
    5.  Sequence pooling (mean or attention) reduces the output to a single vector.
    6.  A final feed-forward network produces the embedding.
    7.  Embeddings are L2-normalised before being returned.
    
    The model handles inputs as:
     - 3-bit representations of DNA (native)
     - lists of DNA base strings
    """

    def __init__(self, embed_dims, max_seq_len, d_model, n_layers, n_head, d_ff, 
                 seq_3bit_representation_dict, name=None, id=None, pooling_type='mean', use_convformer=False, conformer_kernel_size=7,
                 use_conv_stem=False, conv_stem_kernel_size=7, conv_stem_residual=True, conv_stem_init_scale=0.05,
                 dropout=0.0, attn_dropout=0.0):
       super(Micro16S, self).__init__()

       self.model_type = 'FMR'

       # Model identity
       self.name = name if name is not None else f"Micro16S_{id if id else int(time.time())}"
       self.id = id if id is not None else int(time.time())

       # Architecture choices
       self.embed_dims = embed_dims
       self.max_seq_len = max_seq_len
       self.d_model = d_model
       self.n_layers = n_layers
       self.n_head = n_head
       self.d_ff = d_ff
       self.use_conv_stem = use_conv_stem
       self.conv_stem_kernel_size = conv_stem_kernel_size
       self.conv_stem_residual = conv_stem_residual
       self.conv_stem_init_scale = conv_stem_init_scale
       self.use_convformer = use_convformer
       self.conformer_kernel_size = conformer_kernel_size
       self.dropout_p = dropout          # Global dropout for residual/FFN/conv paths
       self.attn_dropout_p = attn_dropout # Dropout for attention weights only
       if not isinstance(pooling_type, str):
           raise ValueError("pooling_type must be a string.")
       pooling_type = pooling_type.lower()
       if pooling_type not in ('mean', 'attention'):
           raise ValueError(f"Unsupported pooling_type: {pooling_type}. Must be 'mean' or 'attention'.")
       self.pooling_type = pooling_type

       # Save the sequence encoding dictionary
       self.seq_3bit_representation_dict = seq_3bit_representation_dict

       # Effective sequence length (direct embedding, no CNN)
       self.effective_seq_len = self.max_seq_len
       self.transformer_seq_len = self.effective_seq_len

       # Input layer
       # Embedding lookup: 5 tokens (4 nucleotides + 1 padding)
       # Token indices: A=0, C=1, G=2, T=3, PAD=4
       self.nucleotide_embedding = nn.Embedding(5, d_model, padding_idx=4)

       # Conv stem (optional local mixing before first attention layer)
       # Gives the model motif-level features before self-attention sees individual nucleotides.
       # We keep the internal LN here as it is the first operation on the raw embeddings.
       if self.use_conv_stem:
           self.conv_stem = MaskAwareConvModule(
               d_model=d_model, 
               kernel_size=conv_stem_kernel_size, 
               dropout=dropout,
               use_norm=True
           )
           if self.conv_stem_residual:
               self.conv_stem_scale = nn.Parameter(torch.tensor(conv_stem_init_scale))

       self.dropout = nn.Dropout(dropout)  # Pre-transformer token dropout

       if self.pooling_type == 'attention':
           self.attention_pool = nn.Linear(d_model, 1)
       else:
           self.attention_pool = None

       # Transformer Encoder (custom classes for RoPE + optional convformer)
       encoder_layer = CustomTransformerEncoderLayer(
           d_model=d_model,
           nhead=n_head,
           dim_feedforward=d_ff,
           dropout=dropout,
           attn_dropout=attn_dropout,
           batch_first=True,
           max_seq_len=self.transformer_seq_len,
           use_conv=self.use_convformer,
           conv_kernel_size=self.conformer_kernel_size
       )
       self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=n_layers)

       # Output FFN
       self.output_ffn = nn.Sequential(
           nn.Linear(d_model, d_model),
           nn.GELU(),
           nn.Linear(d_model, embed_dims)
       )
       
       self.initialize_parameters()

       # Print the effective sequence length
       print(f"Model initialised with effective sequence length: {self.effective_seq_len}")

    def forward(self, x):
        """
        Forward pass for Micro16S.

        Accepts 3-bit encoded tensors or lists of DNA strings and returns L2-normalized embeddings.
        Input shape: [..., max_seq_len, 3]
        Output shape: [..., embed_dims]
        """
        # --- Input Processing ---
        if isinstance(x, list):
            if not x: return torch.empty(0, self.embed_dims)
            if not all(self.is_valid_dna_string(seq) for seq in x):
                raise ValueError("Input list contains invalid DNA strings")
            x = self.encode_sequences_3bit(x)

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        
        if x.numel() == 0: return torch.empty(0, self.embed_dims, device=x.device)
        if x.dtype != torch.float32: x = x.to(torch.float32)

        original_shape = x.shape
        if len(original_shape) < 2 or original_shape[-1] != 3 or original_shape[-2] != self.max_seq_len:
            raise ValueError(f"Invalid input shape: {original_shape}. Expected [..., {self.max_seq_len}, 3]")

        # Flatten all but the last two dimensions
        prefix_dims = original_shape[:-2]
        n_seqs = int(np.prod(prefix_dims)) if prefix_dims else 1
        x = x.view(n_seqs, self.max_seq_len, 3)

        # Separate mask from nucleotide data
        padding_mask = x[:, :, 0].bool()  # (N, L), True where padded
        nucleotides = x[:, :, 1:]         # (N, L, 2)

        # --- Model Pass ---
        # 1. Input Layer
        # Convert 3-bit encoding to token indices for embedding lookup
        # Token indices: A=0 (bits 00), C=1 (bits 01), G=2 (bits 10), T=3 (bits 11), PAD=4
        # Formula: base_idx = bit1*2 + bit2, then add 4 if padding
        base_idx = (nucleotides[:, :, 0] * 2 + nucleotides[:, :, 1]).long()  # (N, L)
        token_idx = base_idx.clone()
        token_idx[padding_mask] = 4  # Padded positions get index 4 (PAD) regardless of the other bits
        
        # Direct embedding lookup - PAD embedding already included at index 4
        nuc_embeds = self.nucleotide_embedding(token_idx)  # (N, L, d_model)
        effective_padding_mask = padding_mask
        
        # 2. Conv stem (optional local mixing before first attention)
        if self.use_conv_stem:
            if self.conv_stem_residual:
                x = nuc_embeds + self.conv_stem_scale * self.conv_stem(nuc_embeds, effective_padding_mask)
            else:
                x = self.conv_stem(nuc_embeds, effective_padding_mask)
        else:
            x = nuc_embeds

        # 3. Prepare for Transformer
        # Note: RoPE position info is applied inside CustomMultiheadAttention
        x = self.dropout(x)

        # 4. Transformer Encoder
        # The transformer layer expects the padding mask where True indicates a value to be ignored.
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=effective_padding_mask)

        # 5. Sequence Pooling
        valid_positions = ~effective_padding_mask  # (N, L)
        if self.pooling_type == 'attention':
            attn_logits = self.attention_pool(encoder_output).squeeze(-1)  # (N, L)
            attn_logits = attn_logits.masked_fill(~valid_positions, float('-inf'))
            no_valid = ~valid_positions.any(dim=1)
            if no_valid.any():
                attn_logits = attn_logits.clone()
                attn_logits[no_valid] = 0.0
            attn_weights = torch.softmax(attn_logits, dim=1)
            attn_weights = attn_weights.masked_fill(~valid_positions, 0.0)
            weight_sums = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            attn_weights = attn_weights / weight_sums
            pooled = torch.sum(encoder_output * attn_weights.unsqueeze(-1), dim=1)
        else:
            # Mean pooling
            mask_for_pooling = valid_positions.unsqueeze(-1)
            masked_output = encoder_output * mask_for_pooling
            summed = masked_output.sum(dim=1)
            count = mask_for_pooling.sum(dim=1).clamp(min=1e-8)
            pooled = summed / count

        # 6. Output FFN
        x = self.output_ffn(pooled)

        # L2-normalize the output (in fp32 for numerical cleanliness under autocast)
        x = F.normalize(x.float(), p=2, dim=-1)

        # Reshape back to original prefix dimensions
        output_shape = list(prefix_dims) + [self.embed_dims]
        return x.view(*output_shape)

    def initialize_parameters(self):
        """Initialise weights with Xavier initialization and zero out biases."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Weight init
                nn.init.xavier_normal_(module.weight)

                # Bias init
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

                # Set the PAD Embedding to zero
                if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)

        return self

    def get_parameter_groups(self, weight_decay):
        """
        Build optimizer parameter groups with and without weight decay.

        Weight decay is applied to:
            - Embedding table weights (nn.Embedding)
            - All linear layer weights (QKV projections, output projections, FFN)
            - Conv weights (depthwise conv included)
            - Pooling projection weights (attention_pool)
            - Output FFN weights

        Weight decay is NOT applied to:
            - All biases (Linear biases, in-proj bias, etc.)
            - LayerNorm parameters (weight and bias)
            - Scalar gating parameters (conv_stem_scale)

        Args:
            weight_decay: Weight decay coefficient for the decay group.

        Returns:
            List of two dicts suitable for torch.optim parameter groups.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Biases, LayerNorm params, and scalar gating params skip weight decay
            if name.endswith('bias') or 'norm' in name or '.ln.' in name or name == 'conv_stem_scale':
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Sanity check: every trainable parameter should be in exactly one group
        n_total = sum(1 for p in self.parameters() if p.requires_grad)
        assert len(decay_params) + len(no_decay_params) == n_total, (
            f"Parameter group mismatch: {len(decay_params)} decay + {len(no_decay_params)} no-decay != {n_total} total"
        )

        return [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

    @staticmethod
    def is_valid_dna_string(seq):
        """Check if a string is a valid DNA string."""
        return len(seq) != 0 and all(base in 'ACGT' for base in seq)

    def encode_base(self, base):
        """Convert a DNA base to its 3-bit representation."""
        base = base.upper()
        encoding_map = {'A': [0,0,0], 'C': [0,0,1], 'G': [0,1,0], 'T': [0,1,1]}
        return encoding_map.get(base, [1,0,0]) # Default to padding for any other char

    def encode_sequence(self, seq, max_length):
        """Convert a sequence (string) to its 3-bit representation with padding."""
        seq = seq[:max_length]
        encoding = np.zeros((max_length, 3), dtype=np.float32)
        for i, base in enumerate(seq):
            encoding[i] = self.encode_base(base)
        if len(seq) < max_length:
            encoding[len(seq):, 0] = 1 # Mask remaining positions
        return encoding
    
    def encode_sequences_3bit(self, sequences):
        """Encode a list of sequences to their 3-bit representations with padding."""
        encodings = [self.encode_sequence(seq, self.max_seq_len) for seq in sequences]
        return torch.tensor(np.array(encodings), dtype=torch.float32)

    def get_config(self):
        """Return configuration of the model."""
        return {
            'model_type': self.model_type,
            'name': self.name,
            'id': self.id,
            'embed_dims': self.embed_dims,
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_head': self.n_head,
            'd_ff': self.d_ff,
            'seq_3bit_representation_dict': self.seq_3bit_representation_dict,
            'pooling_type': self.pooling_type,
            'use_conv_stem': self.use_conv_stem,
            'conv_stem_kernel_size': self.conv_stem_kernel_size,
            'conv_stem_residual': self.conv_stem_residual,
            'conv_stem_init_scale': self.conv_stem_init_scale,
            'use_convformer': self.use_convformer,
            'conformer_kernel_size': self.conformer_kernel_size,
            'dropout': self.dropout_p,
            'attn_dropout': self.attn_dropout_p,
        }

    def save_model(self, path):
        """Save the model state and attributes to the specified path."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_attributes': self.get_config()
        }, path)

    @classmethod
    def load_model(cls, path, checkpoint=None):
        """Load a Micro16S model from a saved checkpoint."""
        if checkpoint is None:
            checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        attributes = checkpoint['model_attributes']
        
        model = cls(
            embed_dims=attributes['embed_dims'],
            max_seq_len=attributes['max_seq_len'],
            d_model=attributes['d_model'],
            n_layers=attributes['n_layers'],
            n_head=attributes['n_head'],
            d_ff=attributes['d_ff'],
            seq_3bit_representation_dict=attributes['seq_3bit_representation_dict'],
            name=attributes['name'],
            id=attributes['id'],
            pooling_type=attributes.get('pooling_type', 'mean'),
            use_convformer=attributes.get('use_convformer', False),
            conformer_kernel_size=attributes.get('conformer_kernel_size', 7),
            use_conv_stem=attributes.get('use_conv_stem', False),
            conv_stem_kernel_size=attributes.get('conv_stem_kernel_size', 7),
            conv_stem_residual=attributes.get('conv_stem_residual', True),
            conv_stem_init_scale=attributes.get('conv_stem_init_scale', 0.05),
            dropout=attributes.get('dropout', 0.0),
            attn_dropout=attributes.get('attn_dropout', 0.0),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model




class UncertaintyLoss(nn.Module):
    """
    Uncertainty Weighting module for multi-task learning with pair and triplet losses.
    
    Learns task-specific log-variance parameters to dynamically balance loss magnitudes.
    This prevents one loss (e.g., hard-mined triplet loss) from dominating gradients.
    
    Formula per task: weighted_loss = 0.5 * exp(-s) * loss + 0.5 * s
    where s = log(σ²) is the learned log-variance.
    
    Total loss is the sum of weighted losses for triplet and pair tasks.
    
    Note: This module should NOT be part of the main Micro16S model, as these
    parameters are only needed during training, not inference/deployment.
    These parameters typically benefit from a separate (often higher) learning 
    rate than the main model weights.
    """
    
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        # log_vars[0] = s_triplet = log(σ²_triplet)
        # log_vars[1] = s_pair = log(σ²_pair)
        # Initialized to zeros -> σ² = 1 -> equal weighting at start
        self.log_vars = nn.Parameter(torch.zeros(2))
    
    def forward(self, triplet_loss, pair_loss):
        """
        Compute uncertainty-weighted combined loss.
        
        Args:
            triplet_loss: Scalar tensor, mean triplet loss (already weighted by TRIPLET_LOSS_WEIGHT)
            pair_loss: Scalar tensor, mean pair loss (already weighted by PAIR_LOSS_WEIGHT)
        
        Returns:
            total_loss: Scalar tensor, uncertainty-weighted sum of both losses
        """
        s_triplet = self.log_vars[0]
        s_pair = self.log_vars[1]
        
        # Weighted loss formula: 0.5 * exp(-s) * loss + 0.5 * s
        weighted_triplet = 0.5 * torch.exp(-s_triplet) * triplet_loss + 0.5 * s_triplet
        weighted_pair = 0.5 * torch.exp(-s_pair) * pair_loss + 0.5 * s_pair
        
        return weighted_triplet, weighted_pair


def embedding_triplet_loss(embeddings, margins, ranks, buckets=None, normalize_embeddings=True, verbose=False, log=False, batch_num=None, logs_dir=None, return_dists=False):
    """
    Compute the triplet loss for a set of embeddings using cosine similarity.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape [batch_size, 3, embed_dims] containing anchor, 
            positive, and negative embeddings. Each triplet consists of:
            - embeddings[:,0,:] = anchor embeddings
            - embeddings[:,1,:] = positive embeddings (same taxon as anchor at the triplet rank)
            - embeddings[:,2,:] = negative embeddings (different taxon from anchor at the triplet rank)
        margins (torch.Tensor): Tensor of shape [batch_size] containing the margin values
            for each triplet. The margin defines how much more similar positive samples 
            should be compared to negative samples.
        ranks (torch.Tensor): Tensor of shape [batch_size] containing the rank of each triplet.
        buckets (torch.Tensor): Optional tensor of bucket assignments for each triplet.
        verbose (bool): If True, prints statistics about triplet distances.
        log (bool): If True, writes statistics to log and CSV files.
        batch_num (int): Batch number for logging (required if log=True).
        logs_dir (str): Directory for log files (required if log=True).
        normalize_embeddings (bool): If True (default), L2-normalises the embeddings before
            computing cosine distances. Set to False when upstream code already returns
            normalised embeddings to avoid redundant work.
        return_dists (bool): If True, return a tuple (losses, a_p_dist, a_n_dist) so the
            caller can aggregate distances across microbatches for full-batch logging.
    
    Returns:
        torch.Tensor: Tensor of losses for each triplet, shape [batch_size].
            If return_dists=True, returns (losses, a_p_dist, a_n_dist) instead.
        
    The triplet loss encourages:
        - anchor-positive similarity to be larger than anchor-negative similarity
        - anchor-positive similarity to be at least margin larger than anchor-negative
        
    We also scale each active hinge loss by approximately 1/(true_an - true_ap) so that
    tighter ranks carry proportionally larger gradients. With dynamic margins we have
    margin = (true_an - true_ap) * TRIPLET_MARGIN_EPSILON, so
    TRIPLET_MARGIN_EPSILON / (margin + eps) ≈ 1 / (true_an - true_ap). The same scaling
    works for manual margins and lets their magnitudes dictate per-rank emphasis.
    """
    # If there are no triplets, return an empty tensor on the correct device
    if len(embeddings) == 0:
        empty = embeddings.new_zeros((0,))
        return (empty, empty, empty) if return_dists else empty
    
    # 1. Ensure embeddings are L2-normalized (skip if caller already normalised)
    if normalize_embeddings:
        epsilon = 1e-8
        embeddings = F.normalize(embeddings, p=2, dim=2, eps=epsilon)

    # 2. Compute cosine similarities
    a_p_sim = (embeddings[:, 0] * embeddings[:, 1]).sum(dim=1)
    a_n_sim = (embeddings[:, 0] * embeddings[:, 2]).sum(dim=1)

    # 3. Compute cosine distances
    a_p_dist = 1 - a_p_sim
    a_n_dist = 1 - a_n_sim

    # 4. Compute triplet loss
    losses = F.relu(a_p_dist - a_n_dist + margins)
    
    # 5. Compute loss scaling by the approximate inverse of the true distance gap
    # Purpose: So high-rank triplets (small delta_true) stay influential. 

    # Use margin + per-rank epsilon in the denominator for stable scaling.
    epsilons_tensor = torch.tensor(gb.RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS, device=margins.device, dtype=margins.dtype)
    triplet_epsilons = epsilons_tensor[ranks.long()]
    margin_denominators = margins + triplet_epsilons

    # Compute scale depending on margins
    if gb.MANUAL_TRIPLET_MARGINS:
        scale = gb.TRIPLET_MARGIN_EPSILON / margin_denominators
    else:
        approx_delta_true = margin_denominators / gb.TRIPLET_MARGIN_EPSILON
        scale = 1.0 / approx_delta_true
    
    # 6. Scale losses
    scaled_losses = losses * scale

    # 7. Cap losses
    if gb.TRIPLET_RELATIVE_LOSS_CAP is not None:
        scaled_losses = torch.clamp(scaled_losses, max=gb.TRIPLET_RELATIVE_LOSS_CAP)

    # Verbose printing
    if verbose:
        print_triplet_loss_stats(a_p_dist, a_n_dist, margins, scaled_losses, ranks, buckets)

    # Logging
    if log and batch_num is not None and logs_dir is not None:
        write_triplet_loss_log(batch_num, logs_dir, a_p_dist, a_n_dist, margins, scaled_losses, ranks, buckets)
    
    if return_dists:
        return scaled_losses, a_p_dist.detach(), a_n_dist.detach()
    return scaled_losses


def embedding_pair_loss(embeddings, true_distances, ranks, buckets=None, region_pairs=None, normalize_embeddings=True, verbose=False, log=False, batch_num=None, logs_dir=None, return_dists=False):
    """
    Compute the pair loss for a set of embeddings using cosine similarity.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape [batch_size, 2, embed_dims] containing
            pairs of embeddings to compare.
        true_distances (torch.Tensor): Tensor of shape [batch_size] containing the target
            cosine distances between each pair of embeddings. 
        ranks (torch.Tensor): Tensor of shape [batch_size] containing the rank of each pair.
        buckets (torch.Tensor): Optional tensor of bucket assignments for each pair.
        region_pairs (torch.Tensor or np.ndarray, optional): Region indices for each side of the pair (shape [batch_size, 2]).
        verbose (bool): If True, prints statistics about pair losses.
        log (bool): If True, writes statistics to log and CSV files.
        batch_num (int): Batch number for logging (required if log=True).
        logs_dir (str): Directory for log files (required if log=True).
        normalize_embeddings (bool): If True (default), L2-normalises embeddings before computing
            cosine similarity. Disable when embeddings already live on the unit hypersphere.
        return_dists (bool): If True, return a tuple (losses, pred_dists) so the caller can
            aggregate predicted distances across microbatches for full-batch logging.
    
    Returns:
        torch.Tensor: Tensor of losses for each pair, shape [batch_size].
            If return_dists=True, returns (losses, pred_dists) instead.
        
    The pair loss encourages:
        - The cosine similarity between embeddings to match the target similarity
    """
    # If there are no pairs, return an empty tensor on the correct device
    if len(embeddings) == 0:
        empty = embeddings.new_zeros((0,))
        return (empty, empty) if return_dists else empty

    # Validate optional metadata
    if region_pairs is not None and len(region_pairs) != embeddings.shape[0]:
        raise ValueError("region_pairs must match the number of pairs provided to embedding_pair_loss.")

    # 1. L2-normalise embeddings so they sit on the hypersphere (skip if already normalised)
    if normalize_embeddings:
        epsilon = 1e-8
        embeddings = F.normalize(embeddings, p=2, dim=2, eps=epsilon)

    # 2. Compute pairwise cosine similarity
    # shape: [batch_size]
    cosine_sims = (embeddings[:, 0] * embeddings[:, 1]).sum(dim=1)

    # 3. Convert to Cosine Distance (0 to 2 range)
    pred_dists = 1 - cosine_sims
    
    # 4. Calculate Squared Error
    squared_errors = (pred_dists - true_distances) ** 2

    # 5. Calculate denominator using true distance + per-rank epsilon
    # loss = ((pred - true)^2) / (true + eps)
    epsilons_tensor = torch.tensor(gb.RELATIVE_ERROR_EPSILONS_PAIR_LOSS, device=true_distances.device, dtype=true_distances.dtype)
    pair_epsilons = epsilons_tensor[ranks.long()]
    denom = true_distances + pair_epsilons

    # 6. Calculate relative squared losses
    relative_losses = squared_errors / denom
    
    # 7. Cap losses
    if gb.PAIR_RELATIVE_LOSS_CAP is not None:
        relative_losses = torch.clamp(relative_losses, max=gb.PAIR_RELATIVE_LOSS_CAP)

    # Verbose printing
    if verbose:
        print_pair_loss_stats(relative_losses, pred_dists, true_distances, ranks, buckets, region_pairs=region_pairs)
    
    # Logging
    if log and batch_num is not None and logs_dir is not None:
        write_pair_loss_log(batch_num, logs_dir, relative_losses, pred_dists, true_distances, ranks, buckets, region_pairs=region_pairs)
    
    if return_dists:
        return relative_losses, pred_dists.detach()
    return relative_losses


def run_inference(model, seq_reps_3bit_, device=None, batch_size=1e4, output_device='cpu', return_numpy=True, pin_inputs=False):
    """
    Batched inference for a Micro16S model.

    seq_reps_3bit_ shape : [..., seq_len, 3]
        (any number of prefix dims, last two dims are the sequence encoding)

    Args:
        model (Micro16S): Model used for inference.
        seq_reps_3bit_ (array-like): Inputs with shape [..., seq_len, 3].
        device (torch.device, optional): Device used to run the model. Defaults to the model's current device.
        batch_size (int): Number of sequences per batch.
        output_device (torch.device or str or None): Device that the embeddings should live on. 
            Set to None to keep them on the model's device.
        return_numpy (bool): Whether to convert the final embeddings to numpy. If True, embeddings are moved to CPU.
        pin_inputs (bool): Whether to pin CPU memory before transferring to the GPU. Enables async H2D copies.

    Returns
    -------
    torch.Tensor or np.ndarray with shape prefix_dims + (embed_dims,)
    """
    
    # Ensure batch_size is a usable int
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # Convert to torch tensor and pin memory if needed
    seq_reps_3bit = torch.as_tensor(seq_reps_3bit_)
    if pin_inputs and seq_reps_3bit.device.type == 'cpu' and not seq_reps_3bit.is_pinned():
        seq_reps_3bit = seq_reps_3bit.pin_memory()

    # Flatten every dimension except the last two (seq_len, bits)
    prefix_shape = seq_reps_3bit.shape[:-2]
    seq_len, n_bits = seq_reps_3bit.shape[-2], seq_reps_3bit.shape[-1]
    n_samples = int(np.prod(prefix_shape))
    seq_reps_3bit_flat = seq_reps_3bit.reshape(n_samples, seq_len, n_bits)

    # If the device is not specified, automatically get it
    if device is None:
        device = next(model.parameters()).device
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # If there is a output device specified, ensure it is a torch device
    if output_device is not None and not isinstance(output_device, torch.device):
        output_device = torch.device(output_device)

    # Short-circuit when there are no samples to process
    if n_samples == 0:
        empty_shape = tuple(prefix_shape) + (model.embed_dims,)
        target_device = output_device if output_device is not None else device
        empty = torch.empty(empty_shape, dtype=torch.float32, device=target_device)
        if return_numpy:
            return empty.cpu().numpy()
        return empty
    
    # Determine whether to use autocast (only on CUDA)
    use_autocast = device.type == 'cuda'
        
    # Prepare for inference
    outputs = []
    model.eval()
    
    # Use inference_mode if available (PyTorch 1.9+), fallback to no_grad
    inference_context = torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()
    
    with inference_context:

        # Loop over batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            # Get the batch
            batch = seq_reps_3bit_flat[start:end]

            # Ensure the batch is on the correct device and has the correct dtype
            if batch.device != device:
                non_blocking = pin_inputs and batch.device.type == 'cpu' and device.type == 'cuda'
                batch = batch.to(device, dtype=torch.float32, non_blocking=non_blocking)
            elif batch.dtype != torch.float32:
                batch = batch.to(dtype=torch.float32)

            # Run inference with autocast on CUDA for efficiency
            if use_autocast:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(batch)
            else:
                out = model(batch)
            
            # Synchronize to ensure GPU operations are complete before detaching
            # This is critical for single-batch inference and accurate timing
            synchronize_if_cuda(out)
            
            # Detach the output and ensure float32 dtype (autocast may produce float16)
            out = out.detach()
            if out.dtype != torch.float32:
                out = out.to(dtype=torch.float32)

            # Ensure the output is on the correct device
            if output_device is not None and out.device != output_device:
                # Note: We do not use non_blocking=True for D2H (GPU->CPU) transfers
                out = out.to(output_device)
            
            # Collect outputs across batches
            outputs.append(out)

    # Recombine outputs across batches
    embeddings_flat = torch.cat(outputs, dim=0)
    embeddings = embeddings_flat.view(*prefix_shape, -1)

    # Optionally return as numpy
    if return_numpy:
        return embeddings.cpu().numpy()

    # Return as torch tensor float32
    return embeddings


def load_micro16s_model(path):
    """Load a Micro16S model from a saved checkpoint."""
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return Micro16S.load_model(path, checkpoint=checkpoint)
    


class LinearWarmupToLearningRate(torch.optim.lr_scheduler._LRScheduler):
    """
    Linearly warm each param-group LR from a shared starting LR to its configured target LR.

    Notes:
    - start_lr is applied at scheduler initialization, so batch 1 runs at start_lr.
    - After warmup_batches scheduler steps, learning rates have reached their target values.
    """
    def __init__(self, optimizer, start_lr, warmup_batches, last_epoch=-1):
        if not isinstance(warmup_batches, int) or warmup_batches < 1:
            raise ValueError(f"LinearWarmupToLearningRate requires warmup_batches as an integer >= 1, got: {warmup_batches}")
        if not isinstance(start_lr, (int, float)):
            raise ValueError(f"LinearWarmupToLearningRate requires start_lr as a number, got: {type(start_lr).__name__}")
        if start_lr < 0:
            raise ValueError(f"LinearWarmupToLearningRate requires start_lr >= 0, got: {start_lr}")

        self.start_lr = float(start_lr)
        self.warmup_batches = warmup_batches
        self.target_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # last_epoch is the number of scheduler steps already taken.
        # At step 0: use start_lr; at step warmup_batches: use target LR(s).
        if self.last_epoch >= self.warmup_batches:
            return self.target_lrs

        progress = self.last_epoch / self.warmup_batches
        return [self.start_lr + (target_lr - self.start_lr) * progress for target_lr in self.target_lrs]
