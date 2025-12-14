
import torch
import torch.nn as nn  # Neural network modules like Linear, Embedding, etc.
import torch.nn.functional as F  # Functional interface for operations like cross_entropy, silu, etc.
import math
from Config import ModelConfig

def precompute_freqs_cis(config: ModelConfig) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = config.qk_rope_head_dim
    seqlen = config.max_seq_len
    beta_fast = config.beta_fast
    beta_slow = config.beta_slow
    base = config.rope_theta
    factor = config.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > config.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, config.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self, cache, config: ModelConfig):
        super().__init__()
        self.dim = config.d_model
        self.n_heads = config.n_heads
        self.n_local_heads = config.n_local_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.rope_factor = config.rope_factor
        self.mscale = config.mscale
        self.original_seq_len = config.original_seq_len
        self.cache = cache

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
            self.q_norm = nn.RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)

        self.softmax_scale = self.qk_head_dim ** -0.5

        if config.max_seq_len > self.original_seq_len:
            mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if self.cache:
            # Use max_batch_size instead of batch_size for cache
            self.register_buffer("kv_cache", torch.zeros(config.batch_size, config.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(config.batch_size, config.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x, freqs_cis: torch.Tensor, start_pos=0, mask=None):
        bsz, seq_len, _ = x.size()
        end_pos = start_pos + seq_len

        # Query projection
        if self.q_lora_rank == 0:
          # [batch_size, seq_len, model_dim]
            q = self.wq(x)
        else:
          # [batch_size, seq_len, model_dim] -> [batch_size, seq_len, q_lora_rank] -> [batch_size, seq_len, n_heads * qk_head_dim]
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        # [batch_size, seq_len, n_heads * qk_head_dim] -> [batch_size, seq_len, n_heads, qk_head_dim]
        q = q.view(bsz, seq_len, self.n_heads, self.qk_head_dim)
        # [batch_size, seq_len, n_heads, qk_head_dim] -> [batch_size, seq_len, n_heads, q_nope_head_dim],  [batch_size, seq_len, n_heads, q_rope_head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # Key/Value projection
        # [batch_size, seq_len, model_dim] ->  [batch_size, seq_len, kv_lora_rank + qk_rope_head_dim]
        kv = self.wkv_a(x)
        # [batch_size, seq_len, kv_lora_rank + qk_rope_head_dim] -> [batch_size, seq_len, kv_lora_rank], [batch_size, seq_len, qk_rope_head_dim]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # [batch_size, seq_len, 1, qk_rope_head_dim]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        # Get weights (add dequantization if needed)
        # [kv_lora_rank, n_heads (qk_nope_head_dim + v_head_dim)]
        wkv_b = self.wkv_b.weight
        # [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

        # Project q_nope
        # [batch_size, seq_len, n_heads, q_nope_head_dim], [n_heads, qk_nope_head_dim, kv_lora_rank] ->  [batch_size, seq_len, n_heads, kv_lora_rank]
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])

        if self.cache:
            # Update cache - ensure we're only writing to valid positions

            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)

            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

            # Compute scores using full cache up to end_pos
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                     torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        else:
          #  [batch_size, seq_len, kv_lora_rank ]
            kv_proj = self.kv_norm(kv)
            # [batch_size, seq_len, qk_rope_head_dim]
            k_pe = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, kv_proj) + # [batch_size, seq_len, n_heads, kv_lora_rank], [batch_size, seq_len, kv_lora_rank] -> [batch_size, seq_len, n_heads, seq_len]
                     torch.einsum("bshr,btr->bsht", q_pe, k_pe)) * self.softmax_scale # [batch_size, seq_len, n_heads, q_rope_head_dim], [batch_size, seq_len, qk_rope_head_dim] -> [batch_size, seq_len, n_heads, seq_len]

        # Apply mask (use provided mask or create causal mask)
        if mask is not None:
            scores += mask.unsqueeze(1)  # Match official implementation
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        if self.cache:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        else:
          # [batch_size, seq_len, n_heads, seq_len], [n_heads, qk_nope_head_dim, kv_lora_rank] -> [batch_size, seq_len, n_heads, kv_lora_rank]
            x = torch.einsum("bsht,btc->bshc", scores, kv_proj)
          # [batch_size, seq_len, n_heads, kv_lora_rank], [batch_size, v_head_dim, kv_lora_rank] -> [batch_size, seq_len, n_heads, v_head_dim]
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        # [batch_size, seq_len, n_heads * v_head_dim] ->  [batch_size, seq_len, model_dim]
        x = self.wo(x.flatten(2))
        return x

class Expert(nn.Module):
    """Single expert network (essentially a FeedForward layer)"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.linear2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.linear3 = nn.Linear(config.d_model, config.d_ff)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x))*self.linear3(x))
    
class TopKRouter(nn.Module):
    """Router that selects top-k experts for each token"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.gate = nn.Linear(config.d_model, self.num_experts, bias=False)
        self.noise_std = 0.1  # Standard deviation for noise during training

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - router_weights: Softmax weights for selected experts [batch_size, seq_len, top_k]
            - expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            - router_probs: Full probability distribution over experts (for load balancing loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get full probability distribution (for load balancing loss)
        router_probs = F.softmax(router_logits, dim=-1) # [batch_size, seq_len, num_experts]

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1) # [batch_size, seq_len, num_experts]
        top_k_weights = F.softmax(top_k_logits, dim=-1) # [batch_size, seq_len, num_experts]

        return top_k_weights, top_k_indices, router_probs

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing"""
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.load_balancing_weight = config.load_balancing_weight

        # Create experts
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(self.num_experts)
        ])

        # Create router
        self.router = TopKRouter(config)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x) # [batch_size, seq_len, num_experts]

        # Initialize output tensor
        output = torch.zeros_like(x) # [batch_size, seq_len, d_model]

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get weights for this expert - CORRECTED APPROACH
                # First get the mask for this expert's positions
                mask_for_expert = (expert_indices == expert_idx)  # [batch, seq, top_k]
                # Find which position (0 or 1) this expert appears in for relevant tokens
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                # Gather weights only for relevant tokens
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)

                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        This encourages the router to distribute tokens evenly across experts.
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()

        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts

        return aux_loss * self.load_balancing_weight
    
class TransformerBlock(nn.Module):
    def __init__(self, cache,config: ModelConfig):  # Pass the entire config object
        super().__init__()
        self.cache = cache
        self.attention = MultiHeadAttention(cache, config)
        self.feed_forward = MixtureOfExperts(config)
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis ,start_pos = 0, mask = None):
        # [Batch_size, seq_len, embed_dim]
        attn_out = self.attention(self.norm1(x), freqs_cis ,start_pos, mask)
        # [Batch_size, seq_len, embed_dim]
        x = x + self.dropout(attn_out)
        # [Batch_size, seq_len, embed_dim]
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss

class MinimalLLM(nn.Module):
    def __init__(self, cache ,config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        self.cache = cache

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cache, config) for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer("freqs_cis", precompute_freqs_cis(config), persistent=False)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def reset_kv_cache(self):
        for block in self.transformer_blocks:
            block.reset_kv_cache()
    def forward(self, x, start_pos = 0, return_aux_losses = True):
      # [batch_size, seq_len, vocab_size] -> [batch_size, seq_len, embed_dim]
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        seqlen = x.shape[1]
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        aux_losses = []
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device).triu_(1)
      # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, embed_dim]
        for block in self.transformer_blocks:
            x, aux_loss = block(x, freqs_cis, start_pos, mask)
            if aux_loss is not None and return_aux_losses:
                aux_losses.append(aux_loss)
      #  [batch_size, seq_len, embed_dim]
        x = self.norm(x)
      # [batch_size, seq_len, embed_dim]
        x = self.output_dropout(x)
      # [batch_size, seq_len, embed_dim] ->  [batch_size, seq_len, vocab_size]
        logits = self.lm_head(x)
        total_aux_loss = sum(aux_losses) if aux_losses else None
        if return_aux_losses:
            return logits, total_aux_loss
        return logits