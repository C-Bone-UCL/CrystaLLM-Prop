"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py
"""
import math
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from crystallm import CIFTokenizer, CIFTokenizer_extd


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 371  # number of tokens in the vocabulary (CB: updated to 371)
    new_vocab_size: int = 372  # number of tokens in the new vocabulary (CB: added this)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    lora_r: int = 8   # Rank for LoRA
    lora_alpha: int = 8  # Scaling factor for LoRA
    lora_dropout: float = 0.1  # Optional dropout for LoRA
    finetune_method: str = 'finetune_all'  # Method for fine-tuning the model (CB: added this)
    sanity_check: bool = False  # Whether to print the parameters that are being decayed (CB: added this)
    latent_dim: int = 256  # Dimension of the latent space for regression (CB: added this)
    unk_token_id: int = 370
    max_token_length: int = 4792

class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool):
        """
        Initialize the LayerNorm module.

        :param ndim: dimensionality of the input tensor
        :param bias: whether to add a learnable bias to the output
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies causal self-attention to the given tensor,
        with a mask to prevent attention to future positions.

        :param x: tensor of shape (batch size, sequence length, embedding dimension)
        :returns: result of applying the causal self-attention operation
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')  # Use built-in gelu with 'tanh' approximation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Transformer Block module. A Block module includes causal self-attention,
        layer normalization, and MLP, and residual connections.

        :param x: input to the transformer block
        :returns: output of the transformer block, with the same shape as in the input
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# LoRA Implementation, (CB: added this)
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / math.sqrt(rank)
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

# model.py

class LinearWithLoRA(nn.Module):
    def __init__(self, old_linear: nn.Linear, rank: int, alpha: int):
        super().__init__()
        # Create a new linear layer and copy weights and bias
        self.linear = nn.Linear(old_linear.in_features, old_linear.out_features, bias=(old_linear.bias is not None))
        self.linear.weight = old_linear.weight
        if old_linear.bias is not None:
            self.linear.bias = old_linear.bias
        self.lora = LoRALayer(old_linear.in_features, old_linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
class AttentionPooling(nn.Module):
    def __init__(self, n_embd, unk_token_id):
        super().__init__()
        self.attention_weights = nn.Linear(n_embd, 1)  # Linear layer to compute attention scores
        self.unk_token_id = unk_token_id  # ID for the <unk> token

    def forward(self, x, idx):
        # Create a mask for <unk> tokens
        mask = (idx != self.unk_token_id).float()  # Shape: (b, t), 1 for non-<unk>, 0 for <unk>

        # Compute attention scores
        attn_scores = self.attention_weights(x).squeeze(-1)  # Shape: (b, t)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Mask <unk> tokens
        attn_weights = F.softmax(attn_scores, dim=-1)  # Compute softmax over the sequence length

        # Weighted sum of embeddings
        x = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # Shape: (b, n_embd)
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model. Subtract the position embeddings
        by default. The token embeddings are always included, since they are used in the
        final layer due to weight tying.

        :param non_embedding: whether to subtract the position embeddings (default is True)
        :returns: the number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def resize_token_embeddings(self, new_vocab_size: int):
        """
        Resizes the token embeddings matrix of the model and adjusts lm_head accordingly.
        """
        old_vocab_size = self.config.vocab_size
        if new_vocab_size == old_vocab_size:
            return
        elif new_vocab_size < old_vocab_size:
            raise ValueError("Cannot resize token embeddings to a smaller size without tensor issues.")
        else:
            # Update the config
            self.config.vocab_size = new_vocab_size

            # Resize transformer.wte (token embeddings)
            old_embedding_weight = self.transformer.wte.weight.data.clone()
            self.transformer.wte = nn.Embedding(new_vocab_size, self.config.n_embd)
            with torch.no_grad():
                self.transformer.wte.weight[:old_vocab_size, :] = old_embedding_weight
                nn.init.normal_(self.transformer.wte.weight[old_vocab_size:, :], mean=0.0, std=0.02)

            # Resize lm_head (output layer)
            old_lm_head_weight = self.lm_head.weight.data.clone()
            self.lm_head = nn.Linear(self.config.n_embd, new_vocab_size, bias=False)
            with torch.no_grad():
                self.lm_head.weight[:old_vocab_size, :] = old_lm_head_weight
                nn.init.normal_(self.lm_head.weight[old_vocab_size:, :], mean=0.0, std=0.02)

            # Re-establish weight tying
            self.lm_head.weight = self.transformer.wte.weight

    def replace_linear_with_lora(self, rank: int = 16, alpha: int = 16):
        """
        Replace all nn.Linear layers in the transformer with LinearWithLoRA.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                parent_module = self.get_submodule(parent_name)
                old_linear = getattr(parent_module, module_name)
                lora_linear = LinearWithLoRA(old_linear, rank, alpha)
                setattr(parent_module, module_name, lora_linear)

    def get_submodule(self, target_name):
        """
        Recursively find the submodule given a dotted path.
        """
        names = target_name.split('.')
        module = self
        for name in names:
            if name:
                module = getattr(module, name)
        return module

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        # implement by cropping the position embeddings to the desired size, handling the case where pytorch is not new enough (CB: added this)
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            # Check if bias exists (when Flash Attention is not available)
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Configure optimizers with proper weight decay.
        """
        # Gather all named parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear, LinearWithLoRA)
        blacklist_weight_modules = (nn.LayerNorm, LayerNorm, nn.Embedding, LoRALayer)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                fpn = f"{mn}.{pn}" if mn else pn
                if self.config.finetune_method == 'LoRA':
                    decay.add(fpn)
                    if self.config.sanity_check:
                        print(f"weight decay applied to {fpn}, it's a {m.__class__}")
                else:
                    if pn.endswith('bias'):
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)

        decay.discard('lm_head.weight')
        if self.config.sanity_check:
            print("lm_head.weight not decayed for weight tying")

        # Validate that all parameters are assigned
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        # print the decay and no_decay parameters for sanity check
        if self.config.sanity_check:
            print("Decay:")
            for pn in sorted(decay):
                print(pn)
            print("\nNo Decay:")
            for pn in sorted(no_decay):
                print(pn)

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        tokenizer = CIFTokenizer()
        newline_id = tokenizer.token_to_id["\n"]
        prev_id = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # a sequence of two newlines indicates the end of a CIF file
            if prev_id is not None and prev_id == newline_id and idx_next.item() == newline_id:
                break
            prev_id = idx_next.item()

        return idx

class GPT_regression(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_token_length is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.max_token_length, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Initialize AttentionPooling with unk_token_id
        print(f"UNK token ID: {config.unk_token_id}, using it for AttentionPooling")
        self.attention_pooling = AttentionPooling(config.n_embd, unk_token_id=config.unk_token_id)

        print(f"Using latent dimension of {config.latent_dim} for regression head")
        self.lm_head = nn.Sequential(
            torch.nn.Linear(config.n_embd, config.latent_dim),
            torch.nn.ReLU(), # ReLU activation function
            torch.nn.Linear(config.latent_dim, 1)
        )

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model. Subtract the position embeddings
        by default. The token embeddings are always included, since they are used in the
        final layer due to weight tying.

        :param non_embedding: whether to subtract the position embeddings (default is True)
        :returns: the number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # Ensure the block size is not exceeded
        if t > self.config.max_token_length:
            raise ValueError(f"Sequence length {t} exceeds configured block size {self.config.max_token_length}")

        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # shape (b, t, n_embd)

        # Apply attention pooling to get a single embedding for the entire sequence
        x = self.attention_pooling(x, idx)  # shape (b, n_embd)

        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.mse_loss(logits, targets.unsqueeze(-1))
        else:
            loss = None

        return logits, loss

    def resize_token_embeddings(self, new_vocab_size: int):
        """
        Resizes the token embeddings matrix of the model and adjusts lm_head accordingly.
        """
        old_vocab_size = self.config.vocab_size
        if new_vocab_size == old_vocab_size:
            return
        elif new_vocab_size < old_vocab_size:
            raise ValueError("Cannot resize token embeddings to a smaller size without tensor issues.")
        else:
            # Update the config
            self.config.vocab_size = new_vocab_size

            # Resize transformer.wte (token embeddings)
            old_embedding_weight = self.transformer.wte.weight.data.clone()
            self.transformer.wte = nn.Embedding(new_vocab_size, self.config.n_embd)
            with torch.no_grad():
                self.transformer.wte.weight[:old_vocab_size, :] = old_embedding_weight
                nn.init.normal_(self.transformer.wte.weight[old_vocab_size:, :], mean=0.0, std=0.02)

            # Resize lm_head (output layer)
            old_lm_head_weight = self.lm_head.weight.data.clone()
            self.lm_head = nn.Linear(self.config.n_embd, new_vocab_size, bias=False)
            with torch.no_grad():
                self.lm_head.weight[:old_vocab_size, :] = old_lm_head_weight
                nn.init.normal_(self.lm_head.weight[old_vocab_size:, :], mean=0.0, std=0.02)

    def replace_linear_with_lora(self, rank: int = 16, alpha: int = 16):
        """
        Replace all nn.Linear layers in the transformer with LinearWithLoRA.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                parent_module = self.get_submodule(parent_name)
                old_linear = getattr(parent_module, module_name)
                lora_linear = LinearWithLoRA(old_linear, rank, alpha)
                setattr(parent_module, module_name, lora_linear)

    def resize_block(self, max_token_length: int):
        # Update the block size in the model configuration
        if max_token_length > self.config.block_size:
            print(f"Increasing block size from {self.config.block_size} to {max_token_length}.")
            
            # Expand the position embeddings if max_token_length is greater
            old_weights = self.transformer.wpe.weight.data
            new_weights = torch.zeros(max_token_length, old_weights.size(1)).to(old_weights.device)
            # Copy the old weights into the new expanded tensor
            new_weights[:self.config.block_size] = old_weights
            # Initialize the additional positions with random values
            torch.nn.init.normal_(new_weights[self.config.block_size:], mean=0.0, std=0.02)
            self.transformer.wpe.weight = nn.Parameter(new_weights)
        elif max_token_length < self.config.block_size:
            print(f"Decreasing block size from {self.config.block_size} to {max_token_length}.")
            
            # Crop the position embeddings if max_token_length is smaller
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:max_token_length])
            for block in self.transformer.h:
                # Check if bias exists (when Flash Attention is not available)
                if hasattr(block.attn, 'bias'):
                    block.attn.bias = block.attn.bias[:,:,:max_token_length,:max_token_length]
        
        # Update the block size in the model configuration
        self.config.max_token_length = max_token_length

        # Ensure all attention biases are also resized to match the new block size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias') and block.attn.bias is not None:
                block.attn.bias = block.attn.bias[:, :, :max_token_length, :max_token_length]

    
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Configure optimizers with proper weight decay.
        """
        # Gather all named parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear, LinearWithLoRA)
        blacklist_weight_modules = (nn.LayerNorm, LayerNorm, nn.Embedding, LoRALayer)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                fpn = f"{mn}.{pn}" if mn else pn
                if self.config.finetune_method == 'LoRA':
                    decay.add(fpn)
                    if self.config.sanity_check:
                        print(f"weight decay applied to {fpn}, it's a {m.__class__}")
                else:
                    if pn.endswith('bias'):
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)

        # do not remove lm_head.weight from decay set if we are doing regression head bc now weights arent tied

        # Validate that all parameters are assigned
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.max_token_length
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def predict(self, idx):
        self.eval()
        logits, _ = self.forward(idx)
        return logits
