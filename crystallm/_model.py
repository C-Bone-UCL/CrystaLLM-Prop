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

from crystallm import CIFTokenizer


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 371  # number of tokens in the vocabulary (CB: updated to 372)
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
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
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


def gelu(x: Tensor) -> Tensor:
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function, as used in the Google BERT and
    OpenAI GPT models. See: "Gaussian Error Linear Units (GELUs)", https://arxiv.org/abs/1606.08415

    :param x: the tensor to which the GELU activation function will be applied
    :returns: the result tensor after applying the GELU activation function,
              possessing the same shape as the input tensor
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = gelu(x)
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

    def resize_token_embeddings(self, new_vocab_size: int): # CB: added this
        """
        Resizes the token embeddings matrix of the model. If the new vocabulary size is
        greater than the current vocabulary size, the token embeddings matrix is resized
        and the new tokens are randomly initialized. If the new vocabulary size is less
        than the current vocabulary size, the tokens are removed from the vocabulary and
        the corresponding rows of the token embeddings matrix are removed.

        :param new_vocab_size: the new vocabulary size
        """
        if new_vocab_size == self.config.vocab_size:
            return
        if new_vocab_size < self.config.vocab_size:
            self.lm_head = nn.Linear(self.config.n_embd, new_vocab_size, bias=False)
            self.transformer.wte = nn.Embedding(new_vocab_size, self.config.n_embd)
            self.transformer.wte.weight = self.lm_head.weight
            self.config.vocab_size = new_vocab_size
        else:
            old_vocab_size = self.config.vocab_size
            self.config.vocab_size = new_vocab_size
            old_wte = self.transformer.wte.weight.data
            self.transformer.wte = nn.Embedding(new_vocab_size, self.config.n_embd)
            self.transformer.wte.weight.data[:old_vocab_size, :] = old_wte
            new_tokens = new_vocab_size - old_vocab_size
            if new_tokens > 0:
                torch.nn.init.normal_(self.transformer.wte.weight.data[old_vocab_size:, :], mean=0.0, std=0.02)
            self.lm_head = nn.Linear(self.config.n_embd, new_vocab_size, bias=False)
            self.lm_head.weight = self.transformer.wte.weight

    # CB: added this
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
    
    # def _replace_linear(self, module, rank: int, alpha: int):
    #     """
    #     Recursively replace all nn.Linear layers in the module with LinearWithLoRA.
    #     """
    #     for name, child in module.named_children():
    #         if isinstance(child, nn.Linear):
    #             setattr(module, name, LinearWithLoRA(child, rank, alpha))
    #         else:
    #             self._replace_linear(child, rank, alpha)

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
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear, LinearWithLoRA)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding, LoRALayer)

        # if the config sinetune method is LoRA, then put all the LoRA layers in the no_decay set
        # if any other method is used, sort the parameters into decay and no_decay sets

        if self.config.finetune_method == 'LoRA':
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    # Skip parameters that are not trainable
                    if not p.requires_grad:
                        continue

                    fpn = f"{mn}.{pn}" if mn else pn

                    decay.add(fpn)
                    if self.config.sanity_check == True:
                        print(f"weight decay applied to {fpn}, it's a {m.__class__}")
                    continue

        else:
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = "%s.%s" % (mn, pn) if mn else pn # full param name
                    # random note: because named_modules and named_parameters are recursive
                    # we will see the same tensors p many many times. but doing it this way
                    # allows us to know which parent module any tensor p belongs to...
                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

        # subtle: "transformer.wte.weight" and "lm_head.weight" are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurrence, keyed by "transformer.wte.weight", below.
        # so let's manually remove "lm_head.weight" from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.

        # # CB: added this
        # # Manually handle any specific parameters if needed
        
        if self.config.finetune_method != 'LoRA':
            decay.remove("lm_head.weight")
            print("removed lm_head.weight from decay set to not have weight decay applied")
            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
        if self.config.finetune_method == 'LoRA':
            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object according to the above settings
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

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

    # @torch.no_grad()
    # def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    #     """
    #     Generate new tokens using the model, with compatibility for LoRA layers.
    #     """
    #     tokenizer = CIFTokenizer()
    #     newline_id = tokenizer.token_to_id["\n"]
    #     prev_id = None

    #     # Initialize LoRA parameters with zeros if max_new_tokens is specified
    #     # for name, module in self.named_modules():
    #     #     if isinstance(module, LinearWithLoRA):
    #     #         module.lora.A.data.zero_()
    #     #         module.lora.B.data.zero_()

    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    #         # forward the model to get the logits for the index in the sequence
    #         logits, _ = self(idx_cond)
    #         logits = logits[:, -1, :] / temperature

    #         # optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float("Inf")

    #         # apply softmax to convert logits to (normalized) probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)

    #         # append sampled index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)

    #         if prev_id is not None and prev_id == newline_id and idx_next.item() == newline_id:
    #             break
    #         prev_id = idx_next.item()

    #     return idx
