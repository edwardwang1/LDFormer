import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

#adapted from https://github.com/karpathy/nanoGPT


import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import loralib as lora

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
# def new_gelu(x):
#     """
#     Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
#     Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
#     """
#     return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, bias):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            att = None
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # print("v shape", v.shape)
            # print("att shape", att.shape)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    # def forward(self, x, bias):
    #     x = x + self.attn(self.ln_1(x), bias)
    #     x = x + self.mlp(self.ln_2(x))
    #     return x
        
    def forward(self, x, bias):
        attn_output, attn_weights = self.attn(self.ln_1(x), bias)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights

@dataclass
class GPTConfig:
    block_size: int = 246 # maximum sequence length
    vocab_size: int = 1024 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    #attn_mask: torch.Tensor = None
    num_target_tokens: int = 0
    lora_rank: int = 1
    lora_alpha: float = 1.0
    lora_dropout: float = 0.2

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
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

    def sinusoidal_position_encodings(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.to(device)
        return pos_enc
    
    def forward(self, idx, bias, targets=None, return_attention_weights=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        #pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t) #Don't need this if using sinusoidal position encodings

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        #pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        pos_emb = self.sinusoidal_position_encodings(t, self.config.n_embd, device) #Use sinusoidal position encodings instead of learned position encodings

        x = self.transformer.drop(tok_emb + pos_emb)
        attention_weights = []
        for block in self.transformer.h:
            # x = block(x, bias)
            x, att = block(x, bias)
            attention_weights.append(att)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits[:, -self.config.num_target_tokens:, :]
            targets = targets[:, -self.config.num_target_tokens:]

            loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1).long(), ignore_index=-1, label_smoothing=0.1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_attention_weights:
            return logits, loss, attention_weights
        else:
            return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

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
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, bias, targets=None, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        """
        bias is an attention mask. all columns corresponding to the indices of the input sequence which are padding token(0)
        should be set to 0.
        """
        loss = 0
        for curr_token in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, bias)

            # Calculate loss if targets are provided
            if targets is not None:
                loss += F.cross_entropy(logits[:, -1, :], targets[:, curr_token].long())


            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature



            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)


            '''
            This pad assumes that the bias in generate should be triangular. The column padded to the right is set to 0.
            The last row is a copy of the second last row, with the exception that the last element is set to 1.

            '''
            bias = torch.nn.functional.pad(bias, (0, 1, 0, 1), value=0)
            bias[:, :, -1, :] = bias[:, :, -2, :]
            bias[:, :, -1, -1] = 1

        if targets is not None:
            loss = (loss / max_new_tokens).item()

        return idx, loss

    def generate_beam_search(self, idx, max_new_tokens, bias, temperature=1.0, beam_width=5):
        # Initialize beams as a list of tuples (sequence, score)
        beams = [(idx, 0.0)]

        for _ in range(max_new_tokens):
            new_beams = []
            for beam_idx, score in beams:
                # Ensure the sequence does not exceed block size
                idx_cond = beam_idx if beam_idx.size(1) <= self.config.block_size else beam_idx[:,
                                                                                       -self.config.block_size:]
                logits, _ = self(idx_cond, bias)

                # Apply temperature scaling
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                # Get top beam_width candidates for each beam
                top_probs, top_idx = torch.topk(probs, beam_width)

                # Expand each beam with these new candidates
                for i in range(beam_width):
                    next_idx = torch.cat((beam_idx, top_idx[:, [i]]), dim=1)
                    next_score = score + torch.log(top_probs[:, i])
                    new_beams.append((next_idx, next_score))

            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            bias = torch.nn.functional.pad(bias, (0, 1, 0, 1), value=0)
            bias[:, :, -1, :] = bias[:, :, -2, :]
            bias[:, :, -1, -1] = 1

        # Return the best sequence and its score
        best_sequence, best_score = beams[0]
        return best_sequence, best_score.item()

    @torch.no_grad()
    def generateOld(self, idx, max_new_tokens, bias, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        """
        bias is an attention mask. all columns corresponding to the indices of the input sequence which are padding token(0)
        should be set to 0.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            #print(bias.shape, idx_cond.shape)
            logits, _ = self(idx_cond, bias)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            '''
            This padding method assumes that the bias in generate should not be triangular


            #add an extra columm and extra row to the attention mask in bias. the extra column should have all values of 1.
            #the extra row should have the same values as the previous rows (all same values)
            # bias = torch.nn.functional.pad(bias, (0, 1, 0, 1), value=1)
            # bias[:, :, -1, :] = bias[:, :, -2, :]
            
            '''

            '''
            This pad assumes that the bias in generate should be triangular. The column padded to the right is set to 0.
            The last row is a copy of the second last row, with the exception that the last element is set to 1.

            '''
            bias = torch.nn.functional.pad(bias, (0, 1, 0, 1), value=0)
            bias[:, :, -1, :] = bias[:, :, -2, :]
            bias[:, :, -1, -1] = 1

        return idx
    

#Below is the GPT model rewritten with LoRA

class CausalSelfAttentionLoRA(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        #self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_attn = lora.MergedLinear(
            config.n_embd, config.n_embd * 3, 
            r=config.lora_rank,
            lora_alpha=config.lora_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=False,
            merge_weights=False
        )
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, bias):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            att = None
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # print("v shape", v.shape)
            # print("att shape", att.shape)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att

class BlockLoRA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionLoRA(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    # def forward(self, x, bias):
    #     x = x + self.attn(self.ln_1(x), bias)
    #     x = x + self.mlp(self.ln_2(x))
    #     return x
        
    def forward(self, x, bias):
        attn_output, attn_weights = self.attn(self.ln_1(x), bias)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights

class GPTLoRA(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([BlockLoRA(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
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

    def sinusoidal_position_encodings(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.to(device)
        return pos_enc
    
    def forward(self, idx, bias, targets=None, return_attention_weights=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        #pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t) #Don't need this if using sinusoidal position encodings

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        #pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        pos_emb = self.sinusoidal_position_encodings(t, self.config.n_embd, device) #Use sinusoidal position encodings instead of learned position encodings

        x = self.transformer.drop(tok_emb + pos_emb)
        attention_weights = []
        for block in self.transformer.h:
            # x = block(x, bias)
            x, att = block(x, bias)
            attention_weights.append(att)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits[:, -self.config.num_target_tokens:, :]
            targets = targets[:, -self.config.num_target_tokens:]

            loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1).long(), ignore_index=-1, label_smoothing=0.1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_attention_weights:
            return logits, loss, attention_weights
        else:
            return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

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
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, bias, targets=None, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        """
        bias is an attention mask. all columns corresponding to the indices of the input sequence which are padding token(0)
        should be set to 0.
        """
        loss = 0
        for curr_token in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, bias)

            # Calculate loss if targets are provided
            if targets is not None:
                loss += F.cross_entropy(logits[:, -1, :], targets[:, curr_token].long())


            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature



            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)


            '''
            This pad assumes that the bias in generate should be triangular. The column padded to the right is set to 0.
            The last row is a copy of the second last row, with the exception that the last element is set to 1.

            '''
            bias = torch.nn.functional.pad(bias, (0, 1, 0, 1), value=0)
            bias[:, :, -1, :] = bias[:, :, -2, :]
            bias[:, :, -1, -1] = 1

        if targets is not None:
            loss = (loss / max_new_tokens).item()

        return idx, loss

    @torch.no_grad()
    def generate_beam_search(self, idx, max_new_tokens, bias, temperature=1.0, beam_width=5):
        # Initialize beams as a list of tuples (sequence, score)
        beams = [(idx, 0.0)]

        for _ in range(max_new_tokens):
            new_beams = []
            for beam_idx, score in beams:
                # Ensure the sequence does not exceed block size
                idx_cond = beam_idx if beam_idx.size(1) <= self.config.block_size else beam_idx[:,
                                                                                       -self.config.block_size:]
                logits, _ = self(idx_cond, bias)

                # Apply temperature scaling
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                # Get top beam_width candidates for each beam
                top_probs, top_idx = torch.topk(probs, beam_width)

                # Expand each beam with these new candidates
                for i in range(beam_width):
                    next_idx = torch.cat((beam_idx, top_idx[:, [i]]), dim=1)
                    next_score = score + torch.log(top_probs[:, i])
                    new_beams.append((next_idx, next_score))

            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            beam_bias = torch.nn.functional.pad(bias, (0, 1, 0, 1), value=0)
            beam_bias[:, :, -1, :] = beam_bias[:, :, -2, :]
            beam_bias[:, :, -1, -1] = 1
            bias = beam_bias

        # Return the best sequence and its score
        best_sequence, best_score = beams[0]
        return best_sequence, best_score.item()

    @torch.no_grad()
    def generateOld(self, idx, max_new_tokens, bias, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        """
        bias is an attention mask. all columns corresponding to the indices of the input sequence which are padding token(0)
        should be set to 0.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            #print(bias.shape, idx_cond.shape)
            logits, _ = self(idx_cond, bias)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            '''
            This padding method assumes that the bias in generate should not be triangular


            #add an extra columm and extra row to the attention mask in bias. the extra column should have all values of 1.
            #the extra row should have the same values as the previous rows (all same values)
            # bias = torch.nn.functional.pad(bias, (0, 1, 0, 1), value=1)
            # bias[:, :, -1, :] = bias[:, :, -2, :]
            
            '''

            '''
            This pad assumes that the bias in generate should be triangular. The column padded to the right is set to 0.
            The last row is a copy of the second last row, with the exception that the last element is set to 1.

            '''
            bias = torch.nn.functional.pad(bias, (0, 1, 0, 1), value=0)
            bias[:, :, -1, :] = bias[:, :, -2, :]
            bias[:, :, -1, -1] = 1

        return idx


if __name__ == '__main__':
    config = GPTConfig(block_size=32, n_layer=2, n_head=2, n_embd=8)
    model = GPT(config).cuda()

    input_len = 22
    max_new_tokens = 12
    input = torch.ones(1, input_len).long().cuda()
    bias = torch.tril(torch.ones(1, input_len,input_len)).view(1, 1, input_len, input_len).cuda()


    out = model.generate_beam_search(input, max_new_tokens, bias, temperature=1.0, beam_width=2)

    print(out)