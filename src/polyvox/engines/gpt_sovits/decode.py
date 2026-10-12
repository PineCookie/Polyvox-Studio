"""The GPT's token-by-token decoding, faster: upstream's infer_panel_naive
(same model, sampling and stopping rules) with a fixed-size key / value cache,
so one decoding step always has the same shapes and runs as a CUDA graph.

Upstream runs each step from Python (24 layers of small kernels, each growing
its cache with torch.cat) and syncs with the GPU on every token: at ~4 ms per
token that is mostly launch overhead, and the GPT is most of synthesis time.
Here the prompt goes through upstream's own process_prompt, then every step
(embedding, layers, logits and sampling) replays one captured graph; the host
only reads back whether to stop. Off CUDA the same step runs eagerly. Each
layer's math after its cache write (attention, projections, norms, MLP) can be
fused by torch.compile (`compiled`): one compiled function serves all layers.

Sampling as upstream's logits_to_probs: the repetition penalty on every token
so far (prompt included), top-p, temperature, top-k, then exponential-race
sampling; the end of speech is masked for the first 11 steps; decoding stops at
the end of speech (sampled or most likely), the early-stop count or 1500 steps.

v4's mel DiT (fast_dit) runs its 22 blocks per flow-matching step without the
attention mask (one unpadded sequence) and, on CUDA, fused by torch.compile.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MAX_STEPS = 1500  # upstream's decoding loop
MIN_STEPS = 11  # the end of speech is masked before this step (at least 10 tokens, ~0.4 s)


def layer_rest(x, q, k_cache, v_cache, bias, out_w, out_b, n1w, n1b, w1, b1, w2, b2, n2w, n2b,
               heads: int, dim: int, eps1: float, eps2: float):  # fmt: skip
    """One layer after its query / key / value projection and cache write:
    attention as plain matrix products (for a one-token query PyTorch's fused
    attention kernels, built for long queries, took ~100 us a layer), scores
    and softmax in float32; then upstream's output projection, norms and MLP."""
    length = k_cache.shape[1]
    qh = q.view(1, 1, heads, -1).transpose(1, 2)
    kh = k_cache.view(1, length, heads, -1).transpose(1, 2)
    vh = v_cache.view(1, length, heads, -1).transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)).float() * (dim // heads) ** -0.5 + bias
    attn = torch.matmul(torch.softmax(scores, dim=-1).to(vh.dtype), vh)
    x = x + F.linear(attn.transpose(1, 2).reshape(1, 1, -1), out_w, out_b)
    x = F.layer_norm(x, [dim], n1w, n1b, eps1)
    x = x + F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)
    return F.layer_norm(x, [dim], n2w, n2b, eps2)


_compiled_layer_rest = None


def compiled_layer_rest():
    """layer_rest fused by torch.compile (once per process; compiled on first call)."""
    global _compiled_layer_rest
    if _compiled_layer_rest is None:
        _compiled_layer_rest = torch.compile(layer_rest, dynamic=False, fullgraph=True)
    return _compiled_layer_rest


# ---------------------------------------------------------------- v4's DiT


def unmasked_block(block, x, t, mask=None, rope=None):
    """A v4 DiT block for one sequence with no padding: its mask is all true,
    so the block runs without it (same output; SDPA without a mask, no padding
    of the mask, no masked_fill). (cuDNN's attention was faster on one window
    length but slower in use: it plans each new length, and every window has
    its own.)"""
    return type(block).forward(block, x, t, None, rope)


_compiled_block = None


def compiled_block():
    """unmasked_block fused by torch.compile, for any length (dynamic shapes):
    the block's weights are inputs, so one compilation per process serves every
    block of every v4 voice."""
    global _compiled_block
    if _compiled_block is None:
        _compiled_block = torch.compile(unmasked_block, dynamic=True)
    return _compiled_block


def fast_dit(sovits, compiled: bool) -> None:
    """Make a v4 SoVITS model's DiT (sovits.cfm.estimator) faster for
    synthesis, one sequence at a time: blocks unmasked, and fused if
    ``compiled``. Only for inference models: training batches need the mask."""
    import functools

    run = compiled_block() if compiled else unmasked_block
    for block in sovits.cfm.estimator.transformer_blocks:
        block.forward = functools.partial(run, block)


class Decoder:
    """Decoding for one loaded GPT (Text2SemanticDecoder, eval mode). It uses
    the model's own weight tensors, so weights loaded into the model in place
    (load_state_dict) take effect without a new graph."""

    def __init__(self, gpt, max_len: int = 2048, graphs: bool | None = None, compiled: bool = False):
        self.gpt = gpt
        self._layer_rest = compiled_layer_rest() if compiled else layer_rest
        weight = gpt.ar_predict_layer.weight
        self.device, self.dtype = weight.device, weight.dtype
        self.graphs = self.device.type == "cuda" if graphs is None else graphs
        self.max_len = max_len
        self.heads, self.dim = gpt.num_head, gpt.model_dim
        self.layers = [
            (
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                (layer.norm1.weight, layer.norm1.bias, layer.norm1.eps),
                (layer.linear1.weight, layer.linear1.bias, layer.linear2.weight, layer.linear2.bias),
                (layer.norm2.weight, layer.norm2.bias, layer.norm2.eps),
            )
            for layer in gpt.h.layers
        ]
        vocab = gpt.vocab_size
        dev = self.device

        def scalar(value, dtype=torch.long):
            return torch.tensor(value, dtype=dtype, device=dev)

        # The state one step reads and advances (all on the device: a graph replays it).
        n = len(self.layers)
        self.k = torch.zeros(n, 1, max_len, self.dim, dtype=self.dtype, device=dev)
        self.v = torch.zeros_like(self.k)
        self.token = torch.zeros(1, 1, dtype=torch.long, device=dev)  # the last sampled token
        self.cache_pos = scalar(0)  # where its keys / values go
        self.audio_pos = scalar(0)  # its position among the semantic tokens
        self.step_idx = scalar(0)  # upstream's loop index of the logits being sampled
        self.seen = torch.zeros(1, vocab, dtype=torch.bool, device=dev)  # tokens so far (repetition penalty)
        self.stop = torch.zeros((), dtype=torch.bool, device=dev)
        self.temperature = scalar(1.0, torch.float32)
        self.penalty = scalar(1.35, torch.float32)
        self.top_p = scalar(1.0, torch.float32)
        self.positions = torch.arange(max_len, device=dev)
        pos = gpt.ar_audio_position
        pos.extend_pe(torch.zeros(1, max_len + MAX_STEPS, dtype=self.dtype, device=dev))
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_key = None

    # ------------------------------------------------------------ sampling
    def _sample(self, logits: torch.Tensor, top_k: int, use_top_p: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """(token [1, 1], stop) from logits [1, vocab], as upstream's sample() and its stop check."""
        eos = self.gpt.EOS
        allowed = self.step_idx >= MIN_STEPS
        logits = logits.clone()
        logits[:, eos] = torch.where(allowed, logits[:, eos], torch.full_like(logits[:, eos], -float("inf")))
        score = torch.where(logits < 0, logits * self.penalty, logits / self.penalty)
        logits = torch.where(self.seen, score, logits)
        most_likely = logits.argmax(dim=-1)
        if use_top_p:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > self.top_p
            remove[:, 0] = False
            remove = remove.scatter(1, sorted_indices, remove)
            logits = logits.masked_fill(remove, -float("inf"))
        logits = logits / self.temperature.clamp(min=1e-5)
        if top_k:
            pivot = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = torch.where(logits < pivot, -float("inf"), logits)
        probs = F.softmax(logits, dim=-1)
        q = torch.empty_like(probs).exponential_(1)
        token = torch.argmax(probs / q, dim=-1, keepdim=True)
        stop = (most_likely[0] == eos) | (token[0, 0] == eos)
        return token, stop

    def _advance(self, token: torch.Tensor, stop: torch.Tensor) -> None:
        self.token.copy_(token)
        self.stop.copy_(stop)
        self.seen.scatter_(1, token, True)
        self.step_idx.add_(1)

    # ------------------------------------------------------------ one step
    def _step(self, top_k: int, use_top_p: bool) -> None:
        """Feed the last token, sample the next (the static state in, the static state out)."""
        gpt, heads = self.gpt, self.heads
        emb = gpt.ar_audio_embedding.word_embeddings(self.token)  # [1, 1, dim]
        pos = gpt.ar_audio_position
        # index_select, not pe[:, audio_pos]: a 0-d index tensor becomes a Python int (a sync)
        x = emb * pos.x_scale + pos.alpha * pos.pe.index_select(1, self.audio_pos.view(1)).to(emb.dtype)
        bias = torch.where(self.positions <= self.cache_pos, 0.0, -float("inf")).view(1, 1, 1, -1)
        index = self.cache_pos.view(1)
        for i, (qkv_w, qkv_b, out_w, out_b, norm1, mlp, norm2) in enumerate(self.layers):
            q, k, v = F.linear(x, qkv_w, qkv_b).chunk(3, dim=-1)
            self.k[i].index_copy_(1, index, k)
            self.v[i].index_copy_(1, index, v)
            x = self._layer_rest(
                x, q, self.k[i], self.v[i], bias, out_w, out_b, *norm1[:2], *mlp, *norm2[:2], heads, self.dim,
                norm1[2], norm2[2],
            )  # fmt: skip
        logits = gpt.ar_predict_layer(x[:, -1])
        token, stop = self._sample(logits, top_k, use_top_p)
        self._advance(token, stop)
        self.cache_pos.add_(1)
        self.audio_pos.add_(1)

    def _run_step(self, top_k: int, use_top_p: bool) -> None:
        if self.graphs:
            self._graph.replay()  # captured by generate() before decoding
        else:
            self._step(top_k, use_top_p)

    def _ensure_graph(self, top_k: int, use_top_p: bool) -> None:
        """Capture the step before a generation starts (not halfway through
        one), so every generation, the first included, runs the same steps
        and repeats exactly with the same seed."""
        key = (top_k, use_top_p)
        if self._graph is None or self._graph_key != key:
            self._capture(key)

    def _capture(self, key) -> None:
        """Record one step; replays then run it without Python. The state is
        saved and restored around the warm-up and capture runs."""
        saved = [t.clone() for t in self._state()]
        # The warm-up runs draw random numbers: without this the call that
        # captures would sample differently from later calls with the same seed.
        rng = torch.cuda.get_rng_state(self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step(*key)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step(*key)
        for t, s in zip(self._state(), saved, strict=True):
            t.copy_(s)
        torch.cuda.set_rng_state(rng, self.device)
        self._graph, self._graph_key = graph, key

    def _state(self) -> list[torch.Tensor]:
        return [self.k, self.v, self.token, self.cache_pos, self.audio_pos, self.step_idx, self.seen, self.stop]

    # ------------------------------------------------------------ decoding
    def fits(self, x: torch.Tensor, prompts: torch.Tensor | None) -> bool:
        """Whether the text and prompt leave room for a full decode in the cache."""
        return x.shape[1] + (prompts.shape[1] if prompts is not None else 0) + MAX_STEPS <= self.max_len

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        prompts: torch.Tensor | None,
        bert_feature: torch.Tensor,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        early_stop_num: int = -1,
    ) -> tuple[torch.Tensor, int]:
        """As infer_panel: (prompt + new semantic tokens [1, n], how many are new;
        0 without a prompt, where all of them are)."""
        from ...vendor.gpt_sovits.AR.models import t2s_model

        gpt = self.gpt
        x = gpt.ar_text_position(gpt.ar_text_embedding(x) + gpt.bert_proj(bert_feature.transpose(1, 2)))
        x_len = x.shape[1]
        if prompts is not None:
            y = prompts
            xy_pos = torch.concat([x, gpt.ar_audio_position(gpt.ar_audio_embedding(y))], dim=1)
        else:
            y = torch.zeros(1, 0, dtype=torch.long, device=x.device)
            xy_pos = x
        y_len = y.shape[1]
        src_len = x_len + y_len
        x_mask = F.pad(torch.zeros(x_len, x_len, dtype=torch.bool), (0, y_len), value=True)
        y_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (x_len, 0), value=False)
        xy_mask = torch.concat([x_mask, y_mask]).view(1, 1, src_len, src_len).expand(1, self.heads, -1, -1)
        xy_dec, k_cache, v_cache = gpt.t2s_transformer.process_prompt(xy_pos, xy_mask.to(x.device), None)

        for i, (k, v) in enumerate(zip(k_cache, v_cache, strict=True)):
            self.k[i, :, :src_len] = k
            self.v[i, :, :src_len] = v
        self.seen.zero_()
        if y_len:
            self.seen.scatter_(1, y.long(), True)
        self.step_idx.fill_(0)
        self.temperature.fill_(temperature)
        self.penalty.fill_(repetition_penalty)
        self.top_p.fill_(top_p)
        use_top_p = top_p < 1.0
        if self.graphs:
            self._ensure_graph(top_k, use_top_p)

        new: list[torch.Tensor] = []
        idx = 0
        for idx in t2s_model.tqdm(range(MAX_STEPS)):
            if idx == 0:  # the prompt's last position gives the first logits
                token, stop = self._sample(gpt.ar_predict_layer(xy_dec[:, -1]), top_k, use_top_p)
                self._advance(token, stop)
                self.cache_pos.fill_(src_len)
                self.audio_pos.fill_(y_len)
            else:
                self._run_step(top_k, use_top_p)
            new.append(self.token.clone())
            if bool(self.stop):
                new.pop()  # the end of speech itself
                break
            if early_stop_num != -1 and len(new) > early_stop_num:
                break
        tokens = torch.concat([y.long(), *new], dim=1) if new else y.long()
        if tokens.shape[1] == 0:  # as upstream: never empty
            tokens = torch.zeros(1, 1, dtype=torch.long, device=x.device)
        return tokens, (0 if prompts is None else idx)
