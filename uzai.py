"""UZAI: o'zbek tiliga ixtisoslashgan Transformer modeli.
O'qitish: python uzai.py train
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = "data/corpus.txt"
CKPT_PATH = "uzai.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "block_size": 256,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 6,
    "dropout": 0.1,
}

TRAIN = {
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "max_iters": 20000,
    "eval_every": 500,
    "eval_iters": 50,
    "grad_clip": 1.0,
}


# ---------- Tokenizer (lotin va kirill harflari uchun) ----------
class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

    @property
    def vocab_size(self):
        return len(self.chars)


# ---------- Transformer ----------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.qkv = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"])
        self.dropout = cfg["dropout"]
        self.resid_drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.ln1 = nn.LayerNorm(C)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(
            nn.Linear(C, 4 * C),
            nn.GELU(),
            nn.Linear(4 * C, C),
            nn.Dropout(cfg["dropout"]),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class UZAI(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # og'irliklarni bog'lash
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=200, temperature=0.8, top_k=40):
        for _ in range(max_new):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


@torch.no_grad()
def reply(model, tok, prompt, max_new=200, temperature=0.8, top_k=40):
    ids = tok.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new, temperature, top_k)
    return tok.decode(out[0, len(ids):].tolist())


# ---------- Saqlash / yuklash ----------
def save_checkpoint(model, tok, opt, step, best_loss):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": model.cfg,
        "chars": tok.chars,
        "step": step,
        "best_loss": best_loss,
    }, CKPT_PATH)


def load_model(path=CKPT_PATH):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    tok = CharTokenizer(ck["chars"])
    model = UZAI(ck["config"], tok.vocab_size).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, ck["step"]


# ---------- O'qitish ----------
def get_batch(data, block, bs):
    ix = torch.randint(len(data) - block - 1, (bs,))
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate(model, data, block):
    model.eval()
    losses = []
    for _ in range(TRAIN["eval_iters"]):
        xb, yb = get_batch(data, block, TRAIN["batch_size"])
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    if not os.path.exists(DATA_PATH):
        sys.exit(f"{DATA_PATH} topilmadi. Avval o'zbekcha matnni shu yo'lga joylashtiring.")
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    opt_state = None
    if os.path.exists(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        tok = CharTokenizer(ck["chars"])
        model = UZAI(ck["config"], tok.vocab_size).to(DEVICE)
        model.load_state_dict(ck["model"])
        start, best, opt_state = ck["step"] + 1, ck["best_loss"], ck.get("opt")
        print(f"Davom ettirilmoqda: {start}-qadamdan")
    else:
        tok = CharTokenizer(sorted(set(text)))
        model = UZAI(CONFIG, tok.vocab_size).to(DEVICE)
        start, best = 0, float("inf")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Parametrlar: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} mln")

    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    if opt_state:
        opt.load_state_dict(opt_state)

    block = model.cfg["block_size"]
    model.train()
    for step in range(start, TRAIN["max_iters"]):
        if step % TRAIN["eval_every"] == 0:
            tl = estimate(model, train_data, block)
            vl = estimate(model, val_data, block)
            print(f"qadam {step}: train={tl:.3f} val={vl:.3f}")
            if vl < best:
                best = vl
                save_checkpoint(model, tok, opt, step, best)
                print("  -> saqlandi")

        xb, yb = get_batch(train_data, block, TRAIN["batch_size"])
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
        opt.step()

    print("O'qitish tugadi.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print("Ishlatish: python uzai.py train")
