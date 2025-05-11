import torch, argparse, json, os, numpy as np
from functools import partial
from grpo import GRPOLoss
from reward_model import RewardModel
from utils.metrics import CSVLogger
from model import GPTConfig, GPT                    # local model.py

# Load Shakespeare dataset directly
train_data = np.memmap(os.path.join('data', 'shakespeare_char', 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join('data', 'shakespeare_char', 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(block_size, split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (64,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    return x

def main(cfg_path, log_path):
    cfg = json.load(open(cfg_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy_cfg = GPTConfig(**cfg['model'])
    policy = GPT(policy_cfg).to(device)
    ref    = GPT(policy_cfg).to(device).eval().requires_grad_(False)
    rm     = RewardModel(cfg['rm']).to(device).eval().requires_grad_(False)

    loss_fn = GRPOLoss(cfg['kl_coef'])
    opt     = torch.optim.AdamW(policy.parameters(), lr=cfg['lr'])
    logger  = CSVLogger(log_path)

    get_batch_fn = partial(get_batch, cfg['data']['block_size'])
    for step in range(cfg['steps']):
        x = get_batch_fn('train').to(device)          # [B,T]
        y, logp_tokens = policy.generate(x, cfg['gen']['max_new'], return_logp=True)
        with torch.no_grad():
            r = rm(x, y)
            logp_ref = ref(x, y, return_logprob=True)
        loss, pg, kl, ent, adv = loss_fn(logp_tokens, logp_ref, r)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % cfg['log_interval'] == 0:
            logger.write(step, loss.item(), r.mean().item(), kl.item(), ent.item(), adv.item())
            print(f"{step:>6} loss {loss:.3f} R {r.mean():.2f} KL {kl:.3f} H {ent:.2f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cfg'); ap.add_argument('--log', default='logs/grpo.csv');
    args = ap.parse_args(); main(args.cfg, args.log) 