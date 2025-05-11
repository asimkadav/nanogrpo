import torch, argparse, json, os, numpy as np
from functools import partial
from ppo import PPOClipLoss
from reward_model import RewardModel
from utils.metrics import CSVLogger
from model import GPTConfig, GPT                  # local model.py

# Load Shakespeare dataset directly
train_data = np.memmap(os.path.join('data', 'shakespeare_char', 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join('data', 'shakespeare_char', 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(block_size, split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (64,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    return x

def main(cfg_path, log_path):
    cfg = json.load(open(cfg_path)); device='cuda' if torch.cuda.is_available() else 'cpu'
    gpt_cfg = GPTConfig(**cfg['model'])
    policy  = GPT(gpt_cfg).to(device)
    value_head = torch.nn.Linear(gpt_cfg.n_embd, 1).to(device)
    old_policy = GPT(gpt_cfg).to(device)

    opt = torch.optim.AdamW(list(policy.parameters())+list(value_head.parameters()), lr=cfg['lr'])
    loss_fn = PPOClipLoss(cfg['eps'], cfg['vf_coef'], cfg['ent_coef'])
    logger  = CSVLogger(log_path)

    get_batch_fn = partial(get_batch, cfg['data']['block_size'])
    for step in range(cfg['steps']):
        x = get_batch_fn('train').to(device)
        y, logp = policy.generate(x, cfg['gen']['max_new'], return_logp=True)
        with torch.no_grad():
            logp_old = old_policy(x, y, return_logprob=True)
        # returns via reward model (no value‑bootstrap for simplicity)
        with torch.no_grad():
            returns = RewardModel(cfg['rm']).to(device)(x, y)
        # value prediction
        h = policy.last_hidden_state            # pretend generate() cached it
        values = value_head(h.mean(1)).squeeze(-1)
        entropy = -(logp.exp() * logp).mean()
        loss,_ = loss_fn(logp, logp_old, returns, values, entropy)
        opt.zero_grad(); loss.backward(); opt.step()
        old_policy.load_state_dict(policy.state_dict())
        if step % cfg['log_interval'] == 0:
            logger.write(step, loss.item(), returns.mean().item(), 0, entropy.item(), 0)
            print(f"{step:>6} loss {loss:.3f} R {returns.mean():.2f} H {entropy:.2f}")

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('cfg'); ap.add_argument('--log', default='logs/ppo.csv');
    args=ap.parse_args(); main(args.cfg, args.log) 