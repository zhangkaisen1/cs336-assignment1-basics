import os
import numpy as np
import torch
import hydra
import time
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint

from cs336_basics.config import TrainConfig

@torch.no_grad()
def evaluate(model:TransformerLM, data, cfg: TrainConfig, device):
    """
    Estimates the loss over a number of batches.
    """
    model.eval()
    losses = []
    for k in tqdm(range(cfg.training.eval_iters), desc="Evaluating", leave=False):
        x, y = get_batch(data, cfg.training.batch_size, cfg.model.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    mean_loss = np.mean(losses)
    return {
        'val/loss': mean_loss,
        'val/ppl': np.exp(mean_loss),
    }


def setup(cfg: TrainConfig):
    if cfg.optimizer.min_lr is None:
        cfg.optimizer.min_lr = cfg.optimizer.max_lr * 0.1
    if cfg.training.eval_interval is None:
        cfg.training.eval_interval = cfg.training.max_iters // 10
    if cfg.training.max_iters is None:
        cfg.training.max_iters = 327_680_000 // cfg.training.batch_size // cfg.model.context_length
    if cfg.optimizer.warmup_iters is None:
        cfg.optimizer.warmup_iters = cfg.training.max_iters // 10

@hydra.main(config_path="config", config_name="train_config", version_base=None)
def main(cfg: TrainConfig):

    # print(OmegaConf.to_yaml(cfg))    
    setup(cfg)

    torch.manual_seed(cfg.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    print("Loading data...")
    data_path = Path(cfg.data.path)
    train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(data_path / 'val.bin', dtype=np.uint16, mode='r')
    print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

    model = TransformerLM(**cfg.model).to(device)
    optimizer = AdamW(
        model.parameters(), 
        lr=cfg.optimizer.max_lr, 
        betas=cfg.optimizer.betas, 
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps
    )

    start_iter = 0
    
    # --- Checkpoint Loading ---
    if cfg.training.resume_from:
        print(f"Resuming from checkpoint: {cfg.training.resume_from}")
        start_iter = load_checkpoint(cfg.training.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    start_time = time.time()
    for it in tqdm(range(start_iter, cfg.training.max_iters), desc="Training"):
        lr = lr_cosine_schedule(it, cfg.optimizer.max_lr, cfg.optimizer.min_lr, cfg.optimizer.warmup_iters, cfg.training.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = get_batch(train_data, cfg.training.batch_size, cfg.model.context_length, device)

        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = gradient_clipping(model.parameters(), max_l2_norm=1.0)

        optimizer.step()

        if it > 0 and (it % cfg.training.eval_interval == 0 or it == cfg.training.max_iters - 1):
            metrics = evaluate(model, val_data, cfg, device)
            tqdm.write(f"Iter {it}: Val loss={metrics['val/loss']:.4f}")
            if cfg.training.save_ckpt:
                checkpoint_path = output_dir / f'ckpt_{it}.pt'
                tqdm.write(f"Saving checkpoint to {checkpoint_path}")
                save_checkpoint(model, optimizer, it, checkpoint_path)

    tqdm.write("Training finished.")


if __name__ == "__main__":
    main()

