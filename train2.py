import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gnn_model import BigPolymerGINE          # the heavy model
from utils import WeightedMAELoss, calculate_competition_metric

# ---------- helpers ----------------------------------------------------------
class EMA:
    """Exponential moving average (works in-place)."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                self.shadow[n] *= self.decay
                self.shadow[n] += (1.0 - self.decay) * p.data

    def apply_shadow(self, model):
        backup = {n: p.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            p.data.copy_(backup[n])

# ---------- training / eval --------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        y = batch.y.view(batch.num_graphs, 5)
        optimizer.zero_grad()
        with autocast():
            out = model(batch)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, preds, targets = 0, [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = batch.to(device)
        y = batch.y.view(batch.num_graphs, 5)
        out = model(batch)
        loss = criterion(out, y)
        total += loss.item() * batch.num_graphs
        preds.append(out.cpu().numpy())
        targets.append(y.cpu().numpy())
    preds, targets = map(np.vstack, (preds, targets))
    comp_metric = calculate_competition_metric(preds, targets)
    return total / len(loader.dataset), comp_metric, preds, targets

# ---------- main -------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load data
    graphs = torch.load("train_graphs.pt", weights_only=False)
    graphs = [g for g in graphs if not torch.isnan(g.y).all()]
    print("Valid graphs:", len(graphs))

    # 5-fold CV (or pick fold=0 for single run)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    train_idx, val_idx = list(kf.split(graphs))[fold]
    train_ds = [graphs[i] for i in train_idx]
    val_ds   = [graphs[i] for i in val_idx]

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4)

    model = BigPolymerGINE(
        num_node_features=7,
        num_edge_features=3,
        global_features_dim=10,
        hidden_dim=512,
        num_layers=8,
        num_targets=5,
        dropout=0.15
    ).to(device)

    print("Parameters:", sum(p.numel() for p in model.parameters()))

    criterion = WeightedMAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    ema = EMA(model, decay=0.995)

    best_metric, patience, patience_max = 1e9, 0, 20
    hist = {"train": [], "val": [], "metric": []}

    for epoch in range(100):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        ema.update(model)

        # eval with EMA weights
        backup = ema.apply_shadow(model)
        val_loss, val_metric, preds, tgts = evaluate(model, val_loader, criterion, device)
        ema.restore(model, backup)

        scheduler.step(epoch + val_metric)  # use metric as plateau hint
        hist["train"].append(tr_loss)
        hist["val"].append(val_loss)
        hist["metric"].append(val_metric)

        print(f"Ep {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f} | metric {val_metric:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        if val_metric < best_metric:
            best_metric, patience = val_metric, 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience += 1
            if patience >= patience_max:
                print("Early stop")
                break

    # final bar plot
    model.load_state_dict(torch.load("best_model.pt"))
    _, _, preds, tgts = evaluate(model, val_loader, criterion, device)
    props = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    maes = [mean_absolute_error(tgts[:, i][~np.isnan(tgts[:, i])],
                                preds[:, i][~np.isnan(tgts[:, i])]) for i in range(5)]
    plt.bar(props, maes); plt.ylabel("MAE"); plt.title("Property MAE"); plt.savefig("mae.png")

    print("Best competition metric:", best_metric)

if __name__ == "__main__":
    main()