import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def build_loaders(train_dataset, valid_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, valid_loader, test_loader


def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        x, edge_index, edge_attr, batch, y = data_from_loader(data, device)
        pred = model(x, edge_index, edge_attr, batch)
        loss = F.mse_loss(pred.squeeze(dim=-1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader)
    scheduler.step(total_loss)
    return total_loss


def eval_epoch(model, val_loader, device):
    model.eval()
    total_loss = 0
    for data in val_loader:
        x, edge_index, edge_attr, batch, y = data_from_loader(data, device)
        with torch.no_grad():
            pred = model(x, edge_index, edge_attr, batch)
        loss = F.mse_loss(pred.squeeze(dim=-1), y)
        total_loss += loss.item()
    return total_loss / len(val_loader)


def infer_model(model, loader, device):
    all_preds = []
    all_true = []
    for data in loader:
        x, edge_index, edge_attr, batch, y = data_from_loader(data, device)
        with torch.no_grad():
            pred = model(x, edge_index, edge_attr, batch)
        pred_list = [pr for pr in pred.squeeze(dim=-1).cpu().numpy()]
        all_preds.extend(pred_list)
        true_list = [tr for tr in y.cpu().numpy()]
        all_true.extend(true_list)
    return all_preds, all_true


def get_metrics(preds, true):
    rmse = root_mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    r2 = r2_score(true, preds)

    return rmse, mae, r2


def plot_results(preds, true):
    plt.figure(figsize=(10, 6))
    plt.scatter(true, preds, alpha=0.5, edgecolors="w", marker="o", linewidth=0.2)
    plt.plot([min(true), max(true)], [min(true), max(true)], color="red", linestyle="--", linewidth=0.5)
    plt.xlabel("LogP", fontsize=14)
    plt.ylabel("Predicted LogP", fontsize=14)
    # plt.title("True vs Predicted Values", fontsize=16)
    # plt.grid(True)
    plt.savefig("results.png", bbox_inches="tight")


def data_from_loader(data, device):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    batch = data.batch.to(device)
    y = data.y.to(device)

    return x, edge_index, edge_attr, batch, y
