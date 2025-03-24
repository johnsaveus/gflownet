import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import DataLoader
from gflownet.proxy_pyg.make_datasets import split_data, Dataset
from gflownet.proxy_pyg.model import CombinedRepresentation


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.fp, batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y.unsqueeze(1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(train_loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.fp, batch.x, batch.edge_index, batch.batch)
            loss = criterion(pred, batch.y.unsqueeze(1))
            total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 16
    patience = 10  # Early stopping patience

    # Model setup
    model = CombinedRepresentation(fp_dim=167, graph_in_channels=24, hidden_dim=32, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Data loading
    train_dataset, val_dataset, test_dataset = split_data()
    print(train_dataset[0])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training loop
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(num_epochs):
        # Training
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                "best_model.pt",
            )
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print("Early stopping triggered")
            break

    # Load best model and evaluate on test set
    checkpoint = torch.load("best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    # test_loss = evaluate(model, test_loader, criterion, device)
    mae_criterion = nn.L1Loss()
    test_mae = evaluate(model, test_loader, mae_criterion, device)
    print(f"Test MAE: {test_mae:.6f}")


if __name__ == "__main__":
    main()
