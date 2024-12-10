import argparse


def ready_parser():
    parser = argparse.ArgumentParser(description="Train a MPNN model")
    # Project name. For now there are 2 options: SCAFFOLD_RANDOMN and RANDOMN
    parser.add_argument(
        "--project_name",
        type=str,
        help="Wandb project name",
        default="SCAFFOLD_BALANCED",
    )
    parser.add_argument(
        "--scaling",
        type=bool,
        help="Whether to apply scaling on target",
        default=True,
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training-val", default=64
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for dataloader", default=0
    )
    # MPNN hyperparameters
    parser.add_argument(
        "--message_hidden_dim", type=int, help="Hidden dim for message", default=128
    )
    parser.add_argument("--depth", type=int, help="Depth of MPNN", default=3)
    parser.add_argument("--dropout", type=float, help="Dropout for MPNN", default=0.1)
    parser.add_argument(
        "--activation_mpnn",
        type=str,
        help="Activation function for MPNN",
        default="relu",
    )
    parser.add_argument(
        "--aggregation", type=str, help="Aggregation function", default="mean"
    )
    parser.add_argument(
        "--hidden_dim_readout", type=int, help="Hidden dim for readout", default=128
    )
    parser.add_argument(
        "--hidden_layers_readout", type=int, help="Hidden layers for readout", default=1
    )
    parser.add_argument("--dropout_readout", type=float, help="Dropout for readout")
    parser.add_argument("--batch_norm", type=bool, help="Batch norm", default=False)
    # Training hyperparameters
    parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs", default=100
    )
    parser.add_argument(
        "--init_lr", type=float, help="Initial learning rate", default=1e-3
    )
    parser.add_argument("--max_lr", type=float, help="Max learning rate", default=1e-2)
    parser.add_argument(
        "--final_lr", type=float, help="Final learning rate", default=1e-4
    )
    return parser
