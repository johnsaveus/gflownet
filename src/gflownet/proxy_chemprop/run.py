from gflownet.proxy_chemprop.mpnn_pipeline import *
from gflownet.proxy_chemprop.arg_parser import ready_parser

# Constants for setup
INPUT_PATH = "data/KOW.csv"
SMILES_COL = "smiles"
TARGET_COL = ["active"]
SPLIT_SIZE = [0.8, 0.1, 0.1]


def train_and_evaluate(args):
    """
    Train the MPNN model using the provided arguments and evaluate performance.
    """
    set_seed()
    setup_run(args)
    # Data Preparation
    print("Loading data...")
    all_data = load_data(INPUT_PATH, SMILES_COL, TARGET_COL)
    train_data, val_data, test_data = get_split(all_data, args.project_name, SPLIT_SIZE)
    train_dataset, val_dataset, test_dataset, scaler = create_datasets(train_data, val_data, test_data, args.scaling)
    train_loader, val_loader, test_loader = create_dataloader(
        train_dataset, val_dataset, test_dataset, args.batch_size, args.num_workers
    )
    print("Building model...")
    mp = message_passing(args)
    agg = get_aggregation(args.aggregation)
    ffn = readout_mlp(args, scaler)
    model = build_mpnn(mp, agg, ffn, args)

    print("Training model...")
    trainer = ready_trainer(max_epochs=args.max_epochs)
    trainer.fit(model, train_loader, val_loader)

    print("Evaluating model...")
    ckpt = get_best_checkpoint(trainer)
    best_model = load_model(ckpt)
    real, preds = evaluate_and_infer_model(
        train_data,
        val_data,
        test_data,
        train_dataset,
        val_loader,
        test_loader,
        best_model,
        args,
    )
    plot(real, preds)


def evaluate_and_infer_model(
    train_data,
    val_data,
    test_data,
    train_dataset,
    val_loader,
    test_loader,
    model,
    args,
):
    """
    Evaluate the model on train, validation, and test datasets, and calculate metrics.
    """
    train_real = [d.y[0] for d in train_data[0]]
    val_real = [d.y[0] for d in val_data[0]]
    test_real = [d.y[0] for d in test_data[0]]
    real = {"train": train_real, "val": val_real, "test": test_real}

    # Update train loader for predictions
    train_loader = recreate_train_loader(train_dataset, args.batch_size, args.num_workers)
    train_preds, val_preds, test_preds = predict(train_loader, val_loader, test_loader, model)
    preds = {"train": train_preds, "val": val_preds, "test": test_preds}

    # Log metrics
    for split in ["train", "val", "test"]:
        calc_metrics(preds[split], real[split], split)

    return real, preds


def main():
    args = ready_parser().parse_args()
    print("Starting training pipeline...")
    try:
        train_and_evaluate(args)
    except Exception as e:
        print(f"Error during execution: {e}")
    print("Pipeline finished!")


if __name__ == "__main__":
    main()

"""python run.py --project_name "SCAFFOLD_RANDOM" --scaling False --batch_size 248 --message_hidden_dim 300 --depth 2 --dropout 0.2 --activation_mpnn 'relu' --aggregation 'mean' --hidden_dim_readout 64 --hidden_layers_readout 1 --dropout_readout 0.2 --batch_norm False --max_epochs 2 --init_lr 0.0001 --max_lr 0.001 --final_lr 0.00001
"""
