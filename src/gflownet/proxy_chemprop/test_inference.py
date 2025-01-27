from rdkit.Chem import MolFromSmiles
from gflownet.proxy_chemprop.mpnn_pipeline import load_model
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
import torch
from lightning import pytorch as pl
from chemprop import data

MODEL_PATH = "checkpoints/best-epoch=84-val_loss=0.06.ckpt"

model = load_model(MODEL_PATH)
smiles_from_data = [
    "CCON=C(C)c1cccc(Oc2nc(OC)cc(OC)n2)c1C(=O)OC",  # 13045 . value = 2.36
    "CCON=C(CC)C1C(=O)CC(c2cccc(Oc3ccc(Cl)c(Cl)c3)c2)CC1=O",  # 13046 . value = 6.92
]
featurizer = SimpleMoleculeMolGraphFeaturizer()
### ------------ Batch Mol Graph ------------ ###
test_mols = [MolFromSmiles(smile) for smile in smiles_from_data]
test_graphs = [featurizer(mol) for mol in test_mols]
mol_batch = data.BatchMolGraph(test_graphs)
batch_preds = model(mol_batch)

### ------------ Dataloader as per docs ------------ ###

test_data = [data.MoleculeDatapoint.from_smi(smile) for smile in smiles_from_data]
test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
test_loader = data.build_dataloader(test_dset, shuffle=False)
import torch
from lightning import pytorch as pl

trainer = pl.Trainer(logger=None, enable_progress_bar=True, accelerator="cpu", devices=1)
test_preds = trainer.predict(model, test_loader)

print(batch_preds)
print(test_preds)
