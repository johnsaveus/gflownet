import torch
import torch_geometric.data as gd
from rdkit import Chem
from gflownet.proxy_pyg.model import CombinedRepresentation
from gflownet.proxy_pyg.make_datasets import mol2graph

### Load model
ppath = "/home/john-savvas/Documents/gflownet/src/gflownet/proxy_pyg/best_model.pt"
ckp = torch.load(ppath)
model = CombinedRepresentation(fp_dim=167, graph_in_channels=24, hidden_dim=32, output_dim=1)
model.load_state_dict(ckp["model_state_dict"])
model.eval()
# Load some data

smiles = ["CCO", "CN"]
graphs = [mol2graph(Chem.MolFromSmiles(smi)) for smi in smiles]
batch = gd.Batch.from_data_list([i for i in graphs if i is not None])

preds = model(batch["fp"], batch["x"], batch["edge_index"], batch["batch"]).reshape((-1,)).data.cpu()

print(preds)
