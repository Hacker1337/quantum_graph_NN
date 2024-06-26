# %%
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
from torch_geometric.nn import global_mean_pool, GCNConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.transforms import Compose, NormalizeScale, NormalizeFeatures
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch
from torch import nn
import wandb
import os
from math import floor

import argparse


parser = argparse.ArgumentParser(
    prog='train model',)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument('--lr', default=0.01)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--feature_idx', type=int, default=2)
parser.add_argument('--quantum', type=str2bool, default=True)
parser.add_argument('--dataset_frac', type=float, default=1,
                    help="Fraction of the train dataset to use")
parser.add_argument('--model_params_seed', type=int, default=0)
parser.add_argument('--model_n_layers', type=int, default=3)
parser.add_argument('--dataset_reduce_seed', type=int, default=42)
parser.add_argument('--tr_test_split_seed', type=int, default=42)

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epochs = args.num_epochs
quantum = args.quantum
n_layers = args.model_n_layers

wandb_prj_name = "graphQNN"

# ## Dataset preprocessing

# %%


# Load the QM9 dataset and filter graphs with less than 8 nodes
dataset = QM9(root='data/QM9', transform=Compose([NormalizeScale()]))
filtered_dataset = []
max_nodes = 8
for data in dataset:
    if data.num_nodes < max_nodes:
        data.x = data.x[:, :5] # keep only the atomic type
        filtered_dataset.append(data)

# %%
# normalize y values scale

y_values = torch.cat([data.y for data in filtered_dataset], dim=0)

a, b = y_values.min(dim=0)[0], y_values.max(dim=0)[0]
mean = (a + b)/2
std = (b - a)/2
for data in filtered_dataset:
    data.y = (data.y - mean)/std


# %%


generator = torch.Generator().manual_seed(args.tr_test_split_seed)
test_len = len(filtered_dataset)//5
full_train_len = len(filtered_dataset) - test_len

train_ds_full, test_ds = torch.utils.data.random_split(
    filtered_dataset, [full_train_len, test_len], generator=generator)

selected_train_len = floor(full_train_len*args.dataset_frac + 1e-9)
print(f"Training on {selected_train_len} out of {full_train_len} samples.")

generator2 = torch.Generator().manual_seed(args.dataset_reduce_seed)
train_ds, _ = torch.utils.data.random_split(
    train_ds_full, [selected_train_len, full_train_len - selected_train_len],
    generator=generator2)  # reduce the dataset

torch.manual_seed(args.model_params_seed)

# Create a DataLoader for the filtered dataset
dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=1)
criterion = nn.MSELoss()

model_params = {}

if not quantum:
    # Define a simple Graph Neural Network (GNN) model
    class GNNModel(nn.Module):
        def __init__(self, num_features, hidden_dim):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = global_mean_pool(x, batch)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize and train the GNN model
    model_params["c_hidden_dim"] = 3
    model = GNNModel(filtered_dataset[0].num_features, hidden_dim=model_params["c_hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum([p.numel() for p in model.parameters()])
    print("number of parameters is ", n_params)

# %%
if quantum:
    # ## Quantum Network

    import pennylane as qml
    import pennylane.numpy as np

    # final stage
    from typing import Literal
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def encoding_layer(values):
        """
        pooling layer, that encodes params, or edge features in nodes

        values: [num nodes] tensor
        """
        for w in range(values.shape[0]):
            qml.RX(values[w], w)

    def edged_entengling_layer(edge_index, rot_param):
        for i in range(edge_index.shape[1]):
            qml.IsingYY(rot_param, wires=[edge_index[0, i], edge_index[1, i]])

    def measurement(n_qubits):
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def parametrised_rotations(angles, wires):
        for w in wires:
            qml.RY(*angles, w)
        

    model_params["n_layers"] = n_layers

    def circuit(inputs, params, edge_weights, atoms_weight: list[5]):
        data = inputs
        num_atoms = data.x.shape[0]
        atom_embeddings = data.x @ atoms_weight   # [num_atoms, 5] * [5] -> [num_atoms]
        # assert atom_embeddings.shape == (1,)
        edge_index = data.edge_index
        for l in range(n_layers):
            encoding_layer(atom_embeddings)
            parametrised_rotations(params[2*l], wires=range(num_atoms))
            edged_entengling_layer(edge_index, edge_weights[l])
            parametrised_rotations(params[2*l+1], wires=range(num_atoms))
        return measurement(num_atoms)

    class QuantumGNN(nn.Module):
        def init_weight(self, weights_shape):
            weights = {}
            for name, shape in weights_shape.items():
                t = torch.nn.Parameter(torch.rand(
                    *shape, requires_grad=True).float()*2*np.pi, requires_grad=True)
                self.register_parameter(name, t)
                weights[name] = t
            return weights

        def __init__(self, readout: Literal["global", "local"], max_qubits: int):
            super().__init__()
            device = qml.device("lightning.qubit", wires=max_qubits)
            self.qnode = qml.QNode(circuit, device, interface="torch")
            weights_shape = {
                "params": [2 * n_layers, 1],
                "edge_weights": [n_layers],
                "atoms_weight": [5]}

            self.qnode_weights = self.init_weight(weights_shape)

            self.fc1 = nn.Linear(1, 2)
            self.fc2 = nn.Linear(2, 1)
        def forward(self, batch):
            results = []
            for i in range(len(batch)):
                g = batch[i]
                exps = self.qnode(g, **self.qnode_weights)
                exps = torch.stack(exps).float()
                avgs = torch.mean(exps, axis=0, keepdim=True)
                results.append(avgs)
            x = torch.stack(results)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # %%
    model = QuantumGNN('global', max_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum([p.numel() for p in model.parameters()])
    print("number of parameters is ", n_params)


run = wandb.init(
    # Set the project where this run will be logged
    project=wandb_prj_name,
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "n_params": n_params,
        "model": "quantum" if quantum else "classical",
        "num_nodes": max_nodes,
        "batch_size": batch_size,
        "model_params": model_params,
        "predict_feature": args.feature_idx,
        "dataset_fraction": args.dataset_frac,
    },
    save_code=True)
wandb.run.log_code("src")

run_id = wandb.run.name[wandb.run.name.rfind("-")+1:]
chkp_folder = f"output/logs/{'q' if quantum else 'c'}_{n_params}p_wandb_id={wandb.run.name}"

os.makedirs(chkp_folder, exist_ok=False)

# Training loop
try:
    for epoch in tqdm(range(0, num_epochs)):
        model.train()
        total_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y[:, args.feature_idx:args.feature_idx+1].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)

        model.eval()
        total_loss = 0
        for data in test_dataloader:
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, data.y[:, args.feature_idx:args.feature_idx+1].float())
            total_loss += loss.item()
        avg_loss_test = total_loss / len(test_dataloader)

        wandb.log({"epoch": epoch+1, "loss": avg_loss, "test_loss": avg_loss_test})
        torch.save(model.state_dict(), os.path.join(chkp_folder, f'chpt_{epoch+1}.pth'))
except KeyboardInterrupt:
    print("Interrupting training and computing metrics")

# %%
# ## Predictions visualization


# batch_size = 1
# dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)
feature_idx = args.feature_idx

model.eval()

predictions = []
answers = []
with torch.no_grad():
    for data in dataloader:
        output = model(data)
        predictions.append(output)
        answers.append(data.y[:, feature_idx:feature_idx+1])
predictions = torch.cat(predictions)
answers = torch.cat(answers)
mse = criterion(answers, predictions).item()
print("MSE loss:", round(mse, 3))
r2 = r2_score(answers, predictions)
print("R2 score:", round(r2, 6))

run.summary["r2_train"] = r2

predictions = []
answers = []
with torch.no_grad():
    for data in test_dataloader:
        output = model(data)
        predictions.append(output)
        answers.append(data.y[:, feature_idx:feature_idx+1])
predictions = torch.cat(predictions)
answers = torch.cat(answers)
mse = criterion(answers, predictions).item()
print("MSE test loss:", round(mse, 3))
r2 = r2_score(answers, predictions)
print("R2 test score:", round(r2, 6))

run.summary["r2_test"] = r2
