import torch
from torch import nn
import dotenv
import wandb
import os

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
parser.add_argument('--batch_size', default=1)
parser.add_argument('--num_epochs', default=300)
parser.add_argument('--quantum', type=str2bool, default=True)

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epochs = args.num_epochs
quantum = args.quantum
print(quantum)


dotenv.load_dotenv("wandb.env")

wandb.login(key=os.environ["api_key"])
wandb_prj_name = "graphQNN"

# ## Dataset preprocessing

# %%
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Compose, NormalizeScale, NormalizeFeatures


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


# %% [markdown]
# Model with global readout.

# %%

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, GCNConv


# Create a DataLoader for the filtered dataset
dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.MSELoss()

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
    model = GNNModel(filtered_dataset[0].num_features, hidden_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    n_params = sum([p.numel() for p in model.parameters()])
    print("number of parameters is ", n_params)

# %%
else:
    # ## Quantum Network

    # %%
    import pennylane as qml
    import pennylane.numpy as np

    # %%
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
            qml.IsingYY(rot_param, wires=[edge_index[0,i], edge_index[1,i]])

    def measurement(n_qubits):
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def parametrised_rotations(angles, wires):
        for w in wires:
            qml.Rot(*angles, w)
        

    n_layers = 3
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
                t = torch.nn.Parameter(torch.rand(*shape, requires_grad=True).float()*2*np.pi, requires_grad=True)
                self.register_parameter(name, t)
                weights[name] = t
            return weights
        def __init__(self, readout: Literal["global", "local"], max_qubits: int):
            super().__init__()
            device = qml.device("lightning.qubit", wires=max_qubits)
            self.qnode = qml.QNode(circuit, device)
            weights_shape = {"params": [2*n_layers, 3], "edge_weights":[n_layers], "atoms_weight": [5]}

            self.qnode_weights = self.init_weight(weights_shape)
            
            self.qlayer = qml.qnn.TorchLayer(self.qnode, weights_shape)

            self.fc1 = nn.Linear(1, 2)
            self.fc2 = nn.Linear(2, 1)
        def forward(self, batch):
            # print(batch)
            results = []
            for i in range(len(batch)):
                g = batch[i]
                # print(g)
                exps = self.qnode(g, **self.qnode_weights)
                exps = torch.stack(exps).float()
                # print(exps)
                # assert exps.shape == (g.num_atoms
                avgs = torch.mean(exps, axis=0, keepdim=True)
                results.append(avgs)
            x = torch.stack(results)    
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # %%
    model = QuantumGNN('global', max_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_params = sum([p.numel() for p in model.parameters()]) - 26  # todo fix this. Remove doublecheck of the params
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
    },
    save_code=True)
wandb.run.log_code("src")
# %%
from tqdm.auto import tqdm 
import os


chkp_folder = f"logs/{'q' if quantum else 'c'}_{n_params}p_wandb_id={wandb.run.id}"

os.makedirs(chkp_folder, exist_ok=False)

# Training loop
num_epochs = 300
for epoch in tqdm(range(0, num_epochs)):
    model.train()
    total_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y[:, 0:1].float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    wandb.log({"epoch": epoch+1, "loss": avg_loss})
    torch.save(model.state_dict(), os.path.join(chkp_folder, f'chpt_{epoch+1}.pth'))


# %%
wandb.finish()

# ## Predictions visualization

# %%
import matplotlib.pyplot as plt


def visualize_predictions(model, dataloader, feature_idx):
    predictions = []
    answers = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            output = model(data)
            predictions.append(output)
            answers.append(data.y[:, feature_idx:feature_idx+1])
    predictions = torch.cat(predictions)
    answers = torch.cat(answers)
    plt.plot(answers, predictions, "o")
    plt.xlabel("correct answer")
    plt.ylabel("prediction")
    mse = criterion(answers, predictions)
    print(mse)
    return predictions, answers

# %%
batch_size = 1
dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

# %%
predictions, answers  = visualize_predictions(model, dataloader, 0)
plt.title("Q model, 26 params")
