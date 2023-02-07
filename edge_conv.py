from pathlib import Path

from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T

current_path = Path.cwd()
dataset_dir = current_path / "modelnet10"

pre_transform = T.Compose([
    T.SamplePoints(1024, remove_faces=True, include_normals=True),
    T.NormalizeScale(),
])

train_dataset = ModelNet(dataset_dir, name="10", train=True, transform=None, pre_transform=pre_transform, pre_filter=None)
test_dataset = ModelNet(dataset_dir, name="10", train=False, transform=None, pre_transform=pre_transform, pre_filter=None)



from torch_geometric.data import DataLoader as DataLoader
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
batch = next(iter(dataloader))
print(batch)



from torch_geometric.nn import knn

assign_index = knn(x=batch.pos, y=batch.pos, k=16, batch_x=batch.batch, batch_y=batch.batch)
print(assign_index.shape)
print(assign_index)


p = batch.pos[assign_index[1, :], :]
q = batch.pos[assign_index[0, :], :]
print(p.shape, q.shape)


