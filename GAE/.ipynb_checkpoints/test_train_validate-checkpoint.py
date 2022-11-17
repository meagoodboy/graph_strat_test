from tqdm.auto import trange, tqdm
import numpy as np
import torch

def train(model,loader,optimizer,device, lossfn):
    model.train()
    loss_per_dp = []
    node_loss_per_dp = []
    edge_loss_per_dp = []
    with tqdm(loader, total=len(loader)) as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index)
            decoded_nodes = model.node_decode(z)
            edge_loss = model.recon_loss(z,  batch.edge_index)
            node_loss = lossfn(decoded_nodes, batch.x, gamma=3, reduction='mean')
            loss = edge_loss + node_loss
            loss.backward()
            optimizer.step()
            node_loss_per_dp.append(float(node_loss.item()))
            edge_loss_per_dp.append(float(edge_loss.item()))
            loss_per_dp.append(float(loss.item()))
            # tepoch.set_postfix(train_loss=get_divisors(i))
        # print('{0: <20}'.format("Train Loss : "), np.mean(node_loss_per_dp), np.mean(edge_loss_per_dp), np.mean(loss_per_dp))
    print("Train Loss : ", np.mean(node_loss_per_dp), np.mean(edge_loss_per_dp), np.mean(loss_per_dp))
    return np.mean(loss_per_dp)

def validate(model,loader,device, lossfn):
    model.eval()
    loss_per_dp = []
    with tqdm(loader, total=len(loader)) as tepoch:
        for batch in tepoch:
            with torch.no_grad():
                batch = batch.to(device)
                z = model.encode(batch.x, batch.edge_index)
                decoded_nodes = model.node_decode(z)
                edge_loss = model.recon_loss(z,  batch.edge_index)
                node_loss = lossfn(decoded_nodes, batch.x, gamma=3, reduction='mean')
                loss = edge_loss + node_loss
                loss_per_dp.append(float(loss.item()))
    print('{0: <20}'.format("Validation Loss : "), np.mean(loss_per_dp))
    return np.mean(loss_per_dp)

def encode(model,loader,device):
    model.eval()
    final_vec = []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index)
            final_vec.append(z.cpu().numpy())
    return final_vec