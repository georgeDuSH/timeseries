import torch
from torch import optim

from dataloader import collate_fn

def SE(input:torch.Tensor, target:torch.Tensor):
    # Squared Error
    return (input-target)**2

def train_model(model, dataset, loss_func
                , learning_rate, weight_decay
                , n_epoch, batch_size):
    
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = []
    for _ in range(n_epoch):
        x, y = collate_fn(dataset, batch_size)
        model.zero_grad()

        preds = model(x)
        ep_loss = loss_func(preds, y).mean()

        loss.append(ep_loss.tolist())

        ep_loss.backward()
        opt.step()
    
    return model, loss
