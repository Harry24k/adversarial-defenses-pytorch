import torch

def get_acc(model, data_loader, atk=None, n_limit=None):
    model = model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0

    for batch_images, batch_labels in data_loader:

        X = batch_images.to(device)
        Y = batch_labels.to(device)

        if atk:
            X = atk(X, Y)
            
        with torch.no_grad():
            pre = model(X)

        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

        if n_limit:
            if total > n_limit:
                break

    return (100 * float(correct) / total)
