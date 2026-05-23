import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import numpy as np
    import sys
    import random

    return (
        DataLoader,
        F,
        datasets,
        nn,
        np,
        optim,
        random,
        sys,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup
    - Set device (GPU/CPU)
    - Print Python & Pytorch versions
    - Set random seeds for reproducibility
    """)
    return


@app.cell
def _(np, random, sys, torch):
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("No GPU detected. Some cells may run slowly.")

    print(f"\nPython {sys.version.split()[0]}")
    print(f"\nPytorch {torch.__version__}")

    # Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Random seed set to {SEED}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hyperparameters
    """)
    return


@app.cell
def _():
    # Hyperparameters
    input_size = 784  # MNIST dataset has images of size 28 x 28 pixels, black & white.
    num_classes = 10  # 0, 1, 2, 3, .... , 9
    learning_rate = 0.001  # needed for Adam optimizer during training
    batch_size = 64
    num_epochs = 10  # 1 epoch trains all of the data once
    return batch_size, input_size, learning_rate, num_classes, num_epochs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load Data
    """)
    return


@app.cell
def _(DataLoader, batch_size, datasets, transforms):
    # Load Data
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fully Connected Network
    """)
    return


@app.cell
def _(F, nn):
    # Fully Connected Network
    class NN(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 50)
            self.fc2 = nn.Linear(50, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return (NN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initialize Network
    """)
    return


@app.cell
def _(NN, device, input_size, num_classes):
    # Initialize Network
    model = NN(input_size=input_size, num_classes=num_classes).to(device=device)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loss & Optimizer
    """)
    return


@app.cell
def _(learning_rate, model, nn, optim):
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    return criterion, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train the Network
    """)
    return


@app.cell
def _(criterion, device, mo, model, num_epochs, optimizer, train_loader):
    # Train the Network
    train_losses = []  # stores average loss per epoch

    with mo.status.progress_bar(total=num_epochs, title="Training...") as bar:
        for epoch in range(num_epochs):
            total_loss = 0  # accumulate batch losses for this epoch
            for batch_idx, (data, targets) in enumerate(train_loader):
                # Get data to cuda if possible
                data = data.to(device=device)  # ([64, 1, 28, 28]) : ([batch_size, channels, height, width])
                targets = targets.to(device=device)
    
                # Get to correct shape
                data = data.reshape(data.shape[0], -1)  # ([64, 1, 28, 28]) -> ([64, 784]). NN expects this shape for input layer
    
                # forward pass
                scores = model(data)
                loss = criterion(scores, targets)
    
                # backward pass
                optimizer.zero_grad()  # resets the gradient from previous run to zero
                loss.backward()  # computes gradient by differentiation using chain rule
    
                # gradient descent or adam step
                optimizer.step()
    
                total_loss += loss.item()  # extract scalar from tensor
    
                # Print training stats every 100 batches
                if batch_idx % 500 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
    
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f">>> Epoch {epoch+1} complete - Avg Loss: {avg_loss:.4f}\n")
            bar.update()  # advances the bar by 1 after each epoch

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check accuracy on training & test dataset
    """)
    return


@app.cell
def _(device, torch):
    # Check accuracy on training & test dataset
    def check_accuracy(loader, model):
        if loader.dataset.train:
            print("Checking accuracy on Training Data:")
        else:
            print("Checking accuracy on Testing Data:")
        num_correct = 0  # initialize the number of correctly classified samples to zero
        num_samples = 0  # initialize the number of total samples classified to zero
        model.eval()

        with torch.no_grad():  # disable calculating gradient descent for inference
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device)
                x = x.reshape(x.shape[0], -1)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += (predictions).size(0)

            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}\n')

        model.train()

    return (check_accuracy,)


@app.cell
def _(check_accuracy, model, test_loader, train_loader):
    check_accuracy(loader=train_loader, model=model)
    check_accuracy(loader=test_loader, model=model)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
