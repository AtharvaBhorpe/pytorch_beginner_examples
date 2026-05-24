import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple Fully Connected Neural Network
    Trained on MNIST dataset.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import random
    import copy
    from torchinfo import summary

    return (
        DataLoader,
        F,
        copy,
        datasets,
        nn,
        np,
        optim,
        plt,
        random,
        random_split,
        summary,
        sys,
        torch,
        transforms,
    )


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
    ---
    ### Hyperparameters
    Tune the parameters and click on the "Train Model" button.
    """)
    return


@app.cell
def _(mo):
    # Hyperparameters
    input_size = 784  # MNIST dataset has images of size 28 x 28 pixels, black & white.
    num_classes = 10  # 0, 1, 2, 3, .... , 9
    in_channels = 1  # MNIST dataset is black & white, i.e., only 1 color channel

    hyper_form = (
        mo.Html("""
            <div style="display:flex; flex-direction:column; gap:16px;">
                <div style="display:flex; gap:32px; align-items:flex-end;">
                    {epochs}
                    {batch_size}
                    {hidden_size}
                </div>
                <div style="display:flex; gap:32px; align-items:flex-end;">
                    {optimizer}
                    {lr_exp}
                </div>
            </div>
        """)
        .batch(
            epochs      = mo.ui.slider(1, 30, value=10, step=1, label="Epochs"),
            batch_size  = mo.ui.slider(16, 256, value=64, step=16, label="Batch Size"),
            hidden_size = mo.ui.number(16, 512, value=128, step=16, label="Hidden Layer Size"),
            optimizer   = mo.ui.dropdown(["Adam", "SGD", "RMSprop"], value="Adam", label="Optimizer"),
            lr_exp = mo.ui.slider(-4, -1, value=-3, step=1, label="Learning Rate (10ⁿ:  -4=0.0001, -3=0.001, -2=0.01, -1=0.1)", show_value=True),
        )
        .form(submit_button_label="🚀 Train Model", bordered=True)
    )
    hyper_form
    return hyper_form, input_size, num_classes


@app.cell
def _(hyper_form, mo):
    mo.stop(hyper_form.value is None, mo.md("*Configure hyperparameters and click **Train Model**.*"))

    num_epochs     = hyper_form.value["epochs"]
    batch_size     = hyper_form.value["batch_size"]
    hidden_size    = hyper_form.value["hidden_size"]
    optimizer_name = hyper_form.value["optimizer"]
    learning_rate  = 10 ** hyper_form.value["lr_exp"]
    return batch_size, hidden_size, learning_rate, num_epochs, optimizer_name


@app.cell
def _(DataLoader, batch_size, datasets, random_split, transforms):
    # Load Data
    full_train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

    # 80/20 split
    train_size = int(0.8 * len(full_train_dataset))  # 48,000 samples for training
    val_size = len(full_train_dataset) - train_size  # 12,000 samples for validation
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, train_loader, val_loader


@app.cell
def _(F, nn):
    # Fully Connected Network
    class NN(nn.Module):
        def __init__(self, input_size, num_classes, hidden_size):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x =  F.relu(self.fc2(x))
            return self.fc3(x)

    return (NN,)


@app.cell
def _(NN, device, hidden_size, input_size, num_classes):
    # Initialize Network
    model = NN(input_size=input_size, num_classes=num_classes, hidden_size=hidden_size).to(device=device)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Model Summary
    """)
    return


@app.cell
def _(model, summary):
    summary(model, input_size=(1, 1, 28, 28))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Model Training
    """)
    return


@app.cell
def _(learning_rate, model, nn, optim, optimizer_name):
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    optimizer_map = {
        "Adam":    optim.Adam,
        "SGD":     optim.SGD,
        "RMSprop": optim.RMSprop,
    }

    optimizer = optimizer_map[optimizer_name](params=model.parameters(), lr=learning_rate)
    return criterion, optimizer


@app.cell
def _(torch):
    def get_accuracy(loader, model, device):
        model.eval()
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum().item()
                num_samples += predictions.size(0)
        model.train()
        return num_correct / num_samples

    return (get_accuracy,)


@app.class_definition
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience      # how many epochs to wait after last improvement
        self.min_delta = min_delta    # minimum change to count as an improvement
        self.counter = 0              # counts epochs without improvement
        self.best_loss = float('inf') # tracks best val_loss seen so far
        self.stop = False             # flag to signal training to halt

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            # improvement found — reset counter
            self.best_loss = val_loss
            self.counter = 0
            print(f"  ✓ Val loss improved to {val_loss:.4f}")
        else:
            # no improvement
            self.counter += 1
            print(f"  ✗ No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.stop = True
                print("Early stopping triggered!")


@app.cell
def _(
    copy,
    criterion,
    device,
    mo,
    model,
    num_epochs,
    optimizer,
    torch,
    train_loader,
    val_loader,
):
    # Train the Network
    # mo.stop(not hyper_form.value, mo.md("*Adjust hyperparameters and click **Train Model** to start.*"))

    epoch_log = []
    early_stopper = EarlyStopping(patience=3, min_delta=0.001)
    best_model_weights = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(num_epochs):
        # --- Train pass: track loss AND accuracy in one sweep ---
        model.train()
        total_loss, correct, total = 0, 0, 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (scores.argmax(1) == targets).sum().item()
            total += targets.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / total)

        # --- Validation pass: loss AND accuracy in one sweep ---
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                scores = model(x)
                val_loss += criterion(scores, y).item()
                correct += (scores.argmax(1) == y).sum().item()
                total += y.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(correct / total)

        # --- Early stopping ---
        early_stopper.step(val_losses[-1])  # latest avg validation loss
        if early_stopper.counter == 0:
            best_model_weights = copy.deepcopy(model.state_dict())

        # --- Live display (single owner of output) ---
        epoch_log.append({
            "Epoch":      f"{epoch+1}/{num_epochs}",
            "Train Loss": f"{train_losses[-1]:.4f}",
            "Val Loss":   f"{val_losses[-1]:.4f}",
            "Val Acc":    f"{val_accs[-1]*100:.2f}%",
            "Early Stop": f"✓ {early_stopper.best_loss:.4f}" if early_stopper.counter == 0 else f"✗ {early_stopper.counter}/{early_stopper.patience}",
        })

        progress_pct = (epoch + 1) / num_epochs * 100
        mo.output.replace(mo.vstack([
            mo.md(f"**Training... {epoch+1}/{num_epochs}** ({progress_pct:.0f}%)"),
            mo.ui.table(epoch_log),
        ]))

        if early_stopper.stop:
            break

    # Restore best weights
    model.load_state_dict(best_model_weights)
    mo.output.replace(mo.vstack([
        mo.md(f"**✓ Training complete** — best val_loss: `{early_stopper.best_loss:.4f}`"),
        mo.ui.table(epoch_log),
    ]))
    return train_accs, train_losses, val_accs, val_losses


@app.cell
def _(
    device,
    get_accuracy,
    mo,
    model,
    plt,
    test_loader,
    train_accs,
    train_loader,
    train_losses,
    val_accs,
    val_loader,
    val_losses,
):
    # Accuracy text
    lines = []
    for name, loader in [("Training", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
        acc = get_accuracy(loader, model, device)
        lines.append(f"**{name}:** {acc*100:.2f}%")

    epochs_range = range(1, len(train_losses) + 1)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Loss curves
    ax1.plot(epochs_range, train_losses, label='Train Loss')
    ax1.plot(epochs_range, val_losses,   label='Val Loss',   linestyle='--')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy curves
    ax2.plot(epochs_range, [a*100 for a in train_accs], label='Train Acc')
    ax2.plot(epochs_range, [a*100 for a in val_accs],   label='Val Acc',   linestyle='--')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    plt.tight_layout()

    mo.vstack([
        mo.md(" · ".join(lines)),
        mo.as_html(fig)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Custom image inference
    """)
    return


@app.cell
def _():
    import anywidget
    import traitlets
    import base64
    from PIL import Image, ImageFilter
    import io

    return Image, ImageFilter, anywidget, base64, io, traitlets


@app.cell
def _(anywidget, mo, traitlets):
    class DrawableCanvas(anywidget.AnyWidget):
        image_data = traitlets.Unicode('').tag(sync=True)

        _esm = """
        function render({ model, el }) {
            const canvas = document.createElement('canvas');
            canvas.width = 280;
            canvas.height = 280;
            canvas.style.border = '2px solid #888';
            canvas.style.cursor = 'crosshair';
            canvas.style.borderRadius = '8px';
            canvas.style.touchAction = 'none';

            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, 280, 280);
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 18;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            let drawing = false;

            function getPos(e) {
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                return [
                    (e.clientX - rect.left) * scaleX,
                    (e.clientY - rect.top) * scaleY
                ];
            }

            canvas.addEventListener('mousedown', (e) => {
                drawing = true;
                const [x, y] = getPos(e);
                ctx.beginPath();
                ctx.moveTo(x, y);
            });

            canvas.addEventListener('mousemove', (e) => {
                if (!drawing) return;
                const [x, y] = getPos(e);
                ctx.lineTo(x, y);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(x, y);
            });

            canvas.addEventListener('mouseup', () => { drawing = false; });
            canvas.addEventListener('mouseleave', () => { drawing = false; });

            const btnStyle = 'padding:8px 20px; cursor:pointer; border-radius:6px; font-size:14px; border:1px solid #666;';

            const clearBtn = document.createElement('button');
            clearBtn.textContent = '🗑 Clear';
            clearBtn.style.cssText = btnStyle;
            clearBtn.onclick = () => {
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, 280, 280);
                model.set('image_data', '');
                model.save_changes();
            };

            const captureBtn = document.createElement('button');
            captureBtn.textContent = '🔍 Run Inference';
            captureBtn.style.cssText = btnStyle + 'font-weight:bold;';
            captureBtn.onclick = () => {
                model.set('image_data', canvas.toDataURL('image/png'));
                model.save_changes();
            };

            const container = document.createElement('div');
            container.style.cssText = 'display:flex; flex-direction:column; align-items:center; gap:12px;';

            const btnRow = document.createElement('div');
            btnRow.style.cssText = 'display:flex; gap:10px;';
            btnRow.appendChild(clearBtn);
            btnRow.appendChild(captureBtn);

            container.appendChild(canvas);
            container.appendChild(btnRow);
            el.appendChild(container);
        }
        export default { render };
        """

    canvas_widget = mo.ui.anywidget(DrawableCanvas())

    mo.vstack([
        mo.md("#### ✏️ Draw a Digit"),
        canvas_widget
    ])
    return (canvas_widget,)


@app.cell
def _(
    F,
    Image,
    ImageFilter,
    base64,
    canvas_widget,
    device,
    io,
    mo,
    model,
    np,
    plt,
    torch,
    transforms,
):
    # Get the base64 image data synced from JavaScript
    raw_data = canvas_widget.value['image_data']

    mo.stop(not raw_data, mo.md("*Draw a digit and click **Run Inference**.*"))

    # --- Decode base64 PNG → PIL Image ---
    base64_str = raw_data.split(',')[1]           # strip "data:image/png;base64,"
    img_bytes = base64.b64decode(base64_str)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert('L')  # grayscale

    # --- Preprocess to match MNIST ---
    img_blurred = img_pil.filter(ImageFilter.GaussianBlur(1))  # smooth jagged mouse strokes
    img_28 = img_blurred.resize((28, 28), Image.LANCZOS)       # downsample
    img_tensor = transforms.ToTensor()(img_28)                  # [1, 28, 28], normalized [0,1]

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        _x = img_tensor.to(device)
        _scores = model(_x)
        probs = F.softmax(_scores, dim=1).squeeze().cpu().numpy()
        predicted = int(probs.argmax())
        confidence = probs[predicted] * 100

    # --- Display ---
    figure, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(14, 4))

    # Your drawing (280×280)
    _ax1.imshow(np.array(img_pil), cmap='gray')
    _ax1.set_title("Your Drawing (280×280)")
    _ax1.axis('off')

    # What model sees (28×28)
    _ax2.imshow(np.array(img_28), cmap='gray')
    _ax2.set_title("Model Input (28×28)")
    _ax2.axis('off')

    # Softmax probabilities
    colors = ['tomato' if i == predicted else 'steelblue' for i in range(10)]
    bars = _ax3.bar(range(10), probs, color=colors)
    _ax3.set_xticks(range(10))
    _ax3.set_xticklabels([str(i) for i in range(10)])
    _ax3.set_xlabel('Digit')
    _ax3.set_ylabel('Confidence')
    _ax3.set_title(f"Predicted: {predicted}  |  {confidence:.1f}%")
    _ax3.set_ylim(0, 1)

    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            _ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{prob:.2f}',
                ha='center', va='bottom', fontsize=8
            )

    plt.tight_layout()
    # mo.as_html(figure)
    mo.vstack([
        mo.md("#### Results"),
        mo.as_html(figure)
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
