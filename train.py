from initing import *
from dataset import get_dataloaders
from model import make_model



def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(outputs, labels) * bs
        n += bs
    return total_loss/n, total_acc/n

def validate(model, loader, criterion):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy_from_logits(outputs, labels) * bs
            n += bs
    return total_loss/n, total_acc/n

# Block 6: Training Loop + Schedulers
def choose_optimizer(model, opt_name="sgd_nesterov", lr=BASE_LR):
    if opt_name == "sgd_nesterov":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=MOMENTUM, nesterov=True,
                               weight_decay=WEIGHT_DECAY)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(opt_name)

def choose_scheduler(optimizer, sched_name="ReduceLROnPlateau"):
    if sched_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.75, patience=5)
    elif sched_name == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6)
    else:
        return None

def fit(model, train_loader, val_loader, test_loader=None,
        optimizer_name="sgd_nesterov", scheduler_name="ReduceLROnPlateau",
        epochs=EPOCHS):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(model, optimizer_name, BASE_LR)
    scheduler = choose_scheduler(optimizer, scheduler_name)
    scaler = GradScaler()

    best_val = 0
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        elif scheduler: scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} | "
              f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | "
              f"Time {time.time()-t0:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"✅ Saved new best model (ValAcc={val_acc:.4f})")

    if test_loader:
        criterion = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE))
        test_loss, test_acc = validate(model, test_loader, criterion)
        print(f"\nFinal Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    return model




