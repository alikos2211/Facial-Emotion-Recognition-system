from initing import *
from dataset import get_dataloaders
from model import make_model
from train import fit
from train import train_one_epoch, validate

# Block 8: Fine-Tuning (optional small-LR retrain)
def fine_tune(model_path=os.path.join(OUTPUT_DIR, "best_model.pth"),
              lr=1e-4, extra_epochs=20):
    model = make_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=MOMENTUM, nesterov=True,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=extra_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    train_loader, val_loader, _ = get_dataloaders()

    for epoch in range(extra_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        print(f"[FT] Epoch {epoch+1}/{extra_epochs} | "
              f"TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f} | "
              f"Time {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "fine_tuned.pth"))
    print("Fine-tuned model saved.")
    return model