from dataset import get_dataloaders
from model import make_model
from train import fit


if __name__ == "__main__":
   
    train_loader, val_loader, test_loader = get_dataloaders()
    model = make_model()
    model = fit(model, train_loader, val_loader, test_loader,
            optimizer_name="sgd_nesterov",
            scheduler_name="ReduceLROnPlateau",
            epochs=50)
    