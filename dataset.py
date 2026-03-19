from initing import *
from torchvision.transforms import RandomErasing
import datasets
from torchvision import datasets



class ImageFolderGray(Dataset):
    """Reads grayscale images from folder layout."""
    def __init__(self, root, transform=None):
        self.samples = []
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for c in self.classes:
            for f in (Path(root) / c).glob("*"):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    self.samples.append((str(f), self.class_to_idx[c]))
        self.transform = transform

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("L")
        if self.transform: img = self.transform(img)
        return img, label

# ==== Augmentations ====
train_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=10, translate=(0.2,0.2), scale=(0.8,1.2))
    ], p=0.5),
    transforms.RandomCrop(40),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
    RandomErasing(p=0.5)
])

val_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.CenterCrop(40),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])


train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_ds = ImageFolderGray(train_dir, transform=train_transform)
    val_ds   = ImageFolderGray(test_dir,  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = val_loader
    return train_loader, val_loader, test_loader

# Quick sanity check
train_loader, val_loader, test_loader = get_dataloaders()
#print(f"Train: {len(train_loader)} | Val: {len(val_loader)}")

