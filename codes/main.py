from load_data import get_dataloaders

train_loader, val_loader = get_dataloaders(
    root_dir="C:/Users/Gayatri/Documents/task/data",
    batch_size=32,
    img_size=(112, 112)
)

