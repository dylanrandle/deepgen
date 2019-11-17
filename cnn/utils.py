import torchvision
import torchvision.transforms as transforms
import torch

def load_celebA(data_path='~', batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor()])

    cba_train = torchvision.datasets.CelebA(data_path, download=True,
        split='train', transform=transform, target_transform=lambda x: x.float())
    cba_test = torchvision.datasets.CelebA(data_path, download=True,
        split='test', transform=transform, target_transform=lambda x: x.float())

    train_loader = torch.utils.data.DataLoader(cba_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(cba_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def print_attrs(attr, celebA_dataset):
    """
    human-readable format for CelebA attributes vector
    """
    legible = ''
    for i, a in enumerate(list(attr)):
        if a == 1:
            legible += celebA_dataset.attr_names[i] + ", "
    return legible
