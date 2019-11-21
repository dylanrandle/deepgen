import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

def load_celebA(data_path='~', batch_size=64):
    """ helper function to load CelebA dataset using built-in pytorch methods """
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

def count_parameters(model):
    """ helper function to count trainable parameters in a pytorch model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_examples(img, gen_img, save_to):
    """ plot real and generated images, then save them """
    batch_size = img.shape[0]
    num_examples = 5 if batch_size >= 5 else batch_size # default to 5 examples

    fig, ax = plt.subplots(2, num_examples, figsize=(20,6))
    ax = ax.reshape(2, -1)
    for j, _ax in enumerate(ax[0, :]):
        _ax.imshow(img[j, :, :, :].permute(1, 2, 0).cpu())
        _ax.set_title('Original')
    for j, _ax in enumerate(ax[1, :]):
        _ax.imshow(gen_img[j, :, :, :].permute(1, 2, 0).cpu().detach())
        _ax.set_title('Generated')
    fig.tight_layout()

    plt.savefig(save_to)
    print(f'Saved example to {save_to}')
    plt.close(fig)
