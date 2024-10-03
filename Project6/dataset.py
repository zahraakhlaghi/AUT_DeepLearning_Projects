from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def get_emoji_loader(train_path, test_path, batch_size, image_size):
    transform = transforms.Compose([
                    transforms.Scale(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dloader, test_dloader