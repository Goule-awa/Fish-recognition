import torch
from torch import optim
from torchvision import transforms, datasets
from model.fishnet import FishNet

# ������ǿ
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# �������ݼ�
train_data = datasets.ImageFolder('data/train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# ��ʼ��ģ��
model = FishNet(num_classes=15)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ѵ��ѭ��
for epoch in range(20):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # �������ģ��
    torch.save(model.state_dict(), 'model/best_model.pth')