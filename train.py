from turtle import forward
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from model import*
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


writer = SummaryWriter("logs_train")
tudui = Tudui()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr = learning_rate)

total_train_step = 0

total_test_step = 0

epoch = 10


for i in range(epoch):
    print(f"---------第{i+1}轮训练开始-------------")

    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
            total_test_step += 1
        print(f"整体测试集上的Loss: {total_test_loss.item()}")
        print(f"整体测试集上的正确率：{total_accuracy / test_data_size}")
        print(f"平均测试集上的正确率")
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
