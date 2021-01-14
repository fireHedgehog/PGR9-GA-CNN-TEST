import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from GAOptimizer import GAOptimizer
import csv
from ShapeDataSet import ShapeDataset
from torch.utils.data import DataLoader, random_split
import numpy as np

if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x, dim=1)


    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            pred = model(data)  # batch_size * 10
            loss = F.nll_loss(pred, target)
            # SGD
            optimizer.zero_grad()
            loss.backward()

            # def closure():
            #     if torch.is_grad_enabled():
            #         optimizer.zero_grad()
            #     _pred = model(data)
            #     _loss = F.nll_loss(_pred, target)
            #     if _loss.requires_grad:
            #         _loss.backward()
            #     return _loss
            #
            # loss = optimizer.step(closure=closure)

            print(
                "Train Epoch: {}, iteration: {}, Loss: {}".format(epoch, idx, loss.item())
            )
            with open('ga_opt_shape_history.csv', mode='a') as employee_file:
                history_writer = csv.writer(employee_file,
                                            delimiter=',',
                                            quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL
                                            )
                history_writer.writerow([epoch, idx, loss.item()])
            if idx % 100 == 0:
                print(
                    "Train Epoch: {}, iteration: {}, Loss: {}".format(epoch, idx, loss.item())
                )


    def test(model, device, test_loader):
        model.eval()
        total_loss = 0.
        correct = 0.
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)  # batch_size * 10
                total_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1)  # batch_size * 1
                correct += pred.eq(target.view_as(pred)).sum().item()

        total_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset) * 100.
        print("Test loss: {}, Accuracy: {}".format(total_loss, acc))


    def shape_class_test():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(torch.cuda.is_available())
        print(device)

        dataset = ShapeDataset('data/shapes/')

        trainset, valset = random_split(dataset, [250, 50])

        train_dataloader = DataLoader(trainset,
                                      batch_size=10,
                                      shuffle=True,
                                      num_workers=2
                                      )
        test_dataloader = DataLoader(valset,
                                     batch_size=10,
                                     shuffle=True,
                                     num_workers=2,
                                     )

        lr = 0.01
        momentum = 0.5
        model = Net().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        num_epochs = 2
        for epoch in range(num_epochs):
            train(model, device, train_dataloader, optimizer, epoch)
            test(model, device, test_dataloader)

        # model = Net().to(device)
        # optimizer = GAOptimizer(model.parameters())
        #
        # num_epochs = 5
        # for epoch in range(num_epochs):
        #     train(model, device, train_dataloader, optimizer, epoch)
        #     test(model, device, test_dataloader)

        torch.save(model.state_dict(), "shapes_cnn.pt")


    shape_class_test()
