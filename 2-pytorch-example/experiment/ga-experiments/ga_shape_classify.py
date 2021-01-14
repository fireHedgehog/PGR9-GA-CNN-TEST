import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from GAOptimizer import GAOptimizer
import csv
from torch.utils.data import DataLoader, random_split
import numpy as np

if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)


    class Net(nn.Module):
        input_size = [28, 28]
        output_size = 3
        input_channels = 1
        channels_conv1 = 9
        channels_conv2 = 18
        kernel_conv1 = (3, 3)
        kernel_conv2 = (3, 3)
        pool_conv1 = (2, 2)
        pool_conv2 = [2, 2]
        fcl1_size = 50

        def __init__(self):
            super(Net, self).__init__()

            # Define the convolutional layers
            self.conv1 = nn.Conv2d(in_channels=self.input_channels,
                                   out_channels=self.channels_conv1,
                                   kernel_size=self.kernel_conv1
                                   )
            self.conv2 = nn.Conv2d(in_channels=self.channels_conv1,
                                   out_channels=self.channels_conv2,
                                   kernel_size=self.kernel_conv2)

            # Calculate the convolutional layers output size (stride = 1)
            c1 = np.array(self.input_size) - np.array(self.kernel_conv1) + 1
            p1 = c1 // self.pool_conv1[0]
            c2 = p1 - np.array(self.kernel_conv2) + 1
            p2 = c2 // self.pool_conv2[0]
            self.conv_out_size = int(p2[0] * p2[1] * self.channels_conv2)

            # Define the fully connected layers
            self.fcl1 = nn.Linear(self.conv_out_size, self.fcl1_size)
            self.fcl2 = nn.Linear(self.fcl1_size, self.output_size)

        def forward(self, x):
            # Apply convolution 1 and pooling
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, self.pool_conv1)

            # Apply convolution 2 and pooling
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, self.pool_conv2)

            # Reshape x to one dimmension to use as input for the fully connected layers
            x = x.view(-1, self.conv_out_size)

            # Fully connected layers
            x = self.fcl1(x)
            x = F.relu(x)
            x = self.fcl2(x)

            return F.log_softmax(x, dim=1)


    # net = Net()
    # print(net)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()

        for idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                _pred = model(data)
                _loss = F.nll_loss(_pred, target)
                if _loss.requires_grad:
                    _loss.backward()
                return _loss

            loss = optimizer.step(closure=closure, iteration_index=idx)

            print(
                "Train Epoch: {}, iteration: {}, Loss: {}".format(epoch, idx, loss.item())
            )

            with open('ga_opt_Line_history.csv', mode='a') as history_file:
                history_writer = csv.writer(history_file,
                                            delimiter=',',
                                            quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL
                                            )
                history_writer.writerow([epoch, idx, loss.item()])

            if idx % 100 == 0:
                print(
                    "Train Epoch: {}, iteration: {}, Loss: {}".format(epoch, idx, loss.item())
                )

            if loss.item() < 0.010:
                break


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

        # print(torch.cuda.is_available())
        # print(device)

        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.9464835,), (0.1556641,))
        ])

        dataset = datasets.ImageFolder(root='../data/lines/',
                                       transform=data_transform)

        trainset, valset = random_split(dataset, [170, 50])

        train_dataloader = DataLoader(trainset,
                                      batch_size=10,
                                      shuffle=True,
                                      num_workers=1,
                                      # pin_memory=True,
                                      )

        test_dataloader = DataLoader(valset,
                                     batch_size=10,
                                     shuffle=True,
                                     num_workers=1,
                                     # pin_memory=True,
                                     )

        model = Net().to(device)
        optimizer = GAOptimizer(
            params=model.parameters(),
            generation_size=200,
            pop_size=100,
            mutation_rate=0.65,
            crossover_rate=0.65,
            elite_rate=0.10,
            new_chromosome_rate=0.10,
            weights_val_bit=4,
            weights_upper_lower_range=3599,
            save_csv_files=True
        )

        train(model, device, train_dataloader, optimizer, 0)
        test(model, device, test_dataloader)

        torch.save(model.state_dict(), "LINES_GA_cnn.pt")


    shape_class_test()
