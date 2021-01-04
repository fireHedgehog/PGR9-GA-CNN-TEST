import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)

    pop_size = 20
    generations = 100
    CROSS_RATE = 0.5
    MUTATION_RATE = 0.01

    elite_pool_1 = []
    elite_pool_2 = []


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)  # 28 * 28 -> (28+1-5) 24 * 24
            self.conv2 = nn.Conv2d(20, 50, 5, 1)  # 20 * 20
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            # x: 1 * 28 * 28
            x = F.relu(self.conv1(x))  # 20 * 24 * 24
            x = F.max_pool2d(x, 2, 2)  # 12 * 12
            x = F.relu(self.conv2(x))  # 8 * 8
            x = F.max_pool2d(x, 2, 2)  # 4 *4
            x = x.view(-1, 4 * 4 * 50)  # reshape (5 * 2 * 10), view(5, 20) -> (5 * 20)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            # return x
            return F.log_softmax(x, dim=1)  # log probability


    mnist_data = datasets.MNIST("./mnist_data", train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))

    data = [d[0].data.cpu().numpy() for d in mnist_data]

    print(np.mean(data))
    print(np.std(data))


    def train(model, device, train_loader, epoch):

        for num_gen in range(generations):

            weights_1_pop = []
            weights_2_pop = []

            if num_gen == 0:
                """"
                if it is first generation
                then initialize with a random array
                """
                weights_1_pop = [np.random.uniform(low=-1, high=1, size=(20, 1, 5, 5)) for i in range(20)]
                weights_2_pop = [np.random.uniform(low=-1, high=1, size=(20, 1, 5, 5)) for i in range(20)]
            else:
                """"
                if it is not first generation
                then, run a selection function to generate parents
                and also run GA operators
                """
                print(num_gen)
                for idx_individual in range(pop_size):
                    print(num_gen)

        model.train()
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            conv_1_cur_gen = []
            conv_2_cur_gen = []

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'conv1.weight' == name:
                        conv_1_cur_gen = np.random.uniform(low=-1, high=1, size=(20, 1, 5, 5))
                        p.copy_(torch.tensor(conv_1_cur_gen))
                    if 'conv2.weight' == name:
                        conv_2_cur_gen = np.random.uniform(low=-1, high=1, size=(20, 1, 5, 5))
                        p.copy_(torch.tensor(conv_1_cur_gen))

            pred = model(data)
            loss = F.nll_loss(pred, target)

            update_population(elite_pool_1, num_gen, idx_individual, loss, conv_1_cur_gen, pop_size)
            update_population(elite_pool_2, num_gen, idx_individual, loss, conv_2_cur_gen, pop_size)

            if idx % 100 == 0:
                print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                    epoch, idx, loss.item()))


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


    def mnist_test():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(torch.cuda.is_available())
        print(device)

        batch_size = 32
        train_dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./mnist_data",
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./mnist_data",
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )

        model = Net().to(device)

        num_epochs = 1
        for epoch in range(num_epochs):
            train(model, device, train_dataloader, epoch)
            test(model, device, test_dataloader)

        torch.save(model.state_dict(), "ga_mnist_cnn.pt")


    def update_function(param, grad, iter):
        np_arr = param.cpu().detach().numpy()
        random = np.empty((20, 1, 5, 5))

        """
        if iter % 1000 == 0:
            print("---------------------------- start  ----------------------------")
            print(np_arr.shape)
            print("")
            print(random.shape)
            print("----------------------------- end  -----------------------------")
        """
        return torch.tensor(np_arr)


    def update_population(pool, num_gen, idx_individual, loss, weights_arr, size):
        # if empty, then or individuals not enough
        if not pool or len(pool) < size:
            pool.append({
                'generation': num_gen,
                'individual': idx_individual,
                'loss': loss,
                'weights': weights_arr,
            })
        else:
            # check if the new weights has better loss value
            pool = sorted(pool, key=lambda x: x['loss'], reverse=True)
            if loss < pool[0]['loss']:
                pool[0] = {
                    'generation': num_gen,
                    'individual': idx_individual,
                    'loss': loss,
                    'weights': np.array([
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ]),
                }

        return pool


    mnist_test()
