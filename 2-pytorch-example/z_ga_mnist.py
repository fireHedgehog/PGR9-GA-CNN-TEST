import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random

if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)

    pop_size = 1
    generations = 1

    """
    Genetic Operators
    ___________________________________________________________________________________
    crossover: 
    ___________________________________________________________________________________
        parent 1:                                parent 2:
        
        [0, 0, 0, 0],                            [1, 1, 1, 1],
        [0, 0, 0, 0],                            [1, 1, 1, 1],
        [0, 0, 0, 0],      <=== crossover ====>  [1, 1, 1, 1],
        [0, 0, 0, 0],                            [1, 1, 1, 1],
        [0, 0, 0, 0],                            [1, 1, 1, 1],
        
        
          
        offspring 1:                             offspring 2:
         
        [0, 0, 0, 0],                            [1, 1, 1, 1],
        [0, 0, 0, 0],                            [1, 1, 1, 1],
        [0, 0, 0, 0],                            [1, 1, 1, 1],
        [1, 1, 1, 1],                            [0, 0, 0, 0],
        [1, 1, 1, 1],                            [0, 0, 0, 0],
    ___________________________________________________________________________________
    Mutations: 
    ___________________________________________________________________________________
    
    """
    CROSS_RATE = 0.5
    MUTATION_RATE = 0.1

    # can set to zero if not using elite selection function
    elite_percent = 0.1  # defining the max number of elite individuals

    # can set to zero if new random generations is not needed
    new_chromosome_percent = 0.1  # defining the max number of new random individuals

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


    mnist_data = datasets.MNIST("./mnist_data",
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))


    # data = [d[0].data.cpu().numpy() for d in mnist_data]
    #
    # # get mean and std to normalize it
    # print(np.mean(data))
    # print(np.std(data))

    def train(model, device, train_loader):

        global elite_pool_1
        global elite_pool_2

        # disable gradiant descent
        # with torch.no_grad():
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # need to run GA in every batch
            for num_gen in range(generations):
                weights_1_pop = []  # current gen
                weights_2_pop = []  # current gen

                new_conv1_pop = []  # for next gen
                new_conv2_pop = []  # for next gen

                if not elite_pool_1 or not elite_pool_2:
                    """
                    if it is first generation
                    then initialize with a random array
                    """
                    weights_1_pop = [np.random.uniform(low=-1, high=1, size=(20, 1, 5, 5)) for i in range(pop_size)]
                    weights_2_pop = [np.random.uniform(low=-1, high=1, size=(50, 20, 5, 5)) for i in
                                     range(pop_size)]
                else:
                    """
                    if it is not first generation
                    then, run a [!--selection function--!] to generate parents
                    and also run [!--GA operators--!]
                    """
                    # elite_percent new_chromosome_percent
                    pool_1 = sorted(elite_pool_1, key=lambda x: x['loss'])
                    pool_2 = sorted(elite_pool_2, key=lambda x: x['loss'])

                    num_elite = round(pop_size * elite_percent)
                    num_new = round(pop_size * new_chromosome_percent)

                    num_matate_crossover_pool = pop_size - num_elite - num_new

                    for idx_elite in range(num_elite):
                        weights_1_pop.append(pool_1[idx_elite]["weights"])
                        weights_2_pop.append(pool_2[idx_elite]["weights"])

                    for idx_new in range(num_new):
                        weights_1_pop.append(np.random.uniform(low=-1, high=1, size=(20, 1, 5, 5)))
                        weights_2_pop.append(np.random.uniform(low=-1, high=1, size=(50, 20, 5, 5)))

                    for idx_previous in range(num_matate_crossover_pool):
                        weights_1_pop.append(ga_operation(pool_1))
                        weights_2_pop.append(ga_operation(pool_2))

                best_loss = 99999
                best_loss_idx = 99999
                # then, save the new weights in this model
                # iterate chromosome in this population
                for idx_chromosome in range(pop_size):
                    conv_1_cur_gen = weights_1_pop[idx_chromosome]
                    conv_2_cur_gen = weights_2_pop[idx_chromosome]

                    with torch.no_grad():
                        params = model.named_parameters()
                        for name, p in params:
                            if 'conv1.weight' == name:
                                p.copy_(torch.tensor(conv_1_cur_gen))
                            elif 'conv2.weight' == name:
                                p.copy_(torch.tensor(conv_2_cur_gen))

                        pred = model(data)
                        loss = F.nll_loss(pred, target)

                        # loss.backward()

                        loss_val = loss.item()

                        if loss_val < best_loss:
                            best_loss = loss_val
                            best_loss_idx = idx_chromosome

                        new_conv1_pop.append({
                            'index': idx,
                            'generation': num_gen,
                            'individual': idx_chromosome,
                            'loss': loss_val,
                            'inverse_loss': 999999999 if not loss_val else 1 / loss_val,
                            'weights': conv_1_cur_gen,
                        })

                        new_conv2_pop.append({
                            'index': idx,
                            'generation': num_gen,
                            'individual': idx_chromosome,
                            'loss': loss_val,
                            'inverse_loss': 999999999 if not loss_val else 1 / loss_val,
                            'weights': conv_2_cur_gen,
                        })

                        if idx % 100 == 0:
                            print("Generation: {},Chromosome Idx: {} , iteration: {}, Loss: {}"
                                  .format(num_gen, idx_chromosome, idx, loss_val))

                with open('ga_mnist_history.csv', mode='a') as employee_file:
                    history_writer = csv.writer(employee_file,
                                                delimiter=',',
                                                quotechar='"',
                                                quoting=csv.QUOTE_MINIMAL
                                                )
                    history_writer.writerow([num_gen, best_loss_idx, idx, best_loss])

                for idx_chromosome in range(pop_size):
                    elite_pool_1 = update_population(elite_pool_1, new_conv1_pop[idx_chromosome], pop_size)
                    elite_pool_2 = update_population(elite_pool_2, new_conv2_pop[idx_chromosome], pop_size)


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
                    # transforms.Normalize((0.1307,), (0.3081,))
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
                    # transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )

        model = Net().to(device)

        # num_epochs = 1
        # for epoch in range(num_epochs):
        #     train(model, device, train_dataloader, epoch)
        #     test(model, device, test_dataloader)

        train(model, device, train_dataloader)
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


    def update_population(pool, data_, size):
        # if empty, then or individuals not enough
        if not pool or len(pool) < size:
            pool.append({
                'index': data_["index"],
                'generation': data_["generation"],
                'individual': data_["individual"],
                'loss': data_["loss"],
                'inverse_loss': data_["inverse_loss"],
                'weights': data_["weights"],
            })
        else:
            # check if the new weights has better loss value
            pool = sorted(pool, key=lambda x: x['loss'], reverse=True)
            if data_["loss"] < pool[0]['loss']:
                pool[0] = {
                    'index': data_["index"],
                    'generation': data_["generation"],
                    'individual': data_["individual"],
                    'loss': data_["loss"],
                    'inverse_loss': data_["inverse_loss"],
                    'weights': data_["weights"],
                }
        return pool


    def basic_selection(fitness_value):
        sorted_fitness = sorted(fitness_value, key=lambda x: x['inverse_loss'])
        cum_acc = np.array([e['inverse_loss'] for e in sorted_fitness]).cumsum()

        evaluation = [{
            'index': e['index'],
            'inverse_loss': e['inverse_loss'],
            'weights': e['weights'],
            'cum_acc': acc
        } for e, acc in zip(sorted_fitness, cum_acc)]

        rand = np.random.rand() * cum_acc[-1]

        for e in evaluation:
            if rand < e['cum_acc']:
                return e

        return evaluation[-1]


    def ga_operation(pool):

        next_individual = basic_selection(pool)['weights']

        if np.random.rand() > CROSS_RATE:
            parent_1 = basic_selection(pool)['weights']
            parent_2 = basic_selection(pool)['weights']

            offspring = np.zeros(parent_1.shape)
            for idx, parent in enumerate(parent_1):
                # a = np.array_split(parent_1, 2)
                # b = np.array_split(parent_2, 2)
                # offspring_1 = np.concatenate((a[0], b[1]))
                # offspring_2 = np.concatenate((b[0], a[1]))
                if np.random.rand() > 0.5:
                    offspring[idx] = parent
                else:
                    offspring[idx] = parent_2[idx]

            next_individual = offspring

        if np.random.rand() > MUTATION_RATE:
            next_individual = 1e-3 * np.random.rand() * np.random.normal(0,
                                                                         np.std(next_individual),
                                                                         next_individual.shape
                                                                         )

        return next_individual


    mnist_test()
