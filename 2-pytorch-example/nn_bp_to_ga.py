import torch
import numpy as np
import torch.nn as nn
import random

dtype = torch.float
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object

print(torch.cuda.is_available())
print(device)

np.random.uniform(low=-1, high=1, size=(20, 1, 5, 5))
torch.FloatTensor(20, 1, 5, 5).uniform_(-1, 1)
elite_pool_1 = [np.random.uniform(low=-1, high=1, size=(20, 1, 5, 5)) for i in range(10)]

# for idx, data in enumerate(elite_pool_1):
#     # print(idx, data.shape)
#     print(idx, data[19][0])


# elite_pool_2 = [nn.Conv2d(1, 20, 5, 1) for i in range(20)]
# for idx, data in enumerate(elite_pool_2):
#     # print(idx, data.shape)
#     print(idx, data)

# elite_pool_2 = [nn.Conv2d(20, 50, 5, 1) for i in range(20)]
# for idx, data in enumerate(elite_pool_2):
#     # print(idx, data.shape)
#     print(idx, data)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w_1 = torch.randn(D_in, H, device=device, dtype=dtype)
w_2 = torch.randn(H, D_out, device=device, dtype=dtype)


def bp_nn(w1, w2):
    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def split_concat_array():
    # simulate crossover
    array_1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    array_2 = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])

    print(array_1.shape)
    print(array_2.shape)

    a = np.array_split(array_1, 2)
    b = np.array_split(array_2, 2)

    print(a[0])
    print(b[0])

    c = np.concatenate((a[0], b[1]))
    d = np.concatenate((b[0], a[1]))

    print(c)
    print(d)

    print(np.random.rand() * np.random.normal(0, np.std(c), c.shape))


def pop_pool_test():
    all_parents = []

    for i in range(10):
        loss = np.random.rand()
        all_parents.append({
            'iter': i,
            'loss': loss,
            'inverse_loss': 999999999 if not loss else 1 / loss,
            'weights': np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]),
        })

    sorted_arr = sorted(all_parents, key=lambda x: x['loss'], reverse=True)
    print('\nIter: {} , loss: {}'.format(sorted_arr[-1]['iter'], all_parents[-1]['loss']))

    print(sorted_arr)

    loss_new = np.random.rand()

    # if the new loss is better than old one
    # then replace it
    if loss_new < sorted_arr[0]['loss']:
        print(1)
        sorted_arr[0] = {
            'iter': 11,
            'loss': loss_new,
            'inverse_loss': 999999999 if not loss_new else 1 / loss_new,
            'weights': np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]),
        }
        print(sorted_arr)

    selected = []
    for idx in range(5):
        selected.append(selection_test(sorted_arr))

    print("------selected : \n", selected)


def selection_test(fitness_value):
    sorted_fitness = sorted(fitness_value, key=lambda x: x['inverse_loss'])
    cum_acc = np.array([e['inverse_loss'] for e in sorted_fitness]).cumsum()

    evaluation = [{
        'iter': e['iter'],
        'inverse_loss': e['inverse_loss'],
        'cum_acc': acc
    } for e, acc in zip(sorted_fitness, cum_acc)]

    rand = np.random.rand() * cum_acc[-1]

    for e in evaluation:
        if rand < e['cum_acc']:
            return e

    return evaluation[-1]


# pop_pool_test()

split_concat_array()

# bp_nn(w1=w_1, w2=w_2)

# print(np.empty((20, 1, 3, 3)))
