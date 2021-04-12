import numpy as np
import torch
from torch.optim import Optimizer
import csv


def _get_random_weights_by_shape(shape, bit, upper_lower_range):
    # generate new random weights by the shape of params

    # # obsolete:　return np.random.uniform(low=-1, high=1, size=shape)

    if not np.random.rand() > 0.5:
        bit = bit + 1

    if not shape:
        return np.random.randint(-upper_lower_range, upper_lower_range) / pow(10, bit)
    else:
        return np.random.randint(-upper_lower_range, upper_lower_range, shape) / pow(10, bit)


def _search_best_performed_weights(population, p_index):
    # find best weights values

    for i, obj in enumerate(population):
        if obj['p_id'] == p_index:
            return obj["p_value"]


def _ga_operation(weights_pool, crossover_rate, mutation_rate, bit, upper_lower_range, idx_previous):
    """
    ____________________________________________________________________
        crossover by flatten array

            parent_1: [ a1, b1, c1, d1, e1, f1 ]
                   - crossover -
            parent_2: [ a2, b2, c2, d2, e2, f2 ]

            if rand()>0.5 take parent_1's value, otherwise take parent_2's value

            offspring may be: [ a1, b1, c2, d2, e1, f2 ]
    ____________________________________________________________________
            mutating by flatten array

            parent: [ a, b, c, d, e, f ]

            if rand()>0.5 take original value, otherwise take mutated random value

            offspring may be: [ a, b', c', d, e', f ]
    ____________________________________________________________________
    """
    # use roulette_selection() to select an individual
    # if this turn no need to crossover/mutation
    # then just return this to pool
    next_individual = _roulette_selection(weights_pool)['param_list']

    p_cross = np.random.rand()
    p_mutate = np.random.rand()

    # check if it need to crossover
    if p_cross > crossover_rate or p_mutate > mutation_rate:
        # initialize empty offspring value
        offspring_arr = []

        # crossover need 2 parent
        another_parent = _roulette_selection(weights_pool)['param_list']

        # iterate parent params, conv weights, fully connected weights and ...
        for idx, parent_1 in enumerate(next_individual):

            # convert both to flat array
            flatten_param_parent_1 = parent_1['p_value'].flatten()
            flatten_param_parent_2 = another_parent[idx]['p_value'].flatten()

            # initialize offspring
            offspring = []
            for j, element_1 in enumerate(flatten_param_parent_1):
                # ____________________________________________
                # check if it needs crossover or not
                # ____________________________________________
                value_by_index = element_1  # take parent_1's as default
                if p_cross > crossover_rate:
                    if np.random.rand() > crossover_rate:
                        # take parent_2's weights
                        value_by_index = flatten_param_parent_2[j]

                # ____________________________________________
                # check if it needs mutation or not
                # ____________________________________________
                if p_mutate > mutation_rate:
                    if np.random.rand() > mutation_rate:
                        # random value
                        value_by_index = _get_random_weights_by_shape(shape=False,
                                                                      bit=bit,
                                                                      upper_lower_range=upper_lower_range
                                                                      )

                offspring.append(value_by_index)

            # use original shape to reshape it
            offspring_arr.append({"p_id": idx, "p_value": np.array(offspring).reshape(parent_1['p_value'].shape)})

        next_individual = offspring_arr

    return next_individual


def _roulette_selection(sorted_fitness):
    # roulette selection

    cum_acc = np.array([e['inverse_loss'] for e in sorted_fitness]).cumsum()

    evaluation = [{
        'gen_num': e['gen_num'],
        'pop_num': e['pop_num'],
        'param_list': e['param_list'],
        'loss_val': e['loss_val'],
        'inverse_loss': e['inverse_loss'],
        'cum_acc': acc
    } for e, acc in zip(sorted_fitness, cum_acc)]

    rand = np.random.rand() * cum_acc[-1]

    for e in evaluation:
        if rand < e['cum_acc']:
            return e

    return evaluation[-1]


def _update_population(pool, new_pool, size):
    # replace param by loss value
    sorted_new_pool = sorted(new_pool, key=lambda x: x['loss_val'])
    if not new_pool:
        sorted_new_pool = sorted_new_pool[:size / 2]
    for idx, value in enumerate(sorted_new_pool):
        pool = sorted(pool, key=lambda x: x['loss_val'])
        # check if the new weights has better loss value
        #   than the worst performed one in previous population
        if value["loss_val"] < pool[-1]['loss_val']:
            pool[-1] = value

    return sorted(pool, key=lambda x: x['loss_val'])


class GAOptimizer(Optimizer):
    r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups


        weights_val_bit and weights_upper_lower_range
          define the bit and density of weights values
          for example:
                 when weights_upper_lower_range = 999 and  weights_val_bit = 6
                 so, weights values will be a random number from [-0.00999 , 0.00999]

              example 2:
                 when weights_upper_lower_range = 9999 and  weights_val_bit = 5
                 so, weights values will be a random number from [-0.09999 , 0.09999]


    """

    def __init__(self,
                 params,
                 generation_size=20,
                 pop_size=20,
                 mutation_rate=0.6,
                 crossover_rate=0.6,
                 elite_rate=0.10,
                 new_chromosome_rate=0.10,
                 weights_val_bit=3,
                 weights_upper_lower_range=999,
                 save_csv_files=False
                 ):

        # NOTE: state management is same like LBFGS class

        defaults = dict(
            generation_size=generation_size,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_rate=elite_rate,
            new_chromosome_rate=new_chromosome_rate,
            weights_val_bit=weights_val_bit,
            weights_upper_lower_range=weights_upper_lower_range,
            save_csv_files=save_csv_files,
        )

        super(GAOptimizer, self).__init__(params, defaults)

        self.population = []
        self._params = self.param_groups[0]['params']

    # def __setstate__(self, state):
    #     super(GAOptimizer, self).__setstate__(state)
    #     for group in self.param_groups:
    #         group.setdefault('population', [])

    @torch.no_grad()
    def step(self, closure=None, iteration_index=0):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        group = self.param_groups[0]

        generation_size = group['generation_size']
        pop_size = group['pop_size']
        mutation_rate = group['mutation_rate']
        crossover_rate = group['crossover_rate']
        elite_rate = group['elite_rate']
        new_chromosome_rate = group['new_chromosome_rate']
        bit = group['weights_val_bit']
        upper_lower_range = group['weights_upper_lower_range']
        save_csv_files = group['save_csv_files']

        params = group['params']

        # iterate generations
        for gen_num in range(generation_size):
            # check if GA population pool is initialized or not
            if not self.population:
                initial_weights_pool = []
                # if not been initialized, need to generate random values by param type and population size
                for pop_num in range(pop_size):

                    param_list = []
                    for p_index, p in enumerate(params):
                        # if p.grad is None:  # if grad is turned off, this param may not need to be turned
                        #     continue

                        # initialize a random weights array
                        random_p_value = _get_random_weights_by_shape(p.cpu().detach().numpy().shape,
                                                                      bit,
                                                                      upper_lower_range)

                        # copy this random value as torch tensor
                        p.copy_(torch.tensor(random_p_value))

                        # append p_id and p_value to initialize population pool
                        param_list.append({
                            "p_id": p_index,
                            "p_value": random_p_value,
                        })

                    # calculate loss value of this individual
                    loss = closure()
                    loss_val = loss.item()
                    initial_weights_pool.append({
                        # number of this generation, for debugging purpose
                        'gen_num': gen_num,
                        # number of this individual in this population, for debugging purpose
                        'pop_num': pop_num,
                        # param values of this population in this generation
                        'param_list': param_list,
                        # loss_value
                        'loss_val': loss_val,
                        # inverse of loss_value -> easy to compute roulette selection
                        'inverse_loss': 9999999 if not loss_val else 100 / loss_val,
                    })

                # ascent sort. first element is the best performed individual
                sorted_params = sorted(initial_weights_pool, key=lambda x: x['loss_val'])
                self.population = sorted_params

            else:
                # initialize the number of elites, new chromosomes, and mutation/crossover pool
                num_elite = round(pop_size * elite_rate)
                num_new_individual = round(pop_size * new_chromosome_rate)
                num_ga_operation_pool = pop_size - num_elite - num_new_individual

                sorted_params = sorted(self.population, key=lambda x: x['loss_val'])

                next_weights_pool = []
                for idx_elite in range(num_elite):
                    # add elite by elite rate, if set to zero means not elite selection
                    next_weights_pool.append(sorted_params[idx_elite]["param_list"])

                for idx_new in range(num_new_individual):
                    # add new random params, can be zero as well
                    param_list = []
                    for p_index, p in enumerate(params):
                        # if p.grad is None:  # if grad is turned off, this param may not need to be turned
                        #     continue

                        # initialize a random weights array
                        random_p_value = _get_random_weights_by_shape(p.cpu().detach().numpy().shape,
                                                                      bit,
                                                                      upper_lower_range)

                        param_list.append({
                            "p_id": p_index,
                            "p_value": random_p_value,
                        })

                    next_weights_pool.append(param_list)

                # iterate by population size
                for idx_previous in range(num_ga_operation_pool):
                    # run roulette selection function
                    # then also run crossover and mutation function
                    next_weights_pool.append(
                        _ga_operation(sorted_params,
                                      crossover_rate,
                                      mutation_rate,
                                      bit,
                                      upper_lower_range,
                                      idx_previous)
                    )

                new_weights_pool = []

                for pop_num in range(pop_size):

                    weight_in_population = next_weights_pool[pop_num]

                    for p_index, p in enumerate(params):
                        # if p.grad is None:  # if grad is turned off, this param may not need to be turned
                        #     continue

                        # copy this weight value as torch tensor
                        p.copy_(torch.tensor(weight_in_population[p_index]['p_value']))

                    # calculate loss value of this individual
                    loss = closure()
                    loss_val = loss.item()
                    inverse = 9999999 if not loss_val else 100 / loss_val

                    new_weights_pool.append({
                        'gen_num': gen_num,
                        'pop_num': pop_num,
                        'param_list': weight_in_population,
                        'loss_val': loss_val,
                        'inverse_loss': inverse,
                    })

                    if save_csv_files:
                        with open('ga-experiments/history_data/mnist/mnist1_genetic_ga_opt_history.csv',
                                  mode='a') as history_file:
                            history_writer = csv.writer(history_file,
                                                        delimiter=',',
                                                        quotechar='"',
                                                        quoting=csv.QUOTE_MINIMAL
                                                        )
                            history_writer.writerow([iteration_index, gen_num, pop_num, loss_val])

                # ascent sort. first element is the best performed individual
                self.population = _update_population(sorted_params,
                                                     new_weights_pool,
                                                     pop_size)

        # copy the best performed weights
        best_individual = sorted(self.population, key=lambda x: x['loss_val'])[0]['param_list']
        for p_index, p in enumerate(params):
            # if p.grad is None:  # if grad is turned off, this param may not need to be turned
            #     continue

            # find best weights
            best_weights = _search_best_performed_weights(best_individual, p_index)
            # copy this random value as torch tensor
            p.copy_(torch.tensor(best_weights))

        # get the best performed loss
        loss = closure()

        return loss
