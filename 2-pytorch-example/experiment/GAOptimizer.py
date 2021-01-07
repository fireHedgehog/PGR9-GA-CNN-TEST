import torch
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
import math


class GAOptimizer(Optimizer):
    r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
    """

    def __init__(self,
                 params,
                 generation_size=20,
                 pop_size=20,
                 mutation_rate=0.1,
                 crossover_rate=0.6,
                 elite_rate=0.0,
                 new_chromosome_rate=0.0):

        self.population = []

        defaults = dict(
            generation_size=generation_size,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_rate=elite_rate,
            new_chromosome_rate=new_chromosome_rate,
        )

        super(GAOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GAOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            generation_size = group['generation_size']
            pop_size = group['pop_size']
            mutation_rate = group['mutation_rate']
            crossover_rate = group['crossover_rate']
            elite_rate = group['elite_rate']
            new_chromosome_rate = group['new_chromosome_rate']

            params = group['params']

            for a in range(5):
                for p_index, p in enumerate(params):
                    if p.grad is None:
                        continue
                    if not any(obj['id'] == p_index for obj in self.population):
                        self.population.append({
                            "id": p_index,
                            "values": []
                        })
                    else:
                        for i, obj in enumerate(self.population):
                            if obj['id'] == p_index:
                                self.population[i] = {
                                    "id": p_index,
                                    "values": [np.random.randint(1, 9)]
                                }

                        p_data = p.data.add(
                            torch.mul(
                                torch.normal(mean=0, std=1, size=p.size()).cuda(),
                                torch.mean(p)
                            ).cuda()
                        )
                        p.copy_(p_data)

                loss = closure()

            print(loss)

        return loss
