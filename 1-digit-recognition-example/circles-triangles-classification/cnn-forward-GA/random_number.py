import numpy as np
import random

filter_values = []
for i in range(20):  # population
    # firstly, generate 20 different filters
    random_filter = np.random.randn(8, 3, 3) / 9
    # print(random_filter)
    # print(random_filter[0, 0, 0])
    filter_values.append(random_filter)

for k, filter_value in enumerate(filter_values):
    mutation_probability = random.uniform(0, 1)
    # if larger than 0.5 then mutate
    if mutation_probability > 0.5:
        # random number of elements to change
        # because it is 3x3 filter,
        #  8 x (3x3) = 72
        # so, we don't want to change to many element
        number_of_elements = random.randint(1, 20)

        # the elements that have been already changed
        has_changed_list = []
        for i in range(number_of_elements):
            row = random.randint(0, 2)
            col = random.randint(0, 2)
            # filter_size = 8 x (3x3),
            # so randomly change one filter
            the_number = random.randint(0, 7)
            key_value_pair = the_number + row + col
            if key_value_pair not in has_changed_list:
                filter_value[the_number, row, col] = 0
                has_changed_list.append(key_value_pair)
