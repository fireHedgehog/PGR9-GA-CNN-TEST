from mnist import MNIST
from numpy import random

mndata = MNIST('samples')
images, labels = mndata.load_training()

index = random.randint(len(images))  # choose an index ;-)

img = images[index]

print(mndata.display(images[index]))
