import math
import matplotlib.pyplot as plt

epsilon1 = []
epsilon2 = []

def eg(steps_counter):
    ep = 0.00 + (1.0 - 0.00) * math.exp(-1 * steps_counter/500)

    return ep

episodes = 10
steps_counter = 0
for episode in range(episodes):

    for i in range(100):
        ep = (episodes - episode)/episodes
        epsilon1.append(ep)

        steps_counter += 1
        epi = eg(steps_counter)
        epsilon2.append(epi)

plt.plot(range(len(epsilon1)), epsilon1, marker='o')
plt.plot(range(len(epsilon1)), epsilon2, marker='s')

plt.title('Epsilon Range of linear vs exponential')
plt.xlabel('Total Steps')
plt.ylabel('Epsilons')
plt.show()