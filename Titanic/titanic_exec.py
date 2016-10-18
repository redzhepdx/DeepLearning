import readFile
import titanic_network
import numpy as np

training_data , test_data = readFile.load_data()
training_data = list(training_data)
test_data = list(test_data)

net = titanic_network.Network([5,400,2], cost=titanic_network.QuadraticCost)
net.default_weight_initializer()
max_epoch = 50
max_mini_batch_size = 30
max_eta = 20
max_lmbda = 20
accuracies_e = []
accuracies_t = []
accuraciesInfos = []
loop_Count = max_epoch*max_eta*max_mini_batch_size*max_lmbda
e_c , e_a , t_c , t_a = net.SGD(training_data, 1500, 5, 0.01, evaluation_data=test_data, lmbda = 0.003,monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,monitor_training_cost=True, monitor_training_accuracy=True)
