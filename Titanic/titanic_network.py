import random
import numpy as np
import sys

class QuadraticCost(object):

    @staticmethod
    def fn(a,y):
        #return cost associated with an output A and desired output Y
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z,a,y):
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z,a,y):
        return(a-y)

class Network(object):
    def __init__(self, sizes, cost=QuadraticCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        self.biases = [np.random.randn(x,1) for x in self.sizes[1:]]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data = None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)
        training_data = list(training_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
            print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self,mini_batch,eta,lmbda,n):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x,y in mini_batch:
            delta_nabla_w , delta_nabla_b = self.backprop(x,y)
            nabla_w = [nw + w for w,nw in zip(nabla_w,delta_nabla_w)]
            nabla_b = [nb + b for b,nb in zip(nabla_b,delta_nabla_b)]
        self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                        for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b;
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (self.cost).delta(zs[-1],activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())

        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp;
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_w,nabla_b)

    def accuracy(self,data,convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        return sum((x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

def vectorized_result(j):
    a = np.zeros(j.shape,dtype='int64')
    a[0] = int(j[0])
    a[1] = int(j[1])
    e = np.zeros((2, 1))
    e[a] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
