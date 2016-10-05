import sys
import matplotlib.pyplot as plt
import numpy as np

def plot_train_curves(train_curves):
    plt.figure()
    cost_history, acc_history, val_cost_history, val_acc_history = train_curves
    plt.plot(cost_history, 'b--', label='Treinamento')
    plt.plot(val_cost_history, 'r-', label='Validacao')
    plt.xlabel('Numero de iteracoes', fontsize=15)
    plt.ylabel('Custo', fontsize=15)
    plt.legend()
    print "Melhor performance validacao: %.2f%%" % (np.max(val_acc_history) * 100)
    
def iterate_minibatches(x, y, batch_size):
    for batch_start in xrange(0, len(x), batch_size):
        yield x[batch_start:batch_start+batch_size], y[batch_start:batch_start+batch_size]


def train_minibatch(train_fn, val_fn, train_set, valid_set, epochs, batch_size):
    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    
    cost_history = []
    acc_history = []
    val_cost_history = []
    val_acc_history = []

    nbatches = len(x_train / batch_size)
    
    print('progress\tepoch\ttrain_err\tval_err')
    for i in range(epochs):
        epoch_cost = 0
        epoch_acc = 0
        train_batches = 0
        n_batches = len(x_train) / batch_size
        for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size):
            cost, acc = train_fn(x_batch, y_batch)
            sys.stdout.write('#')
            #print 'Batch %d of %d. Cost: %.4f' % (train_batches, n_batches, cost)
            epoch_cost += cost
            epoch_acc += acc
            train_batches += 1

        val_epoch_cost = 0
        val_epoch_acc = 0
        val_batches = 0
        for x_batch, y_batch in iterate_minibatches(x_valid, y_valid, batch_size):
            val_cost, val_acc = val_fn(x_batch, y_batch)
            val_epoch_cost += val_cost
            val_epoch_acc += val_acc
            val_batches += 1
            
        epoch_cost = epoch_cost / train_batches
        cost_history.append(epoch_cost)
        acc_history.append(epoch_acc / train_batches)

        val_epoch_cost = val_epoch_cost / val_batches
        val_cost_history.append(val_epoch_cost)
        val_acc_history.append(val_epoch_acc / val_batches)
        print('\t%d\t%.4f   \t%.4f' % (i+1, epoch_cost, val_epoch_cost))
    return cost_history, acc_history, val_cost_history, val_acc_history
