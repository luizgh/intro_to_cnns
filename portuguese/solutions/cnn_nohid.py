net = {}

net['data'] = InputLayer((None, 28*28))
net['out'] = DenseLayer(net['data'], 10, nonlinearity=softmax)