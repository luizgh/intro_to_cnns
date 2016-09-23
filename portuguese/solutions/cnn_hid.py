def build_hid_layer(nhid):
    net = {}

    net['data'] = InputLayer((None, 28*28))
    net['hidden'] = DenseLayer(net['data'], nhid)
    
    net['out'] = DenseLayer(net['hidden'], 10, nonlinearity=lasagne.nonlinearities.softmax)
    return net
