def build_conv_small():
    net = {}

    net['data'] = InputLayer((None, 1, 28, 28))
    net['conv1'] = Conv2DLayer(net['data'], filter_size=5, num_filters=6)
    net['pool1'] = MaxPool2DLayer(net['conv1'], 3)
        
    net['out'] = DenseLayer(net['pool1'], 10, nonlinearity=lasagne.nonlinearities.softmax)
    return net
