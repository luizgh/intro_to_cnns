def build_conv_larger():
    net = {}

    net['data'] = InputLayer((None, 1, 28, 28))
    net['conv1'] = Conv2DLayer(net['data'], filter_size=5, num_filters=8)
    net['pool1'] = MaxPool2DLayer(net['conv1'], 2)
    
    net['conv2'] = Conv2DLayer(net['pool1'], filter_size=5, num_filters=16)
    net['pool2'] = MaxPool2DLayer(net['conv1'], 3)
    
    net['hid'] = DenseLayer(net['pool2'], 100)
    
    net['out'] = DenseLayer(net['hid'], 10, nonlinearity=lasagne.nonlinearities.softmax)
    return net
