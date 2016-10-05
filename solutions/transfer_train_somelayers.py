def compile_train_function_somelayers(net, lr, w_decay, layers_to_train):
    input_var = net['input'].input_var
    output_var = T.ivector()

    probs = lasagne.layers.get_output(net['out'], inputs=input_var)
    loss = lasagne.objectives.categorical_crossentropy(probs, output_var)
    loss = loss.mean()
    loss += w_decay * regularize_layer_params(net['out'], l2)
    
    y_pred = T.argmax(probs, axis=1)
    acc = T.eq(y_pred, output_var)
    acc = acc.mean()
    
    test_probs = lasagne.layers.get_output(net['out'], inputs=input_var, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_probs, output_var)
    test_loss = test_loss.mean()
    
    test_pred = T.argmax(test_probs, axis=1)
    test_acc = T.eq(test_pred, output_var)
    test_acc = test_acc.mean()
    
    params = []
    for l in layers_to_train:
        params += l.get_params(trainable=True)
        
    updates = lasagne.updates.sgd(loss, params, lr)

    train_fn = theano.function([input_var, output_var], [loss, acc], updates=updates)
    val_fn = theano.function([input_var, output_var], [test_loss, test_acc])
    return train_fn, val_fn
