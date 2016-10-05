for nhid in [10, 100, 1000]:
    print 'Treinando com %d neuronios na camada escondida' % nhid
    model = build_hid_layer(nhid)
    train_fn, valid_fn = compile_train_function(model, lr=0.1)
    train_curves = train_minibatch(train_fn, valid_fn, 
                         train_set=(x_train_flat, y_train), 
                         valid_set=(x_valid_flat, y_valid),
                         epochs=30,
                         batch_size=128)
    plot_train_curves(train_curves)
