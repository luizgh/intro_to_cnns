for lr in [0.01, 0.1, 1, 10]:
    model = build_no_hid_layer()
    train_fn, valid_fn = compile_train_function(model, lr=lr)
    train_curves = train(train_fn, valid_fn, 
                         train_set=(x_train_flat, y_train), 
                         valid_set=(x_valid_flat, y_valid),
                         epochs=10)
    plot_train_curves(train_curves)

