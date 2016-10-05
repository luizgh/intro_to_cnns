model = build_model_for_finetuning(params)

train_fn, valid_fn = compile_train_function_somelayers(model, lr=0.005, w_decay=1e-5, 
                                                       layers_to_train=[model['fc7'], model['out']])

train_curves = train_minibatch(train_fn, valid_fn,     # Treinamento usando Batch Gradient Descent
                     train_set=(x_train, y_train), 
                     valid_set=(x_test, y_test),
                     epochs=20,
                     batch_size=16)
plot_train_curves(train_curves)
