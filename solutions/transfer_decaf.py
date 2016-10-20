input_var = model['input'].input_var
fc7_output = lasagne.layers.get_output(model['fc7'], deterministic=True)

get_fc7 = theano.function([input_var], fc7_output)
x_train_fc7 = get_output_batch(get_fc7, x_train_processed)
x_valid_fc7 = get_output_batch(get_fc7, x_valid_processed)

linear_model = LogisticRegression()
linear_model.fit(x_train_fc7, y_train)
linear_model.score(x_valid_fc7, y_valid)

