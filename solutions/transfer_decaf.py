#Roda uma função em batches e concatena o resultado:
def get_output_batch(function, x, batch_size=32):
    output = []
    for batch_start in xrange(0, len(x), batch_size):
        output.append(function(x[batch_start:batch_start+batch_size]))
    return np.vstack(output)


input_var = model['input'].input_var
fc7_output = lasagne.layers.get_output(model['fc7'], deterministic=True)

get_fc7 = theano.function([input_var], fc7_output)
x_train_fc7 = get_output_batch(get_fc7, x_train)
x_valid_fc7 = get_output_batch(get_fc7, x_valid)

linear_model = sklearn.linear_model.LogisticRegression()
linear_model.fit(x_train_fc7, y_train)
linear_model.score(x_valid_fc7, y_valid)

