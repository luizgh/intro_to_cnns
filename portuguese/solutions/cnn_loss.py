input_var = net['data'].input_var
output_var = T.ivector()

predicted = lasagne.layers.get_output(net['out'], inputs=input_var)
loss = categorical_crossentropy(predicted, output_var)
loss = loss.mean()
