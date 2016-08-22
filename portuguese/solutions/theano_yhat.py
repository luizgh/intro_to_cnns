x = T.matrix()

w = theano.shared(w_init)
b = theano.shared(b_init)

z = x.dot(w) + b
y_hat = 1./ (1 + T.exp(-z))