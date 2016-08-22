a = theano.shared(3)
b = theano.shared(1)
x = T.scalar()
y = a*x + b

f = theano.function([x],y)

print f(5)