x = T.scalar()
y = T.log(2 * x)

dy_dx = T.grad(y,x)

dy_dx.eval({x: 10})
