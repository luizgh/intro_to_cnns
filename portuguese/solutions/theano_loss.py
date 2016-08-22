y = T.vector()
loss = -(y * T.log(y_hat) + (1-y) * T.log(1-y_hat))
loss = loss.mean()