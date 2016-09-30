from sklearn.linear_model import LogisticRegression
linear_model = LogisticRegression()
linear_model.fit(x_train_flat, y_train)
linear_model.score(x_valid_flat, y_valid)
