from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train_flat, y_train)
classifier.score(x_valid_flat, y_valid)
