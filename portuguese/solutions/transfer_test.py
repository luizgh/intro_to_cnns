x_test = process_dataset(x_test)
x_test_fc7 = get_output_batch(get_fc7, x_test)

linear_model.score(x_test_fc7, y_test)
