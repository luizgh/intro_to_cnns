cost_history = []
acc_history = []
val_cost_history = []
val_acc_history = []

for i in range(50):
    cost, acc = train_fn(x_train_flat, y_train)
    cost_history.append(cost)
    acc_history.append(acc)
    
    val_cost, val_acc = valid_fn(x_valid_flat, y_valid)
    val_cost_history.append(val_cost)
    val_acc_history.append(val_acc)
