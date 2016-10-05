model = build_model_for_finetuning(params)
train_fn, valid_fn = compile_train_function(model, lr=0.001, w_decay= 1e-5)

