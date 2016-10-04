def build_model_for_finetuning(params):
    model = build_model()
    lasagne.layers.set_all_param_values(model['prob'], params['values'])
    model['data'] = model['input'] # nosso código de treinamento espera que a entrada chame-se "data"
    
    del model['fc8']
    del model['prob']
    model['out'] = DenseLayer(model['drop7'], 5, nonlinearity=softmax )
    
    return model