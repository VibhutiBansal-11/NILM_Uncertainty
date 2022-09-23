def predict(n_models, model, params_list, X):
    means_list = []
    sigmas_list = []
    for i in range(n_models):
        mean, sigma = model.apply(params_list[i], X, deterministic=True)
        means_list.append(mean)
        sigmas_list.append(sigma)
    return means_list, sigmas_list
