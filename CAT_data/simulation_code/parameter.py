# This is parameters for the simulation model
simu_default_para = {
    'rand_guess_para': {'rate_ppl': 0.1, 'acc': 0.25, 'gamma': [10,0.5]},
    'prek_acc':0.9,
    'mu_theta': [0.5, 0],
    'cov_theta': [[1, 0.12], [0.12, 0.02]]
}

# The following two are factors in simulation study.
para_null = {
        'ppl_num': [1000],
        'item_num': [500],
        'test_len': [50],
        'ci_rate': [0],
        'ewp_rate': [0]
}

para = {'ppl_num': [1000],
        'item_num': [500],
        'test_len': [50],
        'ci_rate': [0.1, 0.2, 0.4],
        'ewp_rate': [0.1, 0.2, 0.4]
}
