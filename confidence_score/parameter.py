# This is parameters for the simulation model
simu_default_para = {
    'rand_guess_para': {'rate_ppl': 0.1, 'acc': 0.25, 'gamma': [10, 0.5]},
    'prek_acc': 0.9,
    'mu_theta': [0.5, 0],
    'cov_theta': [[1, 0.12], [0.12, 0.02]]
}

# The following two are factors in simulation study.
para_null = {
        'ppl_num': [1000],
        'item_num': [500],
        'test_len': [50],
        'ci_rate': [0],
        'ewp_rate': [0],
        'iterat_times': [60],
        'ab_cri': [500]
}

para = {'ppl_num': [1000],
        'item_num': [500],
        'test_len': [50],
        'ci_rate': [0.1, 0.2, 0.4],
        'ewp_rate': [0.1, 0.2, 0.4],
        'iterat_times': [60],
        'ab_cri': [500]
}

# This is some settings used in the detection algorithm and confidence score
exp_default_para = {
    'index_cri': 1.15,
    'num_cri': 0.1,
    'iterat_prop':0.6,
    'false_type': ['false_posi_item', 'false_neg_item', 'false_posi_ppl', 'false_neg_ppl','precision_ppl','precision_item']
}
