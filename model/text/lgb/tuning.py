import lightgbm as lgb

from model.text.lgb.config import config


def choose_eta(ddev, dval, num_class):
    current_score = 10e99
    chosen_eta = 0.00005
    for eta in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        params = {'objective': 'multiclass',
                  'max_depth': 6,
                  'num_leaves': 63,
                  "feature_fraction": 0.7,
                  "bagging_fraction": 0.7,
                  'silent': 1,
                  'nthread': config.n_threads,
                  'num_class': num_class,
                  'eta': eta}

        bst = lgb.train(params, ddev, num_boost_round=config.n_round_eta, valid_sets=[ddev, dval], valid_names=['ddev', 'dval'],
                        early_stopping_rounds=50)
        new_score = bst.best_score['dval']['multi_logloss']
        if new_score < current_score:
            current_score = new_score
            chosen_eta = eta
        else:
            break
    return chosen_eta