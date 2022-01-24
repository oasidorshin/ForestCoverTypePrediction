import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, LGBMRegressor, Booster
from catboost import CatBoostClassifier, Pool
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearnex import patch_sklearn
patch_sklearn()

import optuna
from optuna.importance import get_param_importances
from optuna.samplers import TPESampler

import logging, copy, pickle

_EPS = 1e-12

### Hide annoying warning spam
import warnings
warnings.filterwarnings("ignore")

def train_lightgbm(train_df, 
                   num_columns,
                   cat_columns,
                   target, 
                   validation,
                   foldername, 
                   args):

    fixed_params = {'importance_type': 'gain',
                    'random_state': args.seed,
                    'bagging_freq': 1}
    
    def objective(trial):
        # This config is similar to optuna integrated config, just a little bit modified
        # https://github.com/optuna/optuna/blob/master/optuna/integration/_lightgbm_tuner/optimize.py
        trial_params = {'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-7, 100),
                        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-7, 100),
                        'max_depth': trial.suggest_int('max_depth', 3, 12), # Can vary depending on data, -1 = unlimited
                        'num_leaves': trial.suggest_int('num_leaves', 8, 4096, log = True),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1),
                        'n_estimators': trial.suggest_int('n_estimators', 100, 5000, log = True), 
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 1000, log = True), # Can vary depending on data
                        'feature_fraction': min(trial.suggest_float('feature_fraction', 0.4, 1.0 + _EPS), 1.0),
                        'bagging_fraction': min(trial.suggest_float('bagging_fraction', 0.4, 1.0 + _EPS), 1.0)
                        }
                        
        eval_scores = []

        splits = validation(args.n_splits, args.seed)
        X_columns = num_columns + cat_columns

        for i, (train_index, val_index) in enumerate(splits):
            train_split = train_df.iloc[train_index]
            val_split = train_df.iloc[val_index]

            model = LGBMClassifier(**trial_params, **fixed_params)

            model.fit(train_split[X_columns], 
                      train_split[target], 
                      eval_set = (val_split[X_columns], val_split[target]), 
                      eval_metric = "multi_error",
                      categorical_feature = cat_columns,
                      verbose = -1)

            eval_scores.append(model.evals_result_["valid_0"]["multi_error"])
        
        return np.mean(eval_scores)

    # https://optuna.readthedocs.io/en/v1.4.0/reference/logging.html
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    #optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    sampler = TPESampler(seed=args.seed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    # Log best params and score
    logging.info('\n BEST SCORE:')
    logging.info(study.best_value)
    logging.info('\n BEST PARAMS:')
    logging.info(study.best_params)

    # Save optuna stats
    optuna_trials = study.trials_dataframe()
    optuna_importances = pd.DataFrame(get_param_importances(study), index=["importance"]).transpose().sort_values(by = "importance", ascending=False)

    optuna_trials.to_csv(f'models/{foldername}/optuna_trials.csv', index=False)
    optuna_importances.to_csv(f'models/{foldername}/optuna_importances.csv')

    # Retrain, save models and OOF predictions
    splits = validation(args.n_splits, args.seed)
    X_columns = num_columns + cat_columns

    feature_importances = []
    oof_preds = pd.DataFrame(index=train_df.index)
    oof_preds[[f"class_{j}" for j in range(train_df[target].nunique())]] = 0

    for i, (train_index, val_index) in enumerate(splits):
        train_split = train_df.iloc[train_index]
        val_split = train_df.iloc[val_index]

        model = LGBMClassifier(**study.best_params, **fixed_params)

        model.fit(train_split[X_columns], 
                    train_split[target], 
                    eval_set = (val_split[X_columns], val_split[target]), 
                    eval_metric = "multi_error",
                    categorical_feature = cat_columns,
                    verbose = -1)

        # OOF
        oof_preds.iloc[val_index] = model.predict_proba(val_split[X_columns])

        # Model
        model.booster_.save_model(f"models/{foldername}/lgbm_fold_{i}.lgbm")

        # Importances
        feature_importances.append(model.feature_importances_)

    oof_preds.to_csv(f'models/{foldername}/OOF.csv')

    feature_importances = pd.DataFrame(feature_importances, columns = X_columns).transpose()
    feature_importances["mean"] = feature_importances.mean(axis = 1)
    feature_importances = feature_importances.sort_values(by = "mean", ascending=False)

    feature_importances.to_csv(f'models/{foldername}/feature_importances.csv')



def train_catboost(train_df, 
                   num_columns,
                   cat_columns,
                   target, 
                   validation,
                   foldername, 
                   args):

    fixed_params = {'cat_features': cat_columns,
                    'verbose': 0,
                    'eval_metric': 'Accuracy',
                    'random_state': args.seed}
    
    def objective(trial):
        trial_params = {'iterations': trial.suggest_int('iterations', 100, 5000, log = True),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1),
                        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 100),
                        'depth': trial.suggest_int('depth', 3, 10), # Large depth = super slow
                        'rsm': min(trial.suggest_float('rsm', 0.4, 1.0 + _EPS), 1.0) # This is colsample
                        }
                        
        eval_scores = []

        splits = validation(args.n_splits, args.seed)
        X_columns = num_columns + cat_columns

        for i, (train_index, val_index) in enumerate(splits):
            train_split = train_df.iloc[train_index]
            val_split = train_df.iloc[val_index]

            train_pool = Pool(train_split[X_columns], label = train_split[target], cat_features = cat_columns)
            val_pool = Pool(val_split[X_columns], label = val_split[target], cat_features = cat_columns)

            model = CatBoostClassifier(**fixed_params, **trial_params)

            model.fit(train_pool, eval_set = val_pool)

            eval_scores.append(model.evals_result_["validation"]["Accuracy"][-1])
        
        return np.mean(eval_scores)

    # https://optuna.readthedocs.io/en/v1.4.0/reference/logging.html
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    #optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    sampler = TPESampler(seed=args.seed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    # Log best params and score
    logging.info('\n BEST SCORE:')
    logging.info(study.best_value)
    logging.info('\n BEST PARAMS:')
    logging.info(study.best_params)

    # Save optuna stats
    optuna_trials = study.trials_dataframe()
    optuna_importances = pd.DataFrame(get_param_importances(study), index=["importance"]).transpose().sort_values(by = "importance", ascending=False)

    optuna_trials.to_csv(f'models/{foldername}/optuna_trials.csv', index=False)
    optuna_importances.to_csv(f'models/{foldername}/optuna_importances.csv')

    # Retrain, save models and OOF predictions
    splits = validation(args.n_splits, args.seed)
    X_columns = num_columns + cat_columns

    feature_importances = []
    oof_preds = pd.DataFrame(index=train_df.index)
    oof_preds[[f"class_{j}" for j in range(train_df[target].nunique())]] = 0

    for i, (train_index, val_index) in enumerate(splits):
        train_split = train_df.iloc[train_index]
        val_split = train_df.iloc[val_index]

        train_pool = Pool(train_split[X_columns], label = train_split[target], cat_features = cat_columns)
        val_pool = Pool(val_split[X_columns], label = val_split[target], cat_features = cat_columns)

        model = CatBoostClassifier(**fixed_params, **study.best_params)

        model.fit(train_pool, eval_set = val_pool)

        # OOF
        oof_preds.iloc[val_index] = model.predict_proba(val_split[X_columns])

        # Model
        model.save_model(f"models/{foldername}/cb_fold_{i}.cbm")

        # Importances
        feature_importances.append(model.get_feature_importance())

    oof_preds.to_csv(f'models/{foldername}/OOF.csv')

    feature_importances = pd.DataFrame(feature_importances, columns = X_columns).transpose()
    feature_importances["mean"] = feature_importances.mean(axis = 1)
    feature_importances = feature_importances.sort_values(by = "mean", ascending=False)

    feature_importances.to_csv(f'models/{foldername}/feature_importances.csv')


def train_svm(train_df, 
              num_columns,
              target, 
              validation,
              foldername, 
              args):

    fixed_params = {'max_iter': 5000, # Can get stuck without it
                    'random_state': args.seed}
    
    def objective(trial):
        trial_params = {'C': trial.suggest_loguniform('C', 1e-02, 1e02),
                        'gamma': trial.suggest_loguniform('gamma', 1e-02, 1e02),
                        }
                        
        eval_scores = []

        splits = validation(args.n_splits, args.seed)

        for i, (train_index, val_index) in enumerate(splits):
            train_split = train_df.iloc[train_index]
            val_split = train_df.iloc[val_index]

            model = SVC(**trial_params, **fixed_params)

            model.fit(train_split[num_columns], train_split[target])

            # TODO
            eval_scores.append(accuracy_score(val_split[target], model.predict(val_split[num_columns])))
        
        return np.mean(eval_scores)

    # https://optuna.readthedocs.io/en/v1.4.0/reference/logging.html
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    #optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    sampler = TPESampler(seed=args.seed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    # Log best params and score
    logging.info('\n BEST SCORE:')
    logging.info(study.best_value)
    logging.info('\n BEST PARAMS:')
    logging.info(study.best_params)

    # Save optuna stats
    optuna_trials = study.trials_dataframe()
    optuna_importances = pd.DataFrame(get_param_importances(study), index=["importance"]).transpose().sort_values(by = "importance", ascending=False)

    optuna_trials.to_csv(f'models/{foldername}/optuna_trials.csv', index=False)
    optuna_importances.to_csv(f'models/{foldername}/optuna_importances.csv')

    # Retrain, save models and OOF predictions
    splits = validation(args.n_splits, args.seed)

    oof_preds = pd.DataFrame(index=train_df.index)
    oof_preds["predicted"] = 0

    for i, (train_index, val_index) in enumerate(splits):
        train_split = train_df.iloc[train_index]
        val_split = train_df.iloc[val_index]

        model = SVC(**study.best_params, **fixed_params)

        model.fit(train_split[num_columns], train_split[target])

        # OOF
        oof_preds.iloc[val_index] = np.expand_dims(model.predict(val_split[num_columns]), axis = 1)

        # Model
        pickle.dump(model, open(f'models/{foldername}/svm_fold_{i}.svm', 'wb'))

    oof_preds.to_csv(f'models/{foldername}/OOF.csv')

        


