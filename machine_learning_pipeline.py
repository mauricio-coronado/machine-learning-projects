import pandas as pd

# from dataprep.clean import *
# import sweetviz as sv
# import missingno as msno
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt
import kaleido

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC
# from sklearn.feature_selection import SelectKBest
# from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# import seaborn as sns
import numpy as np

from typing import Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, make_scorer, r2_score

from optuna import Trial, create_study, visualization
# from sklego.preprocessing import ColumnSelector

from category_encoders import OrdinalEncoder, OneHotEncoder, WOEEncoder, TargetEncoder, CatBoostEncoder, SumEncoder, BinaryEncoder, HelmertEncoder
# from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor, AdaBoostRegressor

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from optuna.exceptions import TrialPruned

import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import plotly.figure_factory as ff
import plotly.express as px
from operator import add, sub, mul, truediv
from sklearn.preprocessing import FunctionTransformer
from IPython.display import Image


########## IMPUTERS #########


def instantiate_numerical_simple_imputer(trial : Trial) -> SimpleImputer:

    """
    Instantiates a simple imputer for numerical data
    """

    strategy = trial.suggest_categorical(
        'numerical_strategy', ['mean', 'median']
    )
    return SimpleImputer(strategy=strategy)


def instantiate_categorical_simple_imputer(trial : Trial, fill_value : str='missing_value') -> SimpleImputer:

    """
    Instantiates a simple imputer for categorical data
    """

    strategy = trial.suggest_categorical(
        'categorical_strategy', ['most_frequent', 'constant']
    )
    return SimpleImputer(strategy=strategy, fill_value=fill_value)



########## ENCODERS #########

def instantiate_encoder(trial : Trial):

    """
    Instantiates an encoder for categorical data. Each trial 
    will be able to select a different encoder from the list 
    defined in the encoder_strategy
    """

    encoder_strategy = trial.suggest_categorical(
        'categorical_encoder', [
                                # 'ordinal', 
                                'onehot', 
                                # 'binary', 
                                # 'helmert', 
                                # 'sum', 
                                # 'target',
                                # 'woe', 
                                # 'catboost'
                                ]
    )

    return encoder_selection(encoder_strategy)


def encoder_selection(encoder_strategy):

    """
    Auxiliar function for encoder_selection
    """

    if encoder_strategy == 'ordinal':
        encoder = OrdinalEncoder()
    elif encoder_strategy == 'onehot':
        encoder = OneHotEncoder()
    elif encoder_strategy == 'binary':
        encoder = BinaryEncoder()
    elif encoder_strategy == 'helmert':
        encoder = HelmertEncoder()
    elif encoder_strategy == 'sum':
        encoder = SumEncoder()
    elif encoder_strategy == 'target':
        encoder = TargetEncoder()
    elif encoder_strategy == 'woe':
        encoder = WOEEncoder()
    elif encoder_strategy == 'catboost':
        encoder = CatBoostEncoder()

    return encoder


def instantiate_woe_encoder(trial : Trial) -> WOEEncoder:

    params = {
        'sigma': trial.suggest_float('sigma', 0.001, 5),
        'regularization': trial.suggest_float('regularization', 0, 5),
        'randomized': trial.suggest_categorical('randomized', [True, False])
    }
    return WOEEncoder(**params)



########## SCALERS #########

def instantiate_robust_scaler(trial : Trial) -> RobustScaler:

    """
    Instantiates robust scaler for numerical data
    """

    params = {
        'with_centering': trial.suggest_categorical(
            'with_centering', [True, False]
        ),
        'with_scaling': trial.suggest_categorical(
            'with_scaling', [True, False]
        )
    }
    return RobustScaler(**params)



########## FEATURE ENGINEERING #########

def instantiate_feature_interaction(trial : Trial) -> PolynomialFeatures:

    """
    Feature interactions for numerical data
    """

    params = {
        'degree': trial.suggest_int('degree', 1, 3),
        'interaction_only': trial.suggest_categorical('interaction_only', [True, False]),
        'include_bias': trial.suggest_categorical('include_bias', [True, False])
    }
    return PolynomialFeatures(**params)

from numpy import inf


def numerical_features_interactions(df, 
                                        numerical_features, 
                                        interaction_types=[('+', add), ('*', mul), ('/', truediv)]):


    """
    Basic interactions function (not in use in the pipeline anymore)
    """

    enhanced_df = df.copy()
    for operation_sign, operation in interaction_types:
        for i, feature_1 in enumerate(numerical_features):
            for feature_2 in numerical_features[i+1:]:
                new_feature_series = operation(enhanced_df[feature_1], enhanced_df[feature_2])
                new_feature_series.name = feature_1 + operation_sign + feature_2
                enhanced_df = pd.concat([enhanced_df, new_feature_series], axis=1)
    
    
    enhanced_df = enhanced_df.replace([np.inf, -np.inf], -1)  
    return enhanced_df


def numerical_features_interactions_np(df, 
                                        numerical_features, 
                                        interaction_types=[('+', add), ('*', mul), ('/', truediv)]):


    """
    
    """

    enhanced_df = df.copy()
    for operation_sign, operation in interaction_types:
        for i in range(len((numerical_features))):
            for j in range(len(numerical_features[i+1:])):
                new_feature_series = operation(enhanced_df[:,i], enhanced_df[:,j+1])
                new_feature_series = np.expand_dims(new_feature_series, axis=1)
                # print(enhanced_df.shape)
                # print(new_feature_series.shape)
                enhanced_df = np.concatenate([enhanced_df, new_feature_series], axis=1)
    enhanced_df[enhanced_df == -inf] = -1
    enhanced_df[enhanced_df == inf] = -1
    return enhanced_df



def feature_interaction_names(transformer,
                              numerical_features, 
                              interaction_types=[('+', add), ('*', mul), ('/', truediv)]):


    """
    
    """

    enhanced_numerical_features = numerical_features.copy()

    for operation_sign, operation in interaction_types:
        for i, feature_1 in enumerate(numerical_features):
            for feature_2 in numerical_features[i+1:]:
                enhanced_numerical_features.append(feature_1 + operation_sign + feature_2)

    return enhanced_numerical_features


########## COLUMN SELECTION #########


# def instantiate_column_selector(trial : Trial, columns : list[str]) -> ColumnSelector:
#     choose = lambda column: trial.suggest_categorical(column, [True, False])
#     choices = [*filter(choose, columns)]
#     selector = ColumnSelector(choices)
#     return selector

def choose_columns(trial : Trial, columns : list[str]) -> list[str]:
  
  """
  Randomly selection of featuress
  """

  choose = lambda column: trial.suggest_categorical(column, [True, False])
  choices = [*filter(choose, columns)]
  return choices



########## MODELS #########

def instantiate_linearregression(trial : Trial) -> LinearRegression:
    
    params = {
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'n_jobs': -1,
    }
    return LinearRegression(**params)


def instantiate_ridge(trial : Trial) -> Ridge:
    params = {
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'alpha': trial.suggest_float('alpha', 0, 1000000),
    }
    return Ridge(**params)


def instantiate_extratrees_classifier(trial : Trial) -> ExtraTreesClassifier:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'max_features': trial.suggest_float('max_features', 0, 1),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'n_jobs': -1,
        'random_state': 0
    }
    return ExtraTreesClassifier(**params)


def instantiate_extratrees_regressor(trial : Trial) -> ExtraTreesRegressor:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'max_features': trial.suggest_float('max_features', 0, 1),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'n_jobs': -1,
        'random_state': 0
    }
    return ExtraTreesRegressor(**params)


def instantiate_histgradientboost_classifier(trial : Trial) -> HistGradientBoostingClassifier:
    params = {
        'max_iter': trial.suggest_int('max_iter', 50, 1000),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        # 'max_features': trial.suggest_float('max_features', 0, 1),
        # 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        # 'n_jobs': -1,
        'random_state': 0
    }
    return HistGradientBoostingClassifier(**params)


def instantiate_histgradientboost_regressor(trial : Trial) -> HistGradientBoostingRegressor:
    params = {
        'max_iter': trial.suggest_int('max_iter', 50, 1000),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        # 'max_features': trial.suggest_float('max_features', 0, 1),
        # 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        # 'n_jobs': -1,
        'random_state': 0
    }
    return HistGradientBoostingRegressor(**params)


def instantiate_histgradientboost_multiclass(trial : Trial) -> HistGradientBoostingClassifier:
    params = {
        'max_iter': trial.suggest_int('max_iter', 50, 1000),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        # 'max_features': trial.suggest_float('max_features', 0, 1),
        # 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        # 'n_jobs': -1,
        'random_state': 0
    }
    return HistGradientBoostingClassifier(**params)


def instantiate_adaboost_classifier(trial : Trial) -> AdaBoostClassifier:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        # Tune hyperparameters for the decision tree base estimator
        'estimator': DecisionTreeClassifier(
                max_depth=trial.suggest_int('max_depth', 1, 32),
                min_samples_split=trial.suggest_float('min_samples_split', 0.1, 1.0),
                min_samples_leaf=trial.suggest_float('min_samples_leaf', 0.1, 0.5),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        ),
        # 'max_depth': trial.suggest_int('max_depth', 1, 20),
        # 'max_features': trial.suggest_float('max_features', 0, 1),
        # 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        # 'n_jobs': -1,
        'random_state': 0
    }
    return AdaBoostClassifier(algorithm="SAMME",
                              **params)


def instantiate_adaboost_regressor(trial : Trial) -> AdaBoostRegressor:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        # Tune hyperparameters for the decision tree base estimator
        'estimator': DecisionTreeRegressor(
                max_depth=trial.suggest_int('max_depth', 1, 32),
                min_samples_split=trial.suggest_float('min_samples_split', 0.1, 1.0),
                min_samples_leaf=trial.suggest_float('min_samples_leaf', 0.1, 0.5),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        ),
        # 'max_depth': trial.suggest_int('max_depth', 1, 20),
        # 'max_features': trial.suggest_float('max_features', 0, 1),
        # 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        # 'n_jobs': -1,
        'random_state': 0
    }
    return AdaBoostRegressor(**params)


def instantiate_lgb_classifier(trial : Trial) -> LGBMClassifier:
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 100.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 100.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbosity': -1,
        'random_state': 0
    }
    return LGBMClassifier(**params)


def instantiate_lgb_regressor(trial : Trial) -> LGBMRegressor:
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 1, 8),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbosity': -1,
        'random_state': 0
    }
    return LGBMRegressor(**params)


def instantiate_lgb_multiclass(trial : Trial) -> LGBMClassifier:
    params = {
        'objective': 'multiclass',
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbosity': -1,
        'random_state': 0
    }
    return LGBMClassifier(**params)


def instantiate_xgb_regressor(trial : Trial) -> XGBRegressor:

    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 150, 1000),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        # 'tree_method': 'hist',  # Use GPU for training
        # 'device' : 'cuda',
        'random_state': 0,
        'eval_metric': 'auc',  # Evaluation metric
        # 'verbosity': 2,  # Set verbosity to 0 for less output
    }

    return XGBRegressor(**params)



def instantiate_xgb_classifier(trial : Trial) -> XGBClassifier:
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 150, 1000),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'tree_method': 'hist',  # Use GPU for training
        'device' : 'cuda',
        'random_state': 0,
        'eval_metric': 'auc',  # Evaluation metric
        # 'verbosity': 2,  # Set verbosity to 0 for less output
    }
    return XGBClassifier(**params)



def instantiate_catboost_regressor(trial : Trial) -> CatBoostRegressor:
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1000),
        'depth': trial.suggest_int('depth', 10, 50),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 20),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
    }
    return CatBoostRegressor(**params)



def instantiate_catboost_classifier(trial : Trial) -> CatBoostClassifier:
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1000),
        'depth': trial.suggest_int('depth', 5, 16),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 20),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
    }
    return CatBoostClassifier(**params)


def instantiate_learner(trial : Trial, objective, algorithms):
    
    algorithm = trial.suggest_categorical(
        'algorithm', algorithms
        )
    
    if objective == 'regression':
    
        if algorithm =='histgb':
            model = instantiate_histgradientboost_regressor(trial)
        elif algorithm =='lgb':
            model = instantiate_lgb_regressor(trial)
        elif algorithm =='extratrees':
            model = instantiate_extratrees_regressor(trial)
        elif algorithm =='adaboost':
            model = instantiate_adaboost_regressor(trial)
        elif algorithm == 'linear':
            model = instantiate_linearregression(trial)
        elif algorithm == 'ridge':
            model = instantiate_ridge(trial)
        elif algorithm == 'xgb':
            model = instantiate_xgb_regressor(trial)
        elif algorithm == 'catboost':
            model = instantiate_catboost_regressor(trial)
        # elif algorithm=='knn':
        #     model = instantiate_knn(trial)
    
    elif objective == 'classification':

        if algorithm=='histgb':
            model = instantiate_histgradientboost_classifier(trial)
        elif algorithm=='lgb':
            model = instantiate_lgb_classifier(trial)
        elif algorithm=='extratrees':
            model = instantiate_extratrees_classifier(trial)
        elif algorithm=='adaboost':
            model = instantiate_adaboost_classifier(trial)
        elif algorithm == 'xgb':
            model = instantiate_xgb_classifier(trial)
        elif algorithm == 'catboost':
            model = instantiate_catboost_classifier(trial)

        # elif algorithm=='knn':
        #     model = instantiate_knn(trial)

    elif objective == 'multiclass':

        if algorithm=='histgb':
            model = instantiate_histgradientboost_multiclass(trial)
        elif algorithm=='lgb':
            model = instantiate_lgb_multiclass(trial)
        # elif algorithm=='extratrees':
        #     model = instantiate_extratrees_classifier(trial)
        # elif algorithm=='adaboost':
        #     model = instantiate_adaboost_classifier(trial)
        # # elif algorithm=='knn':
        # #     model = instantiate_knn(trial)
    
    return model

    
def learner_selection(algorithm, objective, **kwargs):

    if objective == 'regression':
    
        if algorithm =='histgb':
            model = HistGradientBoostingRegressor(**kwargs)
        elif algorithm =='lgb':
            model = LGBMRegressor(**kwargs)
        elif algorithm =='extratrees':
            model = ExtraTreesRegressor(**kwargs)
        elif algorithm =='adaboost':
            model = AdaBoostRegressor(**kwargs)
        elif algorithm == 'linear':
            model = LinearRegression(**kwargs)
        elif algorithm == 'ridge':
            model = Ridge(**kwargs)
        elif algorithm == 'xgb':
            model = XGBRegressor(**kwargs)
        elif algorithm == 'catboost':
            model = CatBoostRegressor(**kwargs)
        # elif algorithm=='knn':
        #     model = instantiate_knn(trial)
    
    elif objective == 'classification':

        if algorithm=='histgb':
            model = HistGradientBoostingClassifier(**kwargs)
        elif algorithm=='lgb':
            model = LGBMClassifier(**kwargs)
        elif algorithm=='extratrees':
            model = ExtraTreesClassifier(**kwargs)
        elif algorithm=='adaboost':
            model = AdaBoostClassifier(**kwargs)
        elif algorithm == 'xgb':
            model = XGBClassifier(**kwargs)
        elif algorithm == 'catboost':
            model = CatBoostClassifier(**kwargs)

        # elif algorithm=='knn':
        #     model = instantiate_knn(trial)

    elif objective == 'multiclass':

        if algorithm=='histgb':
            model = HistGradientBoostingClassifier(**kwargs)
        elif algorithm=='lgb':
            model = LGBMClassifier(objective='multiclass', **kwargs)
        # elif algorithm=='extratrees':
        #     model = instantiate_extratrees_classifier(trial)
        # elif algorithm=='adaboost':
        #     model = instantiate_adaboost_classifier(trial)
        # # elif algorithm=='knn':
        # #     model = instantiate_knn(trial)
    
    return model



########## PIPELINES #########


def instantiate_numerical_pipeline(trial : Trial, 
                                   imputation_transformer,
                                   pandarizer,
                                   interactions_transformer=FunctionTransformer()) -> Pipeline:
    pipeline = Pipeline([
        ('imputer', imputation_transformer),
        ("pandarizer", pandarizer),
        ('interactions', interactions_transformer),
        ('scaler', instantiate_robust_scaler(trial))
    ])
    return pipeline


def instantiate_categorical_pipeline(trial : Trial, 
                                     imputation_transformer) -> Pipeline:
    pipeline = Pipeline([
        ('imputer', imputation_transformer),
        ('encoder', instantiate_encoder(trial))
    ])
    return pipeline


def instantiate_processor(trial : Trial, 
                          numerical_columns : list[str], 
                          categorical_columns : list[str],
                          with_feature_selection : bool=False,
                          with_imputation : bool=False, 
                          with_interactions : bool=False,
                          ) -> ColumnTransformer:
    
    if with_feature_selection:
        selected_numerical_columns = choose_columns(trial, numerical_columns)
        selected_categorical_columns = choose_columns(trial, categorical_columns)
    else:
        selected_numerical_columns = numerical_columns
        selected_categorical_columns = categorical_columns

    if with_interactions:
        interactions_transformer = FeaturePairInteractions(operations=['+', '*', '/'])
    else:
        interactions_transformer = PassthroughTransformer()


    if with_imputation:
        numerical_imputation_transformer = instantiate_numerical_simple_imputer(trial)
        categorical_imputation_transformer = instantiate_categorical_simple_imputer(trial)
        numerical_pandarizer = FunctionTransformer(lambda x: pd.DataFrame(x, columns=selected_numerical_columns))
    else:
        # the FunctionTransformer without a function as argument applies the identity transformation
        numerical_imputation_transformer = PassthroughTransformer()
        categorical_imputation_transformer = PassthroughTransformer()
        numerical_pandarizer = PassthroughTransformer()


    numerical_pipeline = instantiate_numerical_pipeline(trial, 
                                                        numerical_imputation_transformer, 
                                                        numerical_pandarizer,
                                                        interactions_transformer)
    
    categorical_pipeline = instantiate_categorical_pipeline(trial, 
                                                            categorical_imputation_transformer,
                                                            )
  
    processor = ColumnTransformer([
      ('numerical_pipeline', numerical_pipeline, selected_numerical_columns),
      ('categorical_pipeline', categorical_pipeline, selected_categorical_columns)
      ])


    # processor = Pipeline([
    #     ('specific_processing', ColumnTransformer([
    #         ('numerical_pipeline', numerical_pipeline, selected_numerical_columns),
    #         ('categorical_pipeline', categorical_pipeline, selected_categorical_columns)
    #     ])),
    #     ('general_processing', Pipeline([
    #         ('feature_interaction', instantiate_feature_interaction(trial))
    #     ]))
    # ])

    return processor


def instantiate_model(trial : Trial, 
                      numerical_columns : list[str], 
                      categorical_columns : list[str], 
                      objective: str, 
                      algorithms: list[str],
                      with_feature_selection: bool=False,
                      with_imputation : bool=False, 
                      with_interactions : bool=False,
                      ) -> Pipeline:
  
    processor = instantiate_processor(trial, 
                                      numerical_columns, 
                                      categorical_columns,
                                      with_feature_selection,
                                      with_imputation,
                                      with_interactions)
  
    learner = instantiate_learner(trial, 
                                  objective, 
                                  algorithms)
  
    model = Pipeline([
      ('processor', processor),
      ('model', learner)
    ])
  
    return model


def objective_slow(trial : Trial, 
              X : pd.DataFrame, 
              y : pd.Series, 
              objective: str,
              algorithms: list[str],
              cv_scoring: str,
              numerical_columns : Optional[list[str]]=None, 
              categorical_columns : Optional[list[str]]=None, 
              random_state : int=0) -> float:
    
    """
    Old version of the optimizarion function without the pruning technique 

    """
    
    if numerical_columns is None:
        numerical_columns = [
            *X.select_dtypes(exclude=['object', 'category']).columns
        ]
        
    if categorical_columns is None:
        categorical_columns = [
            *X.select_dtypes(include=['object', 'category']).columns
        ]
    
    model = instantiate_model(trial, numerical_columns, categorical_columns, objective, algorithms)
    
    # kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    # roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    scores = cross_val_score(model, X, y, scoring=cv_scoring, cv=5, error_score='raise')
    
    return np.min([np.mean(scores), np.median([scores])])



def log_int(x, base : int=2):
  return np.floor(np.log(x)/np.log(base)).astype(int)



def generate_sample_numbers(y : pd.DataFrame, base : int, n_rungs : int) -> list[int]:
  
    data_size = len(y)
    data_scale = log_int(data_size, base)
    min_scale = data_scale - n_rungs
    min_samples = base**min_scale
  
    return [
        *map(lambda scale: base**scale, range(min_scale, data_scale+1))
    ]


def objective(trial: Trial, 
              X : pd.DataFrame, 
              y : pd.DataFrame, 
              objective: str,
              algorithms: list[str],
              cv_scoring: str,              
              numerical_columns : Optional[list[str]]=None, 
              categorical_columns : Optional[list[str]]=None, 
              with_feature_selection: bool=False,
              with_imputation : bool=False, 
              with_interactions : bool=False,
              random_state : int=0, 
              base : int=2, 
              n_rungs=4) -> float:
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=random_state
    )
    
    if numerical_columns is None:
        numerical_columns = [
            *X.select_dtypes(exclude=['object', 'category']).columns
        ]
        
    if categorical_columns is None:
        categorical_columns = [
            *X.select_dtypes(include=['object', 'category']).columns
        ]
    
    model = instantiate_model(trial, 
                              numerical_columns, 
                              categorical_columns, 
                              objective, 
                              algorithms, 
                              with_feature_selection, 
                              with_imputation, 
                              with_interactions)
    
    n_samples_list = generate_sample_numbers(y_train, base, n_rungs)
      
    for n_samples in n_samples_list:
        X_train_sample = X_train.sample(n_samples, random_state=random_state)
        y_train_sample = y_train.sample(n_samples, random_state=random_state)
        
        model.fit(X_train_sample, y_train_sample.values.ravel())
        
        
        if objective == 'classification':

            score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

        elif objective == 'regression':
        
            score = r2_score(y_test, model.predict(X_test))

        elif objective == 'multiclass':

            score = roc_auc_score(y_test, 
                                  model.predict_proba(X_test),
                                  multi_class='ovr',
                                  average="macro")

        trial.report(score, n_samples)
    
        if trial.should_prune():
            raise TrialPruned()

#     kfold = KFold(shuffle=True, random_state=random_state)
#     roc_auc = make_scorer(roc_auc_score, needs_proba=True)
    scores = cross_val_score(model, X, y, 
                             scoring=cv_scoring, 
                             cv=5, error_score='raise')
    return np.min([np.mean(scores), np.median(scores)])





################### MODEL EXPLAINABILITY ####################


def shap_partial_dependence_plots(preprocessed_df, shap_values, n_charts=12):

    """
    
    """

    importance_features = preprocessed_df.columns[np.argsort(-np.abs(shap_values).mean(0))]
    importance_features = importance_features[:n_charts]
    nrows = 4
    ncols = -(-len(importance_features) // nrows)

    fig, ax = plt.subplots(ncols, nrows, figsize=(20, 14))
    ax = ax.flatten()

    for i, f in enumerate(importance_features):

        shap.dependence_plot(
            f, 
            shap_values, 
            preprocessed_df, 
            interaction_index='auto', 
            alpha=.5, 
            x_jitter=.5,
            ax=ax[i], 
            show=False)

    # remove leftover facets
    for i in range(len(importance_features), nrows * ncols):
        ax[i].remove()
        
    fig.suptitle('SHAP, Partial Dependance', x=0, horizontalalignment='left', fontsize=25)
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()


def append_predictions(model, 
                       df, 
                       target_values,
                       target_name,
                       target_type, 
                       df_preprocessed=None, 
                       label_encoder=None):

    """
    Computes predictions from the model and appends it to the X_
    """

    if df_preprocessed is None:
        df_preprocessed = df

                            
    if target_type == 'continuous':

        predictions = model.predict(df)
        train_results = df_preprocessed.copy()
        train_results[target_name + '_prediction'] = predictions
        train_results[target_name] = target_values
        train_results['error'] = train_results[target_name] - train_results[target_name + '_prediction']
        # train_results = train_results.sort_values('error')
        train_results = train_results.reset_index()

    else: # target_type 'binary' or 'multiclass'

        train_results = df_preprocessed.copy()
        pred_cols = ['pred_prob_' + str(clss) for clss in label_encoder.classes_]
        predictions = model.predict_proba(df)
        train_results[pred_cols] = predictions
        train_results[target_name + '_prediction'] = label_encoder.inverse_transform(np.argmax(predictions, 1))
        train_results[target_name] = target_values
        train_results[target_name] = label_encoder.inverse_transform(train_results[target_name])
        train_results = train_results.reset_index()    

    return train_results



def confusion_matrix_plot(y_true, 
                          y_pred, 
                          labels,
                          cmap=plt.cm.Blues):

    """
    Plots a confusion matrix
    """

    cm = confusion_matrix(y_true, 
                          y_pred)
    
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=cmap)



def prediction_probability_distribution_plot(preds_df, 
                                             target_classes,
                                             target_colname,
                                             colors=px.colors.qualitative.G10,
                                             bin_size=.02,
                                             show_curve=True,
                                             renderer='notebook_connected'):
    """
    
    """    
    

    for target_class in target_classes:

        preds_class = preds_df[preds_df[target_colname] == target_class]
        pred_classes = ['pred_prob_' + str(clss) for clss in target_classes]
        if len(target_classes) == 2:
            pred_classes = pred_classes[-1:]
        data = [preds_class[pred] for pred in pred_classes]
                
        fig = ff.create_distplot(data, 
                                 pred_classes, 
                                 colors=colors,
                                 bin_size=bin_size,
                                 histnorm='probability',
                                 show_curve=show_curve,
                                 )

        # Add title
        fig.update_layout(title_text=f'Prediction Probability Distributions for {target_colname} {target_class}', 
                          yaxis_range=[0,0.5],
                          width=600,
                          height=600)
        
        fig.show(renderer=renderer, engine='orca')

##################### EDA #####################
        
import pandas as pd
import seaborn as sns
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np

def split_features(df,
                   target_col,
                   categorical_threshold
                   ):

    """
    Uses a threshold to determine if a column is a numerical or categorical type.
    If the number of unique values for a column is less than the 
    threshold it is considered categorical regardless of its dtype. If the number
    of unique values is greater than 10 and the column is not object it is 
    considered numerical
    """

    int_features = df.select_dtypes(include=['int'])
    unique_df = int_features.nunique()
    int_cat_features = unique_df[unique_df <= categorical_threshold].index.to_list()
    int_num_features = unique_df[unique_df > categorical_threshold].index.to_list()

    numerical_features = df.select_dtypes(exclude=['object', 'int', 'category']).columns.to_list() + int_num_features
    if target_col in numerical_features: numerical_features.remove(target_col)

    categorical_features = df.select_dtypes(include=['object', 'category']).columns.to_list() + int_cat_features
    if target_col in categorical_features: categorical_features.remove(target_col)
    df[categorical_features] = df[categorical_features].astype('object')

    print(f'The numerical features are: {numerical_features}')
    print(f'The categorical features are: {categorical_features}')

    return numerical_features, categorical_features


def pairplot(df,
             numerical_features,
             target_type,
             target_col,
             sample=0.5):


    if target_type != 'continuous':
        pairplot = sns.pairplot(df[numerical_features + [target_col]].sample(frac=sample), 
                                hue=target_col, 
                                corner=True)
    else:
        pairplot = sns.pairplot(df[numerical_features + [target_col]], 
                                corner=True)
        
        return pairplot


def train_test_distribution_plots(train_df,
                                  test_df,
                                  numerical_features,
                                  sample=0.5,
                                  renderer='notebook_connected'
                                  ):

    """
    """

    aux_train = train_df.sample(frac=sample).copy()
    aux_test = test_df.sample(frac=sample).copy()

    fig = make_subplots(rows=len(numerical_features),
                        cols=3,
                        row_titles=numerical_features,
                        column_titles=['Distribution', 'Train', 'Test'],
                        column_widths=[2, 1, 1],
                        )
    
    group_labels = ['Train', 'Test']
        
    for i, col in enumerate(numerical_features):
        
        fig2 = ff.create_distplot([aux_train[col].values, 
                                   aux_test[col].values], 
                                   group_labels)
    
        fig.add_trace(go.Scatter(fig2['data'][2],
                                 line=dict(color='rgb(9,56,125)', width=0.5),
                                 name='train'), 
                      row=i+1, col=1)
    
        fig.add_trace(go.Scatter(fig2['data'][3],
                                 line=dict(color='rgb(107,174,214)', width=0.5),
                                 name=None), 
                      row=i+1, col=1)
        
        fig.add_trace(go.Box(y=aux_train[col].values,
                             name='Train ' + col,
                             showlegend=False,
                             boxpoints=False,
                             pointpos=-1.8,
                             marker_color='rgb(9,56,125)',
                             line_color='rgb(9,56,125)',
                             marker_size=4),
                      row=i+1, col=2)
        
        fig.add_trace(go.Box(y=aux_test[col].values,
                            name='Test ' + col,
                            showlegend=False,
                            pointpos=-1.8,
                            boxpoints=False,
                            marker_color='rgb(107,174,214)',
                            line_color='rgb(107,174,214)',                     
                            marker_size=4),
                    row=i+1, col=3)
    
        
        fig.update_layout(height=len(numerical_features)*200, 
                          width=1000, 
                          title_text="Train vs Test Feature Distributions for Numerical Features", 
                          showlegend=True)
    
    # show static graph for Github rendering
    fig.show(renderer=renderer, engine='orca')



def train_test_categorical_piecharts(train_df,
                                     test_df,
                                     categorical_features,
                                     renderer='notebook_connected'):


    """

    """

    fig = make_subplots(rows=len(categorical_features),
                        cols=2,
                        specs=[[{"type": "pie"}, {"type": "pie"}]]*len(categorical_features),
                        row_titles=categorical_features,
                        column_titles=['Train', 'Test'],
                        )
    
    irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                     'rgb(175, 49, 35)', 'rgb(36, 73, 147)']
    
    for i, col in enumerate(categorical_features):
        
        train_col_counts = train_df[col].value_counts()
        test_col_counts = test_df[col].value_counts()
    
        fig.add_trace(go.Pie(
                            labels=train_col_counts.index.tolist(), 
                             values=train_col_counts.values.tolist(), 
                             marker_colors=irises_colors
                             ), 
                      row=i+1, col=1)
        
        fig.add_trace(go.Pie(
                            labels=test_col_counts.index.tolist(), 
                            values=test_col_counts.values.tolist(), 
                            marker_colors=irises_colors
                            ), 
                    row=i+1, col=2)
        
        fig.update_layout(height=len(categorical_features)*300, 
                          width=800, 
                          title_text="Train vs Test Feature Distributions for Categorical Features",
                          )
    
    # show static graph for Github rendering
    fig.show(renderer=renderer, engine='orca')

def correlation_plot(df):

    """
    Calculates the correlation between columns in a dataframe
    and shows them in a heatmap plot
    """

    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), 
                annot=True, 
                mask=mask, 
                cmap='RdBu_r', 
                vmin=-1, 
                vmax=1)
    plt.show()


################### CUSTOM TRANSFORMERS ####################

from sklearn.base import TransformerMixin, BaseEstimator


# Changed the base classes here, see Point 3
class PassthroughTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X = X
        return X

    def get_feature_names(self):
        return self.X.columns.tolist()


class DoubleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        self.columns = X.columns
        return X * 2

    def get_feature_names_out(self, feature_names):
        return [f'{col}_doubled' for col in self.columns]
    

# create a class to add a new feature AgeMedianByDistGroup
class AgeMedianByDistGroup(BaseEstimator, TransformerMixin):
    '''get the median age of each distance group''' 
    def __init__(self, train):
        self.age_median_by_dist_group = train.groupby('distance').apply(lambda x: x['age'].median())
        self.age_median_by_dist_group.name = 'age_median_by_dist_group'
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X, y=None):
        new_X = pd.merge(X, self.age_median_by_dist_group, 
                         left_on = 'distance', right_index=True, how='left')        
        X['age_median_by_dist_group'] = new_X['age_median_by_dist_group']
        return X
    

# create a class to add a new feature AgeMedianByDistGroup
class FeatureBasicInteractions(BaseEstimator, TransformerMixin):
    
    ''' Appends ''' 
    def __init__(self, train):
        self.age_median_by_dist_group = train.groupby('distance').apply(lambda x: x['age'].median())
        self.age_median_by_dist_group.name = 'age_median_by_dist_group'
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X, y=None):
        new_X = pd.merge(X, self.age_median_by_dist_group, 
                         left_on = 'distance', right_index=True, how='left')        
        X['age_median_by_dist_group'] = new_X['age_median_by_dist_group']
        return X


class FeaturePairInteractions(BaseEstimator, TransformerMixin):
    
    """
    This transformer creates new columns which are the result 
    of performing basic operations between two features in the dataframe.
    Operations supported for the current version are addition ('+'),
    subtraction ('-'), product ('*') and division ('/')
    """
    
    def __init__(self, operations=['+', '-','*', '/']):
        self.operations = operations
        self.expanded_features = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # initializing an empty DataFrame to store the expanded data
        expanded_df = pd.DataFrame()
        self.expanded_features = X.columns.to_list()

        # iterating over specified operations
        for op in self.operations:

            # iterating over pairs of columns in the input DataFrame
            for i, col1 in enumerate(X.columns):
                for j, col2 in enumerate(X.columns[i+1:]):

                    # creating a new column name representing the operation of the pair
                    new_col_name = col1 + op + col2
                    
                    # performing the operation on the pair of columns
                    if op == '+':
                        result_values = X[col1] + X[col2]
                    elif op == '*':
                        result_values = X[col1] * X[col2]
                    elif op == '/':
                        # checking if denominator is zero
                        zero_mask = X[col2] == 0
                        # if denominator is zero, assign a default value (e.g., NaN)
                        result_values = (X[col1] / X[col2]).where(~zero_mask, other=-1)
                            
                    # adding the new column to the expanded DataFrame
                    result_values.name = new_col_name
                    # expanded_df[new_col_name] = result_values
                    expanded_df = pd.concat([expanded_df, result_values], axis=1)
                    # storing the names of the expanded features
                    self.expanded_features.append(new_col_name)
        
        # Concatenate the original DataFrame with the expanded DataFrame
        expanded_df = pd.concat([X, expanded_df], axis=1)
        
        return expanded_df
    
    def get_feature_names_out(self):
        # Return the names of the expanded features
        return self.expanded_features