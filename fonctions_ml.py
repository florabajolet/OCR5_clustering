import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_validate, train_test_split, GridSearchCV

# MACHINE LEARNING FUNCTION AND PIPELINES

def multi_target_train_test(preprocessor, list_estimators, estimators_names, data, list_targets, list_targets_names, 
                            test_size=0.2,):
    """
    Perform a split into train and test set, fit and predict a model, as well as display a few metrics (score, MAE, MSE) 
    and plot predicted vs. real values of the target. Up to 9 different targets and corresponding estimators
    can be used (on the same data).
    
    preprocessor:
        Preprocessor used in the modeling (must be the same for all targets).
    
    list_estimators: list of str
        List of estimators, one for each target.
    
    estimators_names: str
        List of estimators names using string format (order corresponding to the list_estimators input).
    
    data:
        Data to split and on which fit and predict will be used (must be the same for all targets).
    
    list_targets: list
        List containing the targets to predict (order corresponding to the list_estimators input).
    
    list_targets_names: list of str
        List containing the targets names using string format (order corresponding to the list_targets input).
    
    test_size: int or float
        Size of test set, default is 0.2 meaning 20% of the total dataset.
    """
    
    # Check:
    if len(list_estimators)!=len(list_targets):
        print("Need one estimator per target.")
    if len(list_targets)!=len(list_targets_names):
        print("Need as many target names as targets.")
    if len(list_estimators)!=len(estimators_names):
        print("Need as many estimators names as estimators.")
    
    # Set-up figures:
    n = len(list_targets)
    if n == 1 :
        figsize = (6,6)
    elif n == 2:
        figsize =(16, 16)
    elif n == 3:
        figsize = (18, 16)
    elif n >= 4 and n <= 7:
        figsize = (18, 22)
    elif n >=8 and n<=9:
        figsize = (18, 30)
    else:
        print("Too many targets! maximum is 9.")

    subplot = range(331, 331+n)
    
    possible_colors = ["steelblue", "coral", "olivedrab", "teal", "peru", 
                       "mediumorchid", "seagreen", "crimson", "orange"]
    colors = possible_colors[:n]
    
    fig = plt.figure(figsize=figsize, tight_layout=True)
    
    fig.suptitle(f"Valeurs réelles vs. prédites sur le test set")
    
    for estimator, estimator_name, target, target_name, subplot, color in zip(list_estimators, estimators_names, 
                                                                                list_targets, list_targets_names, 
                                                                                subplot, colors):
        # Set-up pipeline:
        model = make_pipeline(preprocessor, estimator)
        est = str(estimator)
        
        # Make train-test and display metrics:
        data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42, 
                                                                            test_size=test_size)
        model.fit(data_train, target_train)
        target_predicted = model.predict(data_test)

        score = model.score(data_test, target_test)
        mae = mean_absolute_error(target_test, target_predicted)
        mse = mean_squared_error(target_test, target_predicted)
        print(f"{target_name}")
        print(f"Score {estimator_name} : {score:.2f}")
        print(f"MAE {estimator_name} : {mae:.2e}")
        print(f"MSE {estimator_name} : {mse:.2e}")
        
        # Get best parameters in case of cross-validation included:
        if "LinearRegression" in est:
            print(f"Regression coefficients: {[round(x,3) for x in model.named_steps['linearregression'].coef_]}")
            print(f"Regression intercept: {model.named_steps['linearregression'].intercept_:.3f}")
        elif "RidgeCV" in est:
            try:
                print(f"Ridge best alpha: {model.named_steps['ridgecv'].alpha_:.3f}")
            except KeyError:
                print(f"Ridge best alpha: {model.named_steps['pipeline']['ridgecv'].alpha_:.3f}")
            else:
                print(f"Ridge best alpha: {model.named_steps['ridgecv'].alpha_:.3f}")
        elif "LassoCV" in est:
            print(f"Lasso best alpha: {model.named_steps['lassocv'].alpha_:.3f}")
        elif "ElasticNetCV" in est:
            print(f"ElasticNet best alpha: {model.named_steps['elasticnetcv'].alpha_:.3f}")
        print(f"----------------------------------------")
        
        # Make plots
        ax = fig.add_subplot(subplot)
        ax.scatter(target_test, target_predicted, alpha=0.5, color=color)
        ax.plot(target_test, target_test, color="seagreen")
        ax.set_title(estimator_name)
        ax.set_xlabel(target_name)
        ax.set_ylabel("Prediction")
        ax.grid(True, linewidth=1);


def get_all_scores(model_name, cv_results, list_scores, train_score=False):
    """
    Create a Series containing metrics values from cross-validation output.
    Take in input the model name, the object containing the cross-validation results, 
    the list of metrics used in the CV and a boolean indicating if train scores were returned or not.
    Return a Series with fit time, score time and the mean of each metric defined in list_scores.
    
    model_name: str
        Model name that will be used to fill the first column of the final dataframe.
        
    cv_results:
        Output of a cross-validation.
    
    list_scores: list of str
        List of metrics used in the cross-valisation.
    
    train_score: boolean
        Whether the cross-validation contains train scores or not. Default is False.
    
    """
    
    # Make a dataframe of the cross_validation results and set some variables
    cv_results = pd.DataFrame(cv_results)
    test_scores = []
    train_scores = []
    n = len(list_scores)
    
    # Add "test_" to the name of list of metrics
    for i in list_scores:
        i = "test_" + i
        test_scores.append(i)
    
    # Create a Series with basic info
    mean_results = pd.DataFrame([[model_name, cv_results["fit_time"].mean(), cv_results["score_time"].mean()]],
                             columns=["model_name", "fit_time", "score_time"])
    
    # If train score was return by cross_validation, create a list with appropriate names
    if train_score==True:
        for i in list_scores:
            i = "train_" + i
            train_scores.append(i)
            
        # Add means for train metrics
        for i in range (0, n):
            mean_results[train_scores[i]] = cv_results[train_scores[i]].mean()
    
    # Add means for test metrics
    for i in range (0, n):
            mean_results[test_scores[i]] = cv_results[test_scores[i]].mean()
    
    return mean_results


def custom_cv(preprocessor, estimator_SEU, estimator_GHGE, name_model, data, target_SEU, target_GHGE, scoring, n_splits=30):
    """
    Make a cross-validation on the two targets to predict (site energy use and GHG emissions) using a shuffle split.
    Use the get_all_scores function to format the resulting metrics.
    Plot histograms of the mean absolute percentage error for train and test sets.
    Return the cross-validation results and Series with metrics for the two targets.
    
    preprocessor:
        Preprocessor to use in the modeling.
    
    estimator_SEU:
        Estimator to use to fit and predict Site Energy Use.
    
    estimator_GHGE:
        Estimator to use to fit and predict GHG emissions.
    
    name_model: str
        Name of the model that will be used in the Series returning the metrics of the cross-validation.
    
    data:
        Data that will be used in the cross-validation
    
    target_SEU:
        Site Energy Use target.
    
    target_GHGE:
        GHG Emissions target.
    
    scoring: list of str
        List of the metrics to use in the cross-validation.
    
    n_splits:
        Number of splits for the cross-validation (default is 30).
    """
    
    # Set-up models
    model_SEU = make_pipeline(preprocessor, estimator_SEU)
    model_GHGE = make_pipeline(preprocessor, estimator_GHGE)
    
    # Make suffle split and cross-validate
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)

    cv_results_SEU = cross_validate(model_SEU, data, target_SEU, cv=cv, n_jobs=-2, return_estimator=True, 
                                    return_train_score=True, scoring=scoring)
    cv_results_GHGE = cross_validate(model_GHGE, data, target_GHGE, cv=cv, n_jobs=-2, return_estimator=True, 
                                     return_train_score=True, scoring=scoring)
    
    # Format metrics output
    models_summary_SEU = get_all_scores(name_model, cv_results_SEU, scoring, train_score=True)
    models_summary_GHGE = get_all_scores(name_model, cv_results_GHGE, scoring, train_score=True)
    
    cv_results_SEU_df = pd.DataFrame(cv_results_SEU)
    cv_results_GHGE_df = pd.DataFrame(cv_results_GHGE)
    
    RMSE_SEU = -cv_results_SEU_df[["train_neg_root_mean_squared_error", "test_neg_root_mean_squared_error"]]
    
    RMSE_GHGE = -cv_results_GHGE_df[["train_neg_root_mean_squared_error","test_neg_root_mean_squared_error"]]
    
    # Plot histograms of train vs. test error
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    sns.histplot(ax=ax[0], data=RMSE_SEU, legend=False)
    ax[0].legend(labels=["Validation", "Train"])
    ax[0].set_xlabel("Root mean squared error")
    ax[0].set_title("Train vs. Validation set\nSite Energy Use (kBtu)")

    sns.histplot(ax=ax[1], data=RMSE_GHGE, legend=False)
    ax[1].legend(labels=["Validation", "Train"])
    ax[1].set_xlabel("Root mean squared error")
    ax[1].set_title("Train vs. Validation set\nGHGE");
    
    return cv_results_SEU, cv_results_GHGE, models_summary_SEU, models_summary_GHGE


def GS_eval(preprocessor, estimator, model_name, data_train, list_target_train, params_GS, scoring, refit,
            data_test, list_target_test, list_targets_names, n_splits=10, test_size=0.2):
    
    """
    Make a shuffle split, then a grid search including cross-validation (GridSearchCV), keep the best 
    hyperparameters and use them to predict the targets on a separate dataset. Then, display predicted vs. 
    real values of the targets.
    Up to 9 targets can be used (prediction from the same data).
    
    preprocessor:
        Preprocessor to be used in the modeling (same for all targets).
    
    estimator:
        Estimator to be used in the modeling (same for all targets).
        
    model_name: str
        Model name used in return scoring results.
    
    data_train:
        Data used to make the GridSearchCV (will be split into train and test sets for cross-validation incuded 
        into the grid search).
    
    list_target_train: list
        List of targets (maximum is 9).
    
    params_GS: dict
        Dictionnary containing the hyperparameters to test.
    
    scoring: str or list of str
        Metrics that will be stored in the dataframe output summarizing results (string format as used in 
        grid search and cross-validate functions of sklearn).
    
    refit: str
        Metric used to select the best model.
    
    data_test:
        Data used to predict the targets with the best hyperparameters.
    
    list_target_test: list
        List of targets to predict with the best hyperparameters.
    
    list_targets_names: list of str
        List of targets names for display.
    
    n_splits: int
        Number of splits for the shuffle split procedure. Default is 10.
    
    test_size: int of float
        Test size of the shuffle split. Default is 0.2, meaning 20% of the data kept for grid search testing.
    
    """
    
    # Check:
    if len(list_target_train)!=len(list_target_test):
        print("Need as many targets for train and test.")
    if len(list_target_train)!=len(list_targets_names):
        print("Need as many target names as targets.")
    
    # Set-up figures:
    n = len(list_targets_names)
    if n == 1 :
        figsize = (6,6)
    elif n == 2:
        figsize =(16, 16)
    elif n == 3:
        figsize = (18, 16)
    elif n >= 4 and n <= 7:
        figsize = (18, 22)
    elif n >=8 and n<=9:
        figsize = (18, 30)
    else:
        print("Too many targets! maximum is 9.")

    subplot = range(331, 331+n)
    possible_colors = ["steelblue", "coral", "olivedrab", "teal", "peru", 
                       "mediumorchid", "seagreen", "crimson", "orange"]
    colors = possible_colors[:n]
    fig = plt.figure(figsize=figsize, tight_layout=True)
    fig.suptitle(f"Valeurs réelles vs. prédites sur le test set avec les meilleurs paramètres")
    
    # Set-up pipeline
    model=make_pipeline(preprocessor, estimator)
    
    # Make a grid search from a shuffle split:
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    
    model_grid_search = GridSearchCV(model, param_grid=params_GS, cv=cv, n_jobs=-2, scoring=scoring, refit=refit,
                                     return_train_score=True)
    
    # Modify metrics names to retrieve cross-validation results from output:
    train_scores = ["mean_fit_time", "mean_score_time"]
    for score in scoring:
        score = "mean_train_" + score
        train_scores.append(score)
    rank_refit = "rank_test_" + refit
    model_name_df = pd.DataFrame([model_name], columns=["model_name"])
    
    # Define dictionnary that will contain results for all targets:
    mean_results = {}
            
    for target_train, target_name, target_test, subplot, color in zip(list_target_train, list_targets_names, 
                                                                      list_target_test, subplot, colors):
        
        # Fit model and get best params:
        model_grid_search.fit(data_train, target_train)

        print(f"Best params {target_name}: {model_grid_search.best_params_}")
        
        # Get train scores for all metrics:
        cv_results = pd.DataFrame(model_grid_search.cv_results_)
        best_model_results = cv_results[rank_refit]==1
        train_results = cv_results[best_model_results][train_scores]
        train_results.reset_index(inplace=True)
        
        # Use model with best params to evaluate it on a separate test set:
        target_predicted = model_grid_search.predict(data_test)
        
        # Get a few metrics on the test set:
        score = model_grid_search.score(data_test, target_test)
        mae = mean_absolute_error(target_test, target_predicted)
        mape = mean_absolute_percentage_error(target_test, target_predicted)
        mse = mean_squared_error(target_test, target_predicted)
        rmse = np.sqrt(mean_squared_error(target_test, target_predicted))
        r2 = r2_score(target_test, target_predicted)
        
        # Make a dataframe containing train and test scores and insert it in dictionnary:
        test_results = []
        test_results = pd.DataFrame([[-mae, -mape, -mse, -rmse, r2]],
                                  columns=["test_neg_mean_absolute_error", "test_neg_mean_absolute_percentage_error",
                                          "test_neg_mean_squared_error", "test_neg_root_mean_squared_error", "test_r2"])
        results_target = pd.concat([model_name_df, train_results, test_results], axis=1)
        results_target.drop("index", axis=1, inplace=True)
        results_target.rename(columns={"mean_fit_time": "fit_time", "mean_score_time":"score_time", 
                               "mean_train_neg_mean_absolute_error":"train_neg_mean_absolute_error",
                              "mean_train_neg_mean_absolute_percentage_error":"train_neg_mean_absolute_percentage_error",
                              "mean_train_neg_mean_squared_error":"train_neg_mean_squared_error",
                              "mean_train_neg_root_mean_squared_error":"train_neg_root_mean_squared_error",
                              "mean_train_r2":"train_r2"}, inplace=True)
        mean_results[target_name] = results_target
        
        # Display a few test metrics: 
        print(f"Score : {score:.2f}")
        print(f"MAE : {mae:.2e}")
        print(f"RMSE : {rmse:.2e}")
        print("-----------------------------------------")
        
        # Plot predicted vs. real values:
        ax = fig.add_subplot(subplot)
        ax.scatter(target_test, target_predicted, alpha=0.5, color=color)
        ax.plot(target_test, target_test, color="seagreen")
        ax.set_xlabel(target_name)
        ax.set_ylabel("Prediction")
        ax.grid(True, linewidth=1);
    
    return mean_results



def linearregression_cv(preprocessor, estimator, alphas, model_name, data_train, list_target_train,
            data_test, list_target_test, list_targets_names):
    """
    To use when doing a linear regression including a cross-validation (RidgeCV, LassoCV, ElasticNetCV).
    Take in argument preprocessor, estimator, target(s), data and target train and test sets, and tested alphas.
    Display best alpha found and main metrics. Plot predicted vs. true values for each target as well
    as MSE vs. alphas curve.
    Return a dictionnary containing a dataframe with cross-validation results for each target 
    (key = target name as in the input list_targets_names, value = dataframe with cv results).
    Alphas must be given both in the estimator arguments and as a function argument.
    store_cv_values must be set to True if using RidgeCV.
    When evaluating the model on the test set, will use the best alpha found with default settings of Ridge, 
    Lasso or ElasticNet functions.
    
    preprocessor:
        Preprocessor to be used in the modeling (same for all targets).
    
    estimator:
        Estimator to be used in the modeling (same for all targets). Must specify alphas to test in the 
        estimator's arguments. store_cv_values must be set to True if using RidgeCV.
        
    alphas: list of int or float, np.array
        Alphas to be tested in the cross-validation (must be the same as in estimator's argument).
        
    model_name: str
        Model name used in return scoring results.
    
    data_train:
        Data used to make the GridSearchCV (will be split into train and test sets for cross-validation incuded 
        into the grid search).
    
    list_target_train: list
        List of targets.
    
    data_test:
        Data used to predict the targets with the best hyperparameters.
    
    list_target_test: list
        List of targets to predict with the best hyperparameters.
    
    list_targets_names: list of str
        List of targets names for display.
    """
    
    # Check:
    if len(list_target_train)!=len(list_target_test):
        print("Need as many targets for train and test.")
    if len(list_target_train)!=len(list_targets_names):
        print("Need as many target names as targets.")
    
    # Set-up figures:
    n = len(list_target_train)
    axes = range(n)
    possible_colors = ["steelblue", "coral", "olivedrab", "teal", "peru", 
                       "mediumorchid", "seagreen", "crimson", "orange"]
    colors = possible_colors[:n]
    fig, ax = plt.subplots(n, 2, figsize=(10, n*5), tight_layout=True)
    fig.suptitle(f"{model_name}")
    
    # Set-up pipeline
    model = make_pipeline(preprocessor, estimator)
    est = str(estimator)
    
    # Define dictionnary that will contain cross-validation results for all targets:
    results_test = {}
    
    # For each taget:
    for axe, target_train, target_name, target_test, color in zip(axes, list_target_train, list_targets_names, 
                                                                      list_target_test, colors):
        
        # Fit model:
        model.fit(data_train, target_train)
        
        # Get best alpha and results of cross-validation:
        print(f"{target_name}")
        if "RidgeCV" in est:
            try:
                print(f"Ridge best alpha: {model.named_steps['ridgecv'].alpha_:.3f}")
            except KeyError:
                cv_results = model.named_steps['pipeline']['ridgecv'].cv_values_
                best_alpha = model.named_steps['pipeline']['ridgecv'].alpha_
                print(f"Ridge best alpha: {best_alpha:.3f}")
            else:
                cv_results = model.named_steps['ridgecv'].cv_values_
                best_alpha = model.named_steps['ridgecv'].alpha_
                print(f"Ridge best alpha: {best_alpha:.3f}")
            
            # Make model with best alpha:
            #best_model = make_pipeline(preprocessor, Ridge(alpha=best_alpha, random_state=0))
            # Make MSE vs. alpha Series:
            errors = pd.DataFrame(cv_results)
            errors.columns = list(alphas)
            errors_alphas = errors.mean()
            errors_alphas_std = errors.std()
            
        elif "LassoCV" in est:
            best_alpha = model.named_steps['lassocv'].alpha_
            print(f"Lasso best alpha: {best_alpha:.3f}")
            
            # Make model with best alpha:
            #best_model = make_pipeline(preprocessor, Lasso(alpha=best_alpha, random_state=0))
            
            # Get MSE paths and make MSE vs. alphas Series:
            mse = model.named_steps['lassocv'].mse_path_.mean(axis=1)
            errors_alphas = pd.Series(mse, index=alphas)
            
            #eps = alphas.min() / alphas.max()
            #alphas_lasso, coefs_lasso, _ = lasso_path(data_train, target_train, eps=eps, alphas=alphas)
            
        elif "ElasticNetCV" in est:
            best_alpha = model.named_steps['elasticnetcv'].alpha_
            print(f"ElasticNet best alpha: {best_alpha:.3f}")
            
            # Make model with best alpha:
            #best_model = make_pipeline(preprocessor, ElasticNet(alpha=best_alpha, random_state=0))
            
            # Get MSE paths and make MSE vs. alphas Series:
            mse = model.named_steps['elasticnetcv'].mse_path_.mean(axis=1)
            errors_alphas = pd.Series(mse, index=alphas)
            
        # Use model with best alpha to evaluate it on a separate test set:
        #best_model.fit(data_train, target_train)
        #target_predicted_train = best_model(data_train)
        target_predicted = model.predict(data_test)
        
        # Get metrics on the test set:
        score = model.score(data_test, target_test)
        mae = mean_absolute_error(target_test, target_predicted)
        mape = mean_absolute_percentage_error(target_test, target_predicted)
        mse = mean_squared_error(target_test, target_predicted)
        rmse = np.sqrt(mean_squared_error(target_test, target_predicted))
        r2 = r2_score(target_test, target_predicted)
        results_test[target_name] = pd.DataFrame([[-mae, -mape, -mse, -rmse, r2]], 
                                    columns=["test_neg_mean_absolute_error", 
                                             "test_neg_mean_absolute_percentage_error", 
                                             "test_neg_mean_squared_error", 
                                             "test_neg_root_mean_squared_error", "test_r2"])
        # Display a few test metrics: 
        print(f"Score : {score:.2f}")
        print(f"MAE : {mae:.2e}")
        print(f"RMSE : {rmse:.2e}")
        print("-----------------------------------------")
        
        # Plot predicted vs. real values:
        ax[axe, 0].scatter(target_test, target_predicted, alpha=0.5, color=color)
        ax[axe, 0].plot(target_test, target_test, color="seagreen")
        ax[axe, 0].set_xlabel(target_name)
        ax[axe, 0].set_ylabel("Prediction")
        ax[axe, 0].set_title("Valeurs réelles vs. prédites sur le test set\navec le meilleur alpha")
        ax[axe, 0].grid(True, linewidth=1)
        
        # Plot errors vs. alphas:
        ax[axe, 1].plot(errors_alphas, marker="+", color=color)
        ax[axe, 1].semilogx()
        ax[axe, 1].set_ylabel(f"MSE pour {target_name}")
        ax[axe, 1].set_xlabel("Alpha")
        ax[axe, 1].set_title("Erreur quadratique moyenne pour les alphas testés", pad=20)
        
        #ax[axe, 2].plot(alphas_lasso, coefs_lasso, marker="+", color=color)
        #ax[axe, 2].semilogx()
        #ax[axe, 2].set_ylabel(f"Coef pour {target_name}")
        #ax[axe, 2].set_xlabel("Alpha")
        #ax[axe, 2].set_title("Coefficients en fonction de alpha", pad=20);
        
    return results_test