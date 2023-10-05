import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import boxcox
from pygam import LinearGAM
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

#Version de scikit-learn: 1.2.0

def evaluate_regression_models(X_train, X_test, y_train, y_test, intercept=True):
    
    if intercept == True:
        # Train and evaluate linear regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_mae = mean_absolute_error(y_test, lr_preds)
        lr_mse = mean_squared_error(y_test, lr_preds)
        lr_r2 = r2_score(y_test, lr_preds)

        # Winsorization
        def perform_winsorization(data, lower_percentile=5, upper_percentile=95):
            lower_limit = np.percentile(data, lower_percentile)
            upper_limit = np.percentile(data, upper_percentile)
            winsorized_data = np.clip(data, lower_limit, upper_limit)
            return winsorized_data

        # Train and evaluate linear regression after winsorization
        X_train_winsorized = perform_winsorization(X_train)
        X_test_winsorized = perform_winsorization(X_test)
        lr_winsorized = LinearRegression()
        lr_winsorized.fit(X_train_winsorized, y_train)
        lr_winsorized_preds = lr_winsorized.predict(X_test_winsorized)
        lr_winsorized_mae = mean_absolute_error(y_test, lr_winsorized_preds)
        lr_winsorized_mse = mean_squared_error(y_test, lr_winsorized_preds)
        lr_winsorized_r2 = r2_score(y_test, lr_winsorized_preds)

        # Logarithmic transformation to the target variable (y)
        y_train_log = np.log(y_train)
        y_test_log = np.log(y_test)

        # Train and evaluate linear regression with logarithmic target
        lr_log = LinearRegression()
        lr_log.fit(X_train, y_train_log)
        lr_log_preds = lr_log.predict(X_test)
        lr_log_mae = mean_absolute_error(y_test_log, lr_log_preds)
        lr_log_mse = mean_squared_error(y_test_log, lr_log_preds)
        lr_log_r2 = r2_score(y_test_log, lr_log_preds)

        # Box-Cox transformation to the target variable (y)
        y_train_boxcox, lambda_boxcox = boxcox(y_train)
        y_test_boxcox = boxcox(y_test, lambda_boxcox)

        # Train and evaluate linear regression with Box-Cox
        lr = LinearRegression()
        lr.fit(X_train, y_train_boxcox)
        lr_preds_boxcox = lr.predict(X_test)
        #lr_preds = (lr_preds_boxcox * lambda_boxcox) + 1 
        lr_bc_mae = mean_absolute_error(y_test, lr_preds_boxcox)
        lr_bc_mse = mean_squared_error(y_test, lr_preds_boxcox)
        lr_bc_r2 = r2_score(y_test, lr_preds_boxcox)

        # Train and evaluate robust regression (Huber Regressor)
        huber = HuberRegressor()
        huber.fit(X_train, y_train)
        huber_preds = huber.predict(X_test)
        huber_mae = mean_absolute_error(y_test, huber_preds)
        huber_mse = mean_squared_error(y_test, huber_preds)
        huber_r2 = r2_score(y_test, huber_preds)

        # Standard Scaler before Ridge
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and evaluate Ridge regression with Standard Scaler
        ridge = Ridge()
        ridge.fit(X_train_scaled, y_train)
        ridge_preds = ridge.predict(X_test_scaled)
        ridge_mae = mean_absolute_error(y_test, ridge_preds)
        ridge_mse = mean_squared_error(y_test, ridge_preds)
        ridge_r2 = r2_score(y_test, ridge_preds)

        # Train and evaluate Lasso regression
        lasso = Lasso()
        lasso.fit(X_train, y_train)
        lasso_preds = lasso.predict(X_test)
        lasso_mae = mean_absolute_error(y_test, lasso_preds)
        lasso_mse = mean_squared_error(y_test, lasso_preds)
        lasso_r2 = r2_score(y_test, lasso_preds)

        # Train and evaluate polynomial regression
        polyreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        polyreg.fit(X_train, y_train)
        polyreg_preds = polyreg.predict(X_test)
        polyreg_mae = mean_absolute_error(y_test, polyreg_preds)
        polyreg_mse = mean_squared_error(y_test, polyreg_preds)
        polyreg_r2 = r2_score(y_test, polyreg_preds)

        # Train and evaluate Generalized Additive Model (GAM)
        gam = LinearGAM().fit(X_train, y_train)
        gam_preds = gam.predict(X_test)
        gam_mae = mean_absolute_error(y_test, gam_preds)
        gam_mse = mean_squared_error(y_test, gam_preds)
        gam_r2 = r2_score(y_test, gam_preds)

        # Train and evaluate decision tree
        dt = DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_test)
        dt_mae = mean_absolute_error(y_test, dt_preds)
        dt_mse = mean_squared_error(y_test, dt_preds)
        dt_r2 = r2_score(y_test, dt_preds)

        # Train and evaluate Random Forest Regression
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_preds)
        rf_mse = mean_squared_error(y_test, rf_preds)
        rf_r2 = r2_score(y_test, rf_preds)

        # Train and evaluate K-Nearest Neighbors (KNN) Regressor
        knn = KNeighborsRegressor()
        knn.fit(X_train, y_train)
        knn_preds = knn.predict(X_test)
        knn_mae = mean_absolute_error(y_test, knn_preds)
        knn_mse = mean_squared_error(y_test, knn_preds)
        knn_r2 = r2_score(y_test, knn_preds)

         # Train and evaluate Support Vector Regression (SVR)
        svr = SVR()
        svr.fit(X_train, y_train)
        svr_preds = svr.predict(X_test)
        svr_mae = mean_absolute_error(y_test, svr_preds)
        svr_mse = mean_squared_error(y_test, svr_preds)
        svr_r2 = r2_score(y_test, svr_preds)

        # Train and evaluate Gradient Boosting Regressor
        gb = GradientBoostingRegressor()
        gb.fit(X_train, y_train)
        gb_preds = gb.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_preds)
        gb_mse = mean_squared_error(y_test, gb_preds)
        gb_r2 = r2_score(y_test, gb_preds)

        # Train and evaluate XGBoost Regressor
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        xgb_mse = mean_squared_error(y_test, xgb_preds)
        xgb_r2 = r2_score(y_test, xgb_preds)

        # Train and evaluate Quantile Regression
        quantile_reg = sm.QuantReg(y_train, X_train).fit(q=0.5)
        quantile_reg_preds = quantile_reg.predict(X_test)
        quantile_reg_mae = mean_absolute_error(y_test, quantile_reg_preds)
        quantile_reg_mse = mean_squared_error(y_test, quantile_reg_preds)
        quantile_reg_r2 = r2_score(y_test, quantile_reg_preds)

        # Train and evaluate neural network
        nn = MLPRegressor(hidden_layer_sizes=(50, 50))
        nn.fit(X_train, y_train)
        nn_preds = nn.predict(X_test)
        nn_mae = mean_absolute_error(y_test, nn_preds)
        nn_mse = mean_squared_error(y_test, nn_preds)
        nn_r2 = r2_score(y_test, nn_preds)

        # Create a dictionary to store results
        results = {
            'Linear Regression': (lr_mae, lr_mse, lr_r2),
            'Winsorization + Linear Regression': (lr_winsorized_mae, lr_winsorized_mse, lr_winsorized_r2),
            'Log + Linear Regression': (lr_log_mae, lr_log_mse, lr_log_r2),
            'Box-Cox + Linear Regression': (lr_bc_mae, lr_bc_mse, lr_bc_r2),
            'Robust Regression (Huber)': (huber_mae, huber_mse, huber_r2),
            'Ridge Regression': (ridge_mae, ridge_mse, ridge_r2),
            'Lasso Regression': (lasso_mae, lasso_mse, lasso_r2),
            'Polynomial Regression': (polyreg_mae, polyreg_mse, polyreg_r2),
            'GAM Regression': (gam_mae, gam_mse, gam_r2),
            'Decision Tree': (dt_mae, dt_mse, dt_r2),
            'Random Forest Regression': (rf_mae, rf_mse, rf_r2),
            'K-Nearest Neighbors (KNN) Regressor': (knn_mae, knn_mse, knn_r2),
            'Support Vector Regression' : (svr_mae, svr_mse, svr_r2),
            'Gradient Boosting Regressor' : (gb_mae, gb_mse, gb_r2),
            'XGBoost Regressor' : (xgb_mae, xgb_mse, xgb_r2),
            'Quantile Regression' : (quantile_reg_mae, quantile_reg_mse, quantile_reg_r2),
            'Neural Network': (nn_mae, nn_mse, nn_r2),
        }

        # Create a DataFrame from the results dictionary
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MAE', 'MSE', 'R-squared'])
        pd.options.display.float_format = '{:.3f}'.format

        # Sort the DataFrame by 'MAE' in ascending order, then by 'MSE' in ascending order
        results_df = results_df.sort_values(by=['MAE', 'MSE', 'R-squared'], ascending=[True, True, False])

        print(results_df)
        return results_df
    
    else:
        # Train and evaluate linear regression
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_mae = mean_absolute_error(y_test, lr_preds)
        lr_mse = mean_squared_error(y_test, lr_preds)
        lr_r2 = r2_score(y_test, lr_preds)

        # Winsorization
        def perform_winsorization(data, lower_percentile=5, upper_percentile=95):
            lower_limit = np.percentile(data, lower_percentile)
            upper_limit = np.percentile(data, upper_percentile)
            winsorized_data = np.clip(data, lower_limit, upper_limit)
            return winsorized_data

        # Train and evaluate linear regression after winsorization
        X_train_winsorized = perform_winsorization(X_train)
        X_test_winsorized = perform_winsorization(X_test)
        lr_winsorized = LinearRegression(fit_intercept=False)
        lr_winsorized.fit(X_train_winsorized, y_train)
        lr_winsorized_preds = lr_winsorized.predict(X_test_winsorized)
        lr_winsorized_mae = mean_absolute_error(y_test, lr_winsorized_preds)
        lr_winsorized_mse = mean_squared_error(y_test, lr_winsorized_preds)
        lr_winsorized_r2 = r2_score(y_test, lr_winsorized_preds)

        # Logarithmic transformation to the target variable (y)
        y_train_log = np.log(y_train)
        y_test_log = np.log(y_test)

        # Train and evaluate linear regression with logarithmic target
        lr_log = LinearRegression(fit_intercept=False)
        lr_log.fit(X_train, y_train_log)
        lr_log_preds = lr_log.predict(X_test)
        lr_log_mae = mean_absolute_error(y_test_log, lr_log_preds)
        lr_log_mse = mean_squared_error(y_test_log, lr_log_preds)
        lr_log_r2 = r2_score(y_test_log, lr_log_preds)

        # Box-Cox transformation to the target variable (y)
        y_train_boxcox, lambda_boxcox = boxcox(y_train)
        y_test_boxcox = boxcox(y_test, lambda_boxcox)

        # Train and evaluate linear regression with Box-Cox
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_train, y_train_boxcox)
        lr_preds_boxcox = lr.predict(X_test)
        #lr_preds = (lr_preds_boxcox * lambda_boxcox) + 1 
        lr_bc_mae = mean_absolute_error(y_test, lr_preds_boxcox)
        lr_bc_mse = mean_squared_error(y_test, lr_preds_boxcox)
        lr_bc_r2 = r2_score(y_test, lr_preds_boxcox)

        # Train and evaluate robust regression (Huber Regressor)
        huber = HuberRegressor(fit_intercept=False)
        huber.fit(X_train, y_train)
        huber_preds = huber.predict(X_test)
        huber_mae = mean_absolute_error(y_test, huber_preds)
        huber_mse = mean_squared_error(y_test, huber_preds)
        huber_r2 = r2_score(y_test, huber_preds)

        # Standard Scaler before Ridge
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and evaluate Ridge regression with Standard Scaler
        ridge = Ridge(fit_intercept=False)
        ridge.fit(X_train_scaled, y_train)
        ridge_preds = ridge.predict(X_test_scaled)
        ridge_mae = mean_absolute_error(y_test, ridge_preds)
        ridge_mse = mean_squared_error(y_test, ridge_preds)
        ridge_r2 = r2_score(y_test, ridge_preds)

        # Train and evaluate Lasso regression
        lasso = Lasso(fit_intercept=False)
        lasso.fit(X_train, y_train)
        lasso_preds = lasso.predict(X_test)
        lasso_mae = mean_absolute_error(y_test, lasso_preds)
        lasso_mse = mean_squared_error(y_test, lasso_preds)
        lasso_r2 = r2_score(y_test, lasso_preds)

        # Train and evaluate polynomial regression
        polyreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression(fit_intercept=False))
        polyreg.fit(X_train, y_train)
        polyreg_preds = polyreg.predict(X_test)
        polyreg_mae = mean_absolute_error(y_test, polyreg_preds)
        polyreg_mse = mean_squared_error(y_test, polyreg_preds)
        polyreg_r2 = r2_score(y_test, polyreg_preds)

        # Train and evaluate Generalized Additive Model (GAM)
        gam = LinearGAM(fit_intercept=False).fit(X_train, y_train)
        gam_preds = gam.predict(X_test)
        gam_mae = mean_absolute_error(y_test, gam_preds)
        gam_mse = mean_squared_error(y_test, gam_preds)
        gam_r2 = r2_score(y_test, gam_preds)

        # Train and evaluate decision tree
        dt = DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_test)
        dt_mae = mean_absolute_error(y_test, dt_preds)
        dt_mse = mean_squared_error(y_test, dt_preds)
        dt_r2 = r2_score(y_test, dt_preds)

        # Train and evaluate Random Forest Regression
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_preds)
        rf_mse = mean_squared_error(y_test, rf_preds)
        rf_r2 = r2_score(y_test, rf_preds)

        # Train and evaluate K-Nearest Neighbors (KNN) Regressor
        knn = KNeighborsRegressor()
        knn.fit(X_train, y_train)
        knn_preds = knn.predict(X_test)
        knn_mae = mean_absolute_error(y_test, knn_preds)
        knn_mse = mean_squared_error(y_test, knn_preds)
        knn_r2 = r2_score(y_test, knn_preds)

         # Train and evaluate Support Vector Regression (SVR)
        svr = SVR()
        svr.fit(X_train, y_train)
        svr_preds = svr.predict(X_test)
        svr_mae = mean_absolute_error(y_test, svr_preds)
        svr_mse = mean_squared_error(y_test, svr_preds)
        svr_r2 = r2_score(y_test, svr_preds)

        # Train and evaluate Gradient Boosting Regressor
        gb = GradientBoostingRegressor()
        gb.fit(X_train, y_train)
        gb_preds = gb.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_preds)
        gb_mse = mean_squared_error(y_test, gb_preds)
        gb_r2 = r2_score(y_test, gb_preds)

        # Train and evaluate XGBoost Regressor
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        xgb_mse = mean_squared_error(y_test, xgb_preds)
        xgb_r2 = r2_score(y_test, xgb_preds)

        # Train and evaluate Quantile Regression
        quantile_reg = sm.QuantReg(y_train, X_train).fit(q=0.5)
        quantile_reg_preds = quantile_reg.predict(X_test)
        quantile_reg_mae = mean_absolute_error(y_test, quantile_reg_preds)
        quantile_reg_mse = mean_squared_error(y_test, quantile_reg_preds)
        quantile_reg_r2 = r2_score(y_test, quantile_reg_preds)

        # Train and evaluate neural network
        nn = MLPRegressor(hidden_layer_sizes=(50, 50))
        nn.fit(X_train, y_train)
        nn_preds = nn.predict(X_test)
        nn_mae = mean_absolute_error(y_test, nn_preds)
        nn_mse = mean_squared_error(y_test, nn_preds)
        nn_r2 = r2_score(y_test, nn_preds)

        # Create a dictionary to store results
        results = {
            'Linear Regression': (lr_mae, lr_mse, lr_r2),
            'Winsorization + Linear Regression': (lr_winsorized_mae, lr_winsorized_mse, lr_winsorized_r2),
            'Log + Linear Regression': (lr_log_mae, lr_log_mse, lr_log_r2),
            'Box-Cox + Linear Regression': (lr_bc_mae, lr_bc_mse, lr_bc_r2),
            'Robust Regression (Huber)': (huber_mae, huber_mse, huber_r2),
            'Ridge Regression': (ridge_mae, ridge_mse, ridge_r2),
            'Lasso Regression': (lasso_mae, lasso_mse, lasso_r2),
            'Polynomial Regression': (polyreg_mae, polyreg_mse, polyreg_r2),
            'GAM Regression': (gam_mae, gam_mse, gam_r2),
            'Decision Tree': (dt_mae, dt_mse, dt_r2),
            'Random Forest Regression': (rf_mae, rf_mse, rf_r2),
            'K-Nearest Neighbors (KNN) Regressor': (knn_mae, knn_mse, knn_r2),
            'Support Vector Regression' : (svr_mae, svr_mse, svr_r2),
            'Gradient Boosting Regressor' : (gb_mae, gb_mse, gb_r2),
            'XGBoost Regressor' : (xgb_mae, xgb_mse, xgb_r2),
            'Quantile Regression' : (quantile_reg_mae, quantile_reg_mse, quantile_reg_r2),
            'Neural Network': (nn_mae, nn_mse, nn_r2),
        }

        # Create a DataFrame from the results dictionary
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MAE', 'MSE', 'R-squared'])
        pd.options.display.float_format = '{:.3f}'.format

        # Sort the DataFrame by 'MAE' in ascending order, then by 'MSE' in ascending order
        results_df = results_df.sort_values(by=['MAE', 'MSE', 'R-squared'], ascending=[True, True, False])

        print(results_df)
        return results_df
    
    
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification_models(X_train, X_test, y_train, y_test, cf_matrix=True):
    # Create a dictionary to store results
    results = {}

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    results['Logistic Regression'] = (accuracy_score(y_test, lr_preds),
                                      recall_score(y_test, lr_preds),
                                      precision_score(y_test, lr_preds),
                                      f1_score(y_test, lr_preds))

    # Bagging Classifier
    bagging = BaggingClassifier()
    bagging.fit(X_train, y_train)
    bagging_preds = bagging.predict(X_test)
    results['Bagging Classifier'] = (accuracy_score(y_test, bagging_preds),
                                     recall_score(y_test, bagging_preds),
                                     precision_score(y_test, bagging_preds),
                                     f1_score(y_test, bagging_preds))

    # Random Forest Classifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results['Random Forest Classifier'] = (accuracy_score(y_test, rf_preds),
                                           recall_score(y_test, rf_preds),
                                           precision_score(y_test, rf_preds),
                                           f1_score(y_test, rf_preds))

    # Support Vector Classifier (SVC) with RBF Kernel
    svc_rbf = SVC(kernel='rbf')
    svc_rbf.fit(X_train, y_train)
    svc_rbf_preds = svc_rbf.predict(X_test)
    results['SVM with RBF Kernel'] = (accuracy_score(y_test, svc_rbf_preds),
                                      recall_score(y_test, svc_rbf_preds),
                                      precision_score(y_test, svc_rbf_preds),
                                      f1_score(y_test, svc_rbf_preds))

    # Support Vector Classifier (SVC) with Polynomial Kernel
    svc_poly = SVC(kernel='poly')
    svc_poly.fit(X_train, y_train)
    svc_poly_preds = svc_poly.predict(X_test)
    results['SVM with Polynomial Kernel'] = (accuracy_score(y_test, svc_poly_preds),
                                             recall_score(y_test, svc_poly_preds),
                                             precision_score(y_test, svc_poly_preds),
                                             f1_score(y_test, svc_poly_preds))

    # K-Nearest Neighbors (KNN) Classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    results['K-Nearest Neighbors (KNN) Classifier'] = (accuracy_score(y_test, knn_preds),
                                                      recall_score(y_test, knn_preds),
                                                      precision_score(y_test, knn_preds),
                                                      f1_score(y_test, knn_preds))

    # Gaussian Naive Bayes
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X_train, y_train)
    gaussian_nb_preds = gaussian_nb.predict(X_test)
    results['Gaussian Naive Bayes'] = (accuracy_score(y_test, gaussian_nb_preds),
                                       recall_score(y_test, gaussian_nb_preds),
                                       precision_score(y_test, gaussian_nb_preds),
                                       f1_score(y_test, gaussian_nb_preds))

    # Multinomial Naive Bayes
    multinomial_nb = MultinomialNB()
    multinomial_nb.fit(X_train, y_train)
    multinomial_nb_preds = multinomial_nb.predict(X_test)
    results['Multinomial Naive Bayes'] = (accuracy_score(y_test, multinomial_nb_preds),
                                          recall_score(y_test, multinomial_nb_preds),
                                          precision_score(y_test, multinomial_nb_preds),
                                          f1_score(y_test, multinomial_nb_preds))

    # Decision Tree Classifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    results['Decision Tree Classifier'] = (accuracy_score(y_test, dt_preds),
                                           recall_score(y_test, dt_preds),
                                           precision_score(y_test, dt_preds),
                                           f1_score(y_test, dt_preds))

    # AdaBoost Classifier
    adaboost = AdaBoostClassifier()
    adaboost.fit(X_train, y_train)
    adaboost_preds = adaboost.predict(X_test)
    results['AdaBoost Classifier'] = (accuracy_score(y_test, adaboost_preds),
                                      recall_score(y_test, adaboost_preds),
                                      precision_score(y_test, adaboost_preds),
                                      f1_score(y_test, adaboost_preds))
    
    # Gradient Boosting Classifier (GBM)
    gbm = GradientBoostingClassifier()
    gbm.fit(X_train, y_train)
    gbm_preds = gbm.predict(X_test)
    results['Gradient Boosting Classifier'] = (accuracy_score(y_test, gbm_preds), 
                                               recall_score(y_test, gbm_preds), 
                                               precision_score(y_test, gbm_preds), 
                                               f1_score(y_test, gbm_preds))

    # Stacking Classifier
    stacking = StackingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svc', SVC())],
                                  final_estimator=LogisticRegression())
    stacking.fit(X_train, y_train)
    stacking_preds = stacking.predict(X_test)
    results['Stacking Classifier'] = (accuracy_score(y_test, stacking_preds),
                                      recall_score(y_test, stacking_preds),
                                      precision_score(y_test, stacking_preds),
                                      f1_score(y_test, stacking_preds))

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy', 'Recall', 'Precision', 'F1 Score'])
    pd.options.display.float_format = '{:.3f}'.format

    # Sort the DataFrame by 'Accuracy' in descending order
    results_df = results_df.sort_values(by='Accuracy', ascending=False)

    print(results_df)

    if cf_matrix == True:
        # Plot confusion matrix for each classifier
        num_models = len(results)
        num_rows = 4  # Number of rows in the subplot grid
        num_cols = 3  # Number of columns in the subplot grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(11, 14))
        #plt.subplots_adjust(hspace=0.5)  # Increased spacing between subplots

        # Get the ordered model names
        ordered_model_names = results_df.index.tolist()

        for i, clf_name in enumerate(ordered_model_names):
            clf = None
            if clf_name == 'Logistic Regression':
                clf = LogisticRegression()
            elif clf_name == 'Bagging Classifier':
                clf = BaggingClassifier()
            elif clf_name == 'Random Forest Classifier':
                clf = RandomForestClassifier()
            elif clf_name == 'SVM with RBF Kernel':
                clf = SVC(kernel='rbf', probability=True)
            elif clf_name == 'SVM with Polynomial Kernel':
                clf = SVC(kernel='poly', probability=True)
            elif clf_name == 'K-Nearest Neighbors (KNN) Classifier':
                clf = KNeighborsClassifier()
            elif clf_name == 'Gaussian Naive Bayes':
                clf = GaussianNB()
            elif clf_name == 'Multinomial Naive Bayes':
                clf = MultinomialNB()
            elif clf_name == 'Decision Tree Classifier':
                clf = DecisionTreeClassifier()
            elif clf_name == 'AdaBoost Classifier':
                clf = AdaBoostClassifier()
            elif clf_name == 'Gradient Boosting Classifier':
                clf = GradientBoostingClassifier()
            elif clf_name == 'Stacking Classifier':
                clf = stacking  # Use the stacking classifier
            else:
                # Skip unknown classifiers
                continue

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            cm = confusion_matrix(y_test, preds)
            row, col = divmod(i, num_cols)
            ax = axes[row, col]
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False, ax=ax)
            ax.set_title(f'{clf_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Remove the empty subplot if the number of models is not a multiple of num_rows * num_cols
        if num_models % (num_rows * num_cols) != 0:
            fig.delaxes(axes.flat[-1])

        # Add a global title to the entire subplot grid
        plt.suptitle("\nConfusion Matrices\n", fontsize=16)

        plt.tight_layout(w_pad=8, h_pad=3)
        plt.show()
