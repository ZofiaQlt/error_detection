import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, normal_ad
from statsmodels.stats.stattools import durbin_watson
import itertools

def log_regression_analysis(data, target_variable, explanatory_vars, intercept=True, hypo=True):
    if intercept == False:
    
        bold = "\033[1m"
        italic = '\033[3m'
        red = '\033[91m'
        cyan = "\033[34m"
        end = "\033[0;0m"

        # Conversion automatique des variables booléennes en variables numériques (0 ou 1)
        for column in data.columns:
            if data[column].dtype == bool:
                data[column] = data[column].astype(int)

        def calculate_vif(data, variables):
            print("\n\n==========================================================================================================================================")
            print(bold + "VÉRIFICATION DU VIF (VARIANCE INFLATION FACTOR)" + end)
            print('===============================================\n')
            vif_data = pd.DataFrame()
            vif_data["Variable"] = variables
            vif_data["VIF"] = [variance_inflation_factor(data[variables].values, i) for i in range(len(variables))]
            print(vif_data)

            # Tri VIF par ordre décroissant
            vif_data = vif_data.sort_values(by="VIF", ascending=True)

            # Afficher les résultats VIF sous forme de graphique
            plt.figure(figsize=(10, 6))
            plt.barh(vif_data["Variable"], vif_data["VIF"])
            plt.xlabel("Variable explicative")
            plt.ylabel("VIF")
            plt.title("VIF pour les variables explicatives")
            plt.xticks(rotation=45)
            plt.show()

        def find_best_regression_model_with_elimination(data, target_variable, explanatory_vars, significance_level=0.05):
            best_model = None
            best_explanatory_vars = []
            best_pr2 = None

            for num_vars in range(1, len(explanatory_vars) + 1):
                for combo in itertools.combinations(explanatory_vars, num_vars):
                    formula = f"{target_variable} ~ {' + '.join(combo)} - 1"
                    model = smf.logit(formula, data=data).fit(disp=False)
                    p_values = model.pvalues
                    max_p_value = p_values.max()

                    if max_p_value > significance_level:
                        remove_var = p_values.idxmax()
                        combo = list(combo)
                        combo.remove(remove_var)
                        #print(f"Removed variable '{remove_var}' with p-value: {max_p_value:.4f}, Remaining variables: {', '.join(combo)}")
                    else:
                        if best_pr2 is None or model.prsquared > best_pr2:
                            best_pr2 = model.prsquared
                            best_model = model
                            best_explanatory_vars = list(combo)

            return best_model, best_explanatory_vars

        def visualize_regression_results(model_stats, explanatory_vars):
            num_vars = len(explanatory_vars)
            num_rows = (num_vars + 2) // 3
            num_cols = min(num_vars, 3)

            print("\n==========================================================================================================================================")
            print(bold + 'VÉRIFICATION DE LA LINEARITÉ' + end)
            print('============================\n')
            fig, a = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
            a = a.flatten()

            for i, col in enumerate(explanatory_vars):
                sns.regplot(data=data, x=col, y=target_variable, lowess=True, line_kws={'color': 'black'}, ax=a[i])
                a[i].set_title(f'Regression plot: {col} vs {target_variable}')
                a[i].set_xlabel(col)
                a[i].set_ylabel(target_variable)

            plt.tight_layout()

            # Supprime automatiquement les derniers sous-graphiques vides
            for i in range(num_vars, num_rows * num_cols):
                fig.delaxes(a[i])

            plt.show()

        # Perform regression analysis
        print("Backward selection\n==================")
        best_model_with_elimination, best_explanatory_vars = find_best_regression_model_with_elimination(data, target_variable, explanatory_vars)

        # Check if all variables are significant
        if len(best_explanatory_vars) == len(explanatory_vars):
            print("Toutes les variables sont significatives.")
        else:
            print(f"Certaines variables ne sont pas significatives, nous les supprimons et gardons : {', '.join(best_explanatory_vars)}.")

        # Display summary of the best model with elimination
        print('\n')
        print(best_model_with_elimination.summary())

        # Calculate VIF for the best model without intercept
        calculate_vif(data, best_explanatory_vars)

        # Calculate VIF for the best model without intercept
        if hypo == True:
            visualize_regression_results(best_model_with_elimination, best_explanatory_vars)
           
    else:
        bold = "\033[1m"
        italic = '\033[3m'
        red = '\033[91m'
        cyan = "\033[34m"
        end = "\033[0;0m"

        # Conversion automatique des variables booléennes en variables numériques (0 ou 1)
        for column in data.columns:
            if data[column].dtype == bool:
                data[column] = data[column].astype(int)

        def calculate_vif(data, variables):
            print("\n\n==========================================================================================================================================")
            print(bold + "VÉRIFICATION DU VIF (VARIANCE INFLATION FACTOR)" + end)
            print('===============================================\n')
            vif_data = pd.DataFrame()
            vif_data["Variable"] = variables
            vif_data["VIF"] = [variance_inflation_factor(data[variables].values, i) for i in range(len(variables))]
            print(vif_data)

            # Tri VIF par ordre décroissant
            vif_data = vif_data.sort_values(by="VIF", ascending=True)

            # Afficher les résultats VIF sous forme de graphique
            plt.figure(figsize=(10, 6))
            plt.barh(vif_data["Variable"], vif_data["VIF"])
            plt.xlabel("Variable explicative")
            plt.ylabel("VIF")
            plt.title("VIF pour les variables explicatives")
            plt.xticks(rotation=45)
            plt.show()

        def find_best_regression_model_with_elimination(data, target_variable, explanatory_vars, significance_level=0.05):
            best_model = None
            best_explanatory_vars = []
            best_pr2 = None

            for num_vars in range(1, len(explanatory_vars) + 1):
                for combo in itertools.combinations(explanatory_vars, num_vars):
                    formula = f"{target_variable} ~ {' + '.join(combo)}"
                    model = smf.logit(formula, data=data).fit(disp=False)
                    p_values = model.pvalues
                    max_p_value = p_values.max()

                    if max_p_value > significance_level:
                        remove_var = p_values.idxmax()
                        combo = list(combo)
                        if remove_var in combo:
                            combo.remove(remove_var)
                        else:
                            pass
                                                
                        #print(f"Removed variable '{remove_var}' with p-value: {max_p_value:.4f}, Remaining variables: {', '.join(combo)}")
                    else:
                        if best_pr2 is None or model.prsquared > best_pr2:
                            best_pr2 = model.prsquared
                            best_model = model
                            best_explanatory_vars = list(combo)

            return best_model, best_explanatory_vars

        def visualize_regression_results(model_stats, explanatory_vars):
            num_vars = len(explanatory_vars)
            num_rows = (num_vars + 2) // 3
            num_cols = min(num_vars, 3)

            print("\n==========================================================================================================================================")
            print(bold + 'VÉRIFICATION DE LA LINEARITÉ' + end)
            print('============================\n')
            fig, a = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
            a = a.flatten()

            for i, col in enumerate(explanatory_vars):
                sns.regplot(data=data, x=col, y=target_variable, lowess=True, line_kws={'color': 'black'}, ax=a[i])
                a[i].set_title(f'Regression plot: {col} vs {target_variable}')
                a[i].set_xlabel(col)
                a[i].set_ylabel(target_variable)

            plt.tight_layout()

            # Supprime automatiquement les derniers sous-graphiques vides
            for i in range(num_vars, num_rows * num_cols):
                fig.delaxes(a[i])

            plt.show()

            

        # Perform regression analysis
        print("Backward selection\n==================")
        best_model_with_elimination, best_explanatory_vars = find_best_regression_model_with_elimination(data, target_variable, explanatory_vars)

        # Check if all variables are significant
        if len(best_explanatory_vars) == len(explanatory_vars):
            print("Toutes les variables sont significatives.")
        else:
            print(f"Certaines variables ne sont pas significatives, nous les supprimons et gardons : {', '.join(best_explanatory_vars)}.")

        # Display summary of the best model with elimination
        print('\n')
        print(best_model_with_elimination.summary())

        # Calculate VIF for the best model without intercept
        calculate_vif(data, best_explanatory_vars)

        # Calculate VIF for the best model without intercept
        if hypo == True:
            visualize_regression_results(best_model_with_elimination, best_explanatory_vars)
           
