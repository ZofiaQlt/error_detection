import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, normal_ad
from statsmodels.stats.stattools import durbin_watson
import itertools

def regression_analysis(data, target_variable, explanatory_vars, intercept=True, hypo=True):
    if intercept == False:
    
        bold = "\033[1m"
        red = '\033[91m'
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
            best_r2 = None

            for num_vars in range(1, len(explanatory_vars) + 1):
                for combo in itertools.combinations(explanatory_vars, num_vars):
                    formula = f"{target_variable} ~ {' + '.join(combo)} - 1"
                    model = smf.ols(formula, data=data).fit()
                    p_values = model.pvalues
                    max_p_value = p_values.max()

                    if max_p_value > significance_level:
                        remove_var = p_values.idxmax()
                        combo = list(combo)
                        combo.remove(remove_var)
                        #print(f"Removed variable '{remove_var}' with p-value: {max_p_value:.4f}, Remaining variables: {', '.join(combo)}")
                    else:
                        if best_r2 is None or model.rsquared > best_r2:
                            best_r2 = model.rsquared
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

            print("\n==========================================================================================================================================")
            print(bold + "VÉRIFICATION DE L'HOMOSCÉDASTICITÉ" + end)
            print('==================================\n')
            print(bold + red + "Scatterplot des résidus\n" + end)
            plt.figure()
            residuals = model_stats.resid
            sns.scatterplot(x=model_stats.predict(), y=residuals)
            #sns.regplot(x=model_stats.predict(), y=residuals, lowess=True, line_kws={'color': 'black'})
            plt.title('Residuals vs Fitted')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            plt.show()

            # Calcul du test de Breusch-Pagan
            print(bold + red + "\nTest d'hétéroscédasticité de Breusch-Pagan\n" + end)
            bp_test = het_breuschpagan(residuals, model_stats.model.exog)
            print("Statistique de test de Breusch-Pagan :", bp_test[0])
            print("P-value du test de Breusch-Pagan :", bp_test[1])
            if bp_test[1] > 0.05:
                print("=> Présence d'homoscédasticité dans les résidus (p_value > 0.05, we fail to reject H0)\n")
            else:
                print("=> Présence d'hétéroscédasticité dans les résidus (p_value < 0.05, we reject H0)\n")
                print("=> Cela suggère que les résidus ne présentent pas une variance constante et que la variance des résidus\ndépend probablement des valeurs prédites du modèle.")

            print("\n==========================================================================================================================================")
            print(bold + "VÉRIFICATION DE LA NORMALITÉ DES RÉSIDUS" + end)
            print('========================================\n')
            print(bold + red + "Histogramme et graphique QQ\n" + end)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            sns.histplot(residuals, kde=True, ax=axes[0])
            axes[0].set_title('Histogramme des Résidus')
            axes[0].set_xlabel('Résidus')
            median = residuals.median()
            axes[0].axvline(median, color='red', linestyle='dashed', linewidth=1, label='Médiane')
            mean = residuals.mean()
            axes[0].axvline(mean, color='green', linestyle='dashed', linewidth=1, label='Moyenne')
            axes[0].legend()

            skewness = residuals.skew()
            kurtosis = residuals.kurtosis()
            _, p_shapiro = shapiro(residuals)
            info_text = f'Skew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}\nShapiro p-value: {p_shapiro:.3f}'
            axes[0].text(0.63, 0.65, info_text, transform=axes[0].transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            qq_plot = sm.qqplot(residuals, line='s', ax=axes[1])
            axes[1].set_title('QQ Plot des Résidus')
            axes[1].set_xlabel("Quantiles théoriques")
            axes[1].set_ylabel("Quantiles des résidus")
            axes[1].text(0.1, 0.9, "Si les points suivent", transform=axes[1].transAxes)
            axes[1].text(0.1, 0.85, "approximativement la", transform=axes[1].transAxes)
            axes[1].text(0.1, 0.8, "ligne rouge, les résidus", transform=axes[1].transAxes)
            axes[1].text(0.1, 0.75, "sont normalement distribués.", transform=axes[1].transAxes)

            plt.tight_layout()
            plt.show()

            print(bold + red + "\nTest de normalité Shapiro-Wilk\n" + end)
            print(shapiro(model_stats.resid))
            if p_shapiro > 0.05:
                print('=> Sample looks Gaussian (p_value > 0.05, we fail to reject H0)\n')
            else:
                print('=> Sample does not look Gaussian (p_value < 0.05, we reject H0)\n')

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
            print(bold + red + "Moyenne et écart-type des résidus\n" + end)
            print("La moyenne des résidus est de", best_model_with_elimination.resid.mean())
            print("L'écart-type des résidus est de", best_model_with_elimination.resid.std())
            print("\n==========================================================================================================================================")
            print(bold + "VÉRIFICATION DU L'AUTOCORRÉLATION DES RÉSIDUS" + end)
            print('=============================================\n')
            #print(bold + red + "\n\nVérification de l'autocorrélation des résidus\n" + end)
            
            # Calculer le test de Durbin-Watson
            def durbin_watson_test(residuals):
                durbin_watson_statistic = durbin_watson(residuals)
                return durbin_watson_statistic
    
            # Appelez la fonction avec vos résidus
            dw_statistic = durbin_watson_test(best_model_with_elimination.resid)
            
            # Afficher la statistique Durbin-Watson
            print(f"Statistique Durbin-Watson : {dw_statistic}")

            # Interprétation du test de Durbin-Watson
            if dw_statistic < 1.5:
                print("=> Durbin-Watson < 1.5 : Autocorrélation positive des résidus  (l'hypothèse n'est pas validée)")
            elif dw_statistic < 2.5:
                print("=> 1.5 < Durbin-Watson < 2.5 : Pas d'autocorrélation des résidus (l'hypothèse est validée)")
            else:
                print("=> Durbin-Watson > 2.5 : Autocorrélation négative des résidus (l'hypothèse n'est pas validée)")

         # Graphiques de régression partielle
            #fig, ax = plt.subplots(1, len(best_explanatory_vars), figsize=(15, 4))
            #for i, var in enumerate(best_explanatory_vars):
                #exog_vars = [v for v in best_explanatory_vars if v != var]  # Exclure la variable actuelle
                #sm.graphics.plot_partregress(target_variable, var, exog_others=exog_vars, data=data, ax=ax[i], obs_labels=False)
                #ax[i].set_title(f'Régression partielle de {var}')
            #plt.tight_layout()
            #plt.show()

            # Générer le graphique d'influence
            #sm.graphics.influence_plot(best_model_with_elimination)
            #sm.graphics.plot_leverage_resid2(best_model_with_elimination)

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
            best_r2 = None

            for num_vars in range(1, len(explanatory_vars) + 1):
                for combo in itertools.combinations(explanatory_vars, num_vars):
                    formula = f"{target_variable} ~ {' + '.join(combo)}"
                    model = smf.ols(formula, data=data).fit()
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
                        if best_r2 is None or model.rsquared > best_r2:
                            best_r2 = model.rsquared
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

            print("\n==========================================================================================================================================")
            print(bold + "VÉRIFICATION DE L'HOMOSCÉDASTICITÉ" + end)
            print('==================================\n')
            print(bold + red + "Scatterplot des résidus\n" + end)
            plt.figure()
            residuals = model_stats.resid
            sns.scatterplot(x=model_stats.predict(), y=residuals)
            #sns.regplot(x=model_stats.predict(), y=residuals, lowess=True, line_kws={'color': 'black'})
            plt.title('Residuals vs Fitted')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            plt.show()

            # Calcul du test de Breusch-Pagan
            print(bold + red + "\nTest d'hétéroscédasticité de Breusch-Pagan\n" + end)
            bp_test = het_breuschpagan(residuals, model_stats.model.exog)
            print("Statistique de test de Breusch-Pagan :", bp_test[0])
            print("P-value du test de Breusch-Pagan :", bp_test[1])
            if bp_test[1] > 0.05:
                print("=> Présence d'homoscédasticité dans les résidus (p_value > 0.05, non rejet de H0)\n")
            else:
                print("=> Présence d'hétéroscédasticité dans les résidus (p_value < 0.05, on rejette H0)\n")
                print("=> Cela suggère que les résidus ne présentent pas une variance constante et que la variance des résidus\ndépend probablement des valeurs prédites du modèle.")
                
            # Test de White
            print(bold + red + "\nTest d'hétéroscédasticité de White\n" + end)
            w_test = het_white(residuals, model_stats.model.exog)
            print("Statistique de test de White :", w_test[0])
            print("P-value du test de White :", w_test[1])
            if w_test[1] > 0.05:
                print("=> Pas d'hétéroscédasticité (homoscédasticité avec p_value > 0.05, non rejet de H0)")
            else:
                print("=> Hétéroscédasticité détectée (p_value < 0.05, on rejette H0)")

            print("\n==========================================================================================================================================")
            print(bold + "VÉRIFICATION DE LA NORMALITÉ DES RÉSIDUS" + end)
            print('========================================\n')
            print(bold + red + "Histogramme et graphique QQ\n" + end)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            sns.histplot(residuals, kde=True, ax=axes[0])
            axes[0].set_title('Histogramme des Résidus')
            axes[0].set_xlabel('Résidus')
            median = residuals.median()
            axes[0].axvline(median, color='red', linestyle='dashed', linewidth=1, label='Médiane')
            mean = residuals.mean()
            axes[0].axvline(mean, color='green', linestyle='dashed', linewidth=1, label='Moyenne')
            axes[0].legend()

            skewness = residuals.skew()
            kurtosis = residuals.kurtosis()
            _, p_shapiro = shapiro(residuals)
            info_text = f'Skew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}\nShapiro p-value: {p_shapiro:.3f}'
            axes[0].text(0.63, 0.65, info_text, transform=axes[0].transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            qq_plot = sm.qqplot(residuals, line='s', ax=axes[1])
            axes[1].set_title('QQ Plot des Résidus')
            axes[1].set_xlabel("Quantiles théoriques")
            axes[1].set_ylabel("Quantiles des résidus")
            axes[1].text(0.1, 0.9, "Si les points suivent", transform=axes[1].transAxes)
            axes[1].text(0.1, 0.85, "approximativement la", transform=axes[1].transAxes)
            axes[1].text(0.1, 0.8, "ligne rouge, les résidus", transform=axes[1].transAxes)
            axes[1].text(0.1, 0.75, "sont normalement distribués.", transform=axes[1].transAxes)

            plt.tight_layout()
            plt.show()

            print(bold + red + "\nTest de normalité Shapiro-Wilk\n" + end)
            print(shapiro(model_stats.resid))
            if p_shapiro > 0.05:
                print('=> Sample looks Gaussian (p_value > 0.05, we fail to reject H0)\n')
            else:
                print('=> Sample does not look Gaussian (p_value < 0.05, we reject H0)\n')

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
            print(bold + red + "Moyenne et écart-type des résidus\n" + end)
            print("La moyenne des résidus est de", best_model_with_elimination.resid.mean())
            print("L'écart-type des résidus est de", best_model_with_elimination.resid.std())
            print("\n==========================================================================================================================================")
            print(bold + "VÉRIFICATION DU L'AUTOCORRÉLATION DES RÉSIDUS" + end)
            print('=============================================\n')
            #print(bold + red + "Vérification de l'autocorrélation des résidus\n" + end)
            
            # Calculer le test de Durbin-Watson
            def durbin_watson_test(residuals):
                durbin_watson_statistic = durbin_watson(residuals)
                return durbin_watson_statistic
    
            # Appelez la fonction avec vos résidus
            dw_statistic = durbin_watson_test(best_model_with_elimination.resid)
            
            # Afficher la statistique Durbin-Watson
            print(f"Statistique Durbin-Watson : {dw_statistic}")

            # Interprétation du test de Durbin-Watson
            if dw_statistic < 1.5:
                print("=> Durbin-Watson < 1.5 : Autocorrélation positive des résidus (l'hypothèse n'est pas validée)")
            elif dw_statistic < 2.5:
                print("=> 1.5 < Durbin-Watson < 2.5 : Pas d'autocorrélation des résidus (l'hypothèse est validée)")
            else:
                print("=> Durbin-Watson > 2.5 : Autocorrélation négative des résidus (l'hypothèse n'est pas validée)")

            

            # Graphiques de régression partielle
            #fig, ax = plt.subplots(1, len(best_explanatory_vars), figsize=(15, 4))
            #for i, var in enumerate(best_explanatory_vars):
                #exog_vars = [v for v in best_explanatory_vars if v != var]  # Exclure la variable actuelle
                #sm.graphics.plot_partregress(target_variable, var, exog_others=exog_vars, data=data, ax=ax[i], obs_labels=False)
                #ax[i].set_title(f'Régression partielle de {var}')
            #plt.tight_layout()
            #plt.show()
            
            # Générer le graphique d'influence
            #sm.graphics.influence_plot(best_model_with_elimination)
            #sm.graphics.plot_leverage_resid2(best_model_with_elimination)
            
