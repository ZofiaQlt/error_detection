import numpy as np
from scipy import stats

def calculate_normality(data_frame, alpha=0.05):
    
    bold = "\033[1m"
    red = '\033[91m'
    cyan = "\033[34m"
    end = "\033[0;0m"
    
    for i in data_frame.columns:
        print("--" * 25 + bold + "\n" + i + " :" + end)
        
        # Check for NaN values in the column
        if data_frame[i].isnull().any():
            print("Skew: nan\nKurtosis: nan\n---\nTest statistiques :\nD'Agostino-Pearson: nan\nShapiro-Wilk: nan")
        else:
            # Calculate skewness and kurtosis
            kurtosis = round(stats.kurtosis(data_frame[i]), ndigits=2)
            skew = round(stats.skew(data_frame[i]), ndigits=2)

            # Interpreting Skewness
            if -0.5 < skew < 0.5:
                skewness_result = (cyan + f'The distribution is approximately symmetric' + end)
            elif -1 < skew < -0.5 or 0.5 < skew < 1.0:
                skewness_result = (red + f'The distribution is moderately skewed' + end)
            else:
                skewness_result = (red + 'The distribution is highly skewed' + end)

            # Interpreting Kurtosis
            if -0.5 < kurtosis < 0.5:
                kurtosis_result = (cyan + f'The distribution is approximately normal, sometimes called mesokurtic distributions' + end)
            elif kurtosis <= -0.5:
                kurtosis_result = (red + f'The distribution is light-tailed (negative), sometimes called platykurtic distributions' + end)
            elif kurtosis >= 0.5:
                kurtosis_result = (red + f'The distribution is heavy-tailed (positive), sometimes called leptokurtic distribution' + end)

            # Print skewness and kurtosis
            print(f"Skew: {skew}   {skewness_result}")
            print(f"Kurtosis: {kurtosis}   {kurtosis_result} \n---")

            # Test normality using D'Agostino-Pearson
            stat_dagostino, p_dagostino = stats.normaltest(data_frame[i])
            if p_dagostino > alpha:
                dagostino_result = (cyan + 'Sample looks Gaussian (fail to reject H0)' + end)
            else:
                dagostino_result = (red + 'Sample does not look Gaussian (reject H0)' + end)

            # Test normality using Shapiro-Wilk
            stat_shapiro, p_shapiro = stats.shapiro(data_frame[i])
            if p_shapiro > alpha:
                shapiro_result = (cyan + 'Sample looks Gaussian (fail to reject H0)' + end)
            else:
                shapiro_result = (red + 'Sample does not look Gaussian (reject H0)' + end)

            # Print normality test results
            print("Tests statistiques :\nD'Agostino-Pearson:" + f" Statistic={stat_dagostino:.4f}, p={p_dagostino:.4f} - {dagostino_result}")
            print("Shapiro-Wilk:" + f" Statistic={stat_shapiro:.4f}, p={p_shapiro:.4f} - {shapiro_result}")
    
    print('--' * 25)
