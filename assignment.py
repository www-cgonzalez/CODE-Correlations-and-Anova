import pandas as pd
import statsmodels.api as sm
import statsmodels.graphics.api as smg
import statsmodels.formula.api as sm_api
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# If any of these libraries are missing from your computer,
# please install them using pip.

flights_filename = 'Flight_Delays_2018.csv'
flights_df = pd.read_csv(flights_filename)

coffee_filename = 'coffee.csv'
coffee_df = pd.read_csv(coffee_filename)

# Correlation matrix for the flights data, and adjusted measurements for better readability.
flights_corr = flights_df.corr(numeric_only=True).round(2)
plt.figure(figsize=(14, 11))
sns.heatmap(flights_corr, annot=True)
plt.title('Flight Delays 2018 correlation matrix')
plt.tight_layout()
plt.show()

# Correlation matrix for the coffee data.
coffee_corr = coffee_df.corr(numeric_only=True).round(2)
plt.figure(figsize=(14, 11))
sns.heatmap(coffee_corr, annot=True)
plt.title('Coffee correlation matrix')
plt.tight_layout()
plt.show()

# Mean and spread of total score for each country.
print(coffee_df.groupby('Location_Country')['Data_Scores_Total'].describe())

# Group 1: countries that should NOT be significantly different.
group1 = ['Brazil', 'China', 'Thailand']
coffee_group1 = coffee_df[coffee_df['Location_Country'].isin(group1)]
coffee_group1.boxplot(column='Data_Scores_Total', by='Location_Country')
plt.suptitle('')
plt.title('Total score by country (group 1)')
plt.ylabel('Data_Scores_Total')
plt.show()

# ANOVA for group 1
model1 = sm_api.ols('Data_Scores_Total ~ C(Location_Country)', data=coffee_group1).fit()
anova_table1 = sm.stats.anova_lm(model1, typ=2)
print(anova_table1)

# Group 2: countries that should be significantly different.
group2 = ['Ethiopia', 'Colombia', 'Mexico']
coffee_group2 = coffee_df[coffee_df['Location_Country'].isin(group2)]
coffee_group2.boxplot(column='Data_Scores_Total', by='Location_Country')
plt.suptitle('')
plt.title('Total score by country (group 2)')
plt.ylabel('Data_Scores_Total')
plt.show()

# ANOVA for group 2.
model2 = sm_api.ols('Data_Scores_Total ~ C(Location_Country)', data=coffee_group2).fit()
anova_table2 = sm.stats.anova_lm(model2, typ=2)
print(anova_table2)