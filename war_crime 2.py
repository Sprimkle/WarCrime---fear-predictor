import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import xlrd

path = "C:/Users/44731/OneDrive/Desktop/Homework 2/"

df_terror = pd.read_csv(path + "share-who-are-worried-about-vs-share-of-deaths-from-terrorism.csv")
counts = df_terror['Year'].value_counts()

df_gdp = pd.read_csv(path + "gdp.csv",skiprows=3)
df_gdp.drop(columns=['Indicator Name','Indicator Code','Country Name','Unnamed: 67'],inplace=True)

df = df_terror.merge(df_gdp,left_on='Code',right_on='Country Code')
df.drop(columns=['Country Code'],inplace=True) #Get rid of this as we already have 'Code'. Same thing.


# Function to get GDP based on 'Year'
def get_gdp(row):
    year_col = str(row['Year'])
    return row[year_col]

# Apply the function to create the 'GDP' column
df['GDP'] = df.apply(get_gdp, axis=1)


# Get numerical column names and drop them, now that we have the relevant years extracted
numerical_columns = df.columns[df.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]
df = df.drop(columns=numerical_columns) # Drop numerical columns (We don't need all the years!)


#I noticed that in the terrorism dataset, the 'Continent' column is not completely mapped - some values are missing. Let's fix this
df_map = df[['Entity','Continent']].dropna() #Create a map of Country and its corresponding continent
df.drop(columns=['Continent'],inplace=True) #Drop the original incomplete column
df = pd.merge(df, df_map,on='Entity') #Replace the 'Continent' column with the complete version from df_map

df = df.rename({'Great deal or Very much':'Fear'},axis=1)







#Obtain our cleaned dataset, ready for analysis
df_clean = df.dropna(subset=['Fear'])

#Drop rows with missing terrorism data or GDP data
df_clean.dropna(inplace=True)



#Sort data in format compatible with machine learning
ml_df = df_clean.drop(columns=['Entity','Code','Year'])
ml_df = ml_df[['Fear','Share of deaths from terrorism','Terrorist attacks','GDP','Continent']]



from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler

# Extract numerical columns
numerical_list = ['Share of deaths from terrorism', 'Terrorist attacks', 'GDP']
#==============================================================================
#Minmax scaling all numerical columns so that they are between 0 and 1
df_scaler = MinMaxScaler()

numerical_variables_list = []
for col in numerical_list:
    if col in ml_df.columns:
        numerical_variables_list.append(col)

columns_to_preprocess = numerical_variables_list
x = ml_df[numerical_variables_list].values
x_scaled = df_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=columns_to_preprocess, index = ml_df.index)
ml_df[columns_to_preprocess] = df_temp

ml_df = pd.get_dummies(ml_df)

#==============================================================================









'''
Part 1: Latent dirichlet allocation as a probabilistic modelling formulation of topic models
'''

# Apply Latent Dirichlet Allocation
num_topics = 3  # You can adjust the number of topics based on your needs
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
topics = lda.fit_transform(ml_df)


#Seeing what the main features of each topic are
feature_names = ml_df.columns
for i, topic in enumerate(lda.components_):
    print(f"Top features for Topic {i + 1}:")
    top_features_indices = topic.argsort()[-3:][::-1]  # Adjust the number of top features as needed
    top_features = [feature_names[idx] for idx in top_features_indices]
    print(top_features)



#Plotting Correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt
# Calculate the correlation matrix
correlation_matrix = ml_df.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()











'''
Bayesian Models of Parameter Estimation
'''
import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #Sort data in format compatible with machine learning
# ml_df = df_clean.drop(columns=['Entity','Code','Year'])
# ml_df = ml_df[['Fear','Share of deaths from terrorism','Terrorist attacks','GDP','Continent']]
# ml_df.dropna(inplace=True)
# ml_df = pd.get_dummies(ml_df)
# ml_df.dtypes
ml_df_col_list = list(ml_df.columns)
items_to_remove = ['Fear', 'Share of deaths from terrorism', 'Terrorist attacks', 'GDP']
ml_df_col_list = [item for item in ml_df_col_list if item not in items_to_remove]
ml_df[ml_df_col_list] = ml_df[ml_df_col_list].astype('uint8')
ml_df.dtypes
# Define Bayesian linear regression model
with pm.Model() as linear_model:
    # Prior distributions for coefficients
    intercept = pm.Normal('intercept', mu=0, sd=10)
    coefficients = pm.Normal('coefficients', mu=0, sd=10, shape=len(ml_df.columns) - 1)
    
    # Linear regression model
    mu = intercept + pm.math.dot(ml_df.iloc[:, 1:], coefficients)
    
    # Likelihood (normal distribution)
    likelihood = pm.Normal('likelihood', mu=mu, sd=1, observed=ml_df['Fear'].values)

# Sample from the posterior distribution
with linear_model:
    trace = pm.sample(2000, tune=1000, cores=-1)  # You can adjust the number of samples and tuning steps

# Plot posterior distributions
pm.plot_posterior(trace, var_names=['intercept'])
plt.show()


summary = pm.summary(trace, var_names=['coefficients'])




# Extract variable names from ml_df
variable_names = ml_df.columns[1:]  # Exclude the dependent variable 'Fear'

# Rename coefficients in the summary DataFrame
summary.rename(index={f'coefficients[{i}]': variable_names[i] for i in range(len(variable_names))}, inplace=True)

#Interpretation: For every standard deviation increase, the 'Fear' factor will change by the mean
#Problem: Numerical columns were previously transformed using Min Max Scaler. Let's inverse transform them back to the raw values



#Inverse transforming
to_inverse_transform = pd.DataFrame(summary[summary.index.isin(numerical_variables_list)]['mean']).T

# Inverse transform the coefficients for the numerical features
inv_transformed_summary = df_scaler.inverse_transform(to_inverse_transform)
inv_transformed_summary = pd.DataFrame(inv_transformed_summary, columns=numerical_variables_list)

#Mapping it back
inv_transformed_summary = inv_transformed_summary.rename(index={0: 'sd'})
T_summary = summary.T

T_summary.update(inv_transformed_summary)
summary_final = T_summary.T #rotate the corrected dataframe back to normal
summary_final['mean_adjusted'] = summary_final['mean'] / summary_final['sd']
