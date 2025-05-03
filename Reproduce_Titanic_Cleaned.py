# Titanic Analysis - Full Python Translation with Visualizations
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.graphics.mosaicplot import mosaic
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full = pd.concat([train, test], sort=False).reset_index(drop=True)

# Visualize missingness like md.pattern()
plt.figure(figsize=(10, 6))
msno.matrix(full)
plt.title("Missing Data Pattern (like md.pattern in R)")
plt.show()

# Feature Engineering
full['Title'] = full['Name'].apply(lambda name: re.search(r',\s*([^\.]+)\.', name).group(1))
full['Title'] = full['Title'].replace(['Dona', 'Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare Title')
full['Title'] = full['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
full['Surname'] = full['Name'].apply(lambda name: name.split(',')[0])
full['Fsize'] = full['SibSp'] + full['Parch'] + 1
full['Family'] = full['Surname'] + '_' + full['Fsize'].astype(str)
full['FsizeD'] = full['Fsize'].apply(lambda x: 'singleton' if x == 1 else 'small' if x < 5 else 'large')

# Deck
full['Deck'] = full['Cabin'].apply(lambda x: x[0] if pd.notna(x) and x != '' else 'Unknown')

# Impute Embarked
full.loc[full['Embarked'].isna(), 'Embarked'] = 'C'

# Impute Fare
fare_median = full[(full['Pclass'] == 3) & (full['Embarked'] == 'S')]['Fare'].median()
full.loc[full['Fare'].isna(), 'Fare'] = fare_median

# Age Imputation
factor_vars = ['PassengerId','Pclass','Sex','Embarked','Title','Surname','FsizeD']
for var in factor_vars:
    full[var] = full[var].astype('category')

impute_features = full.drop(columns=['PassengerId','Name','Ticket','Cabin','Family','Surname','Survived'])
impute_features_encoded = pd.get_dummies(impute_features)
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=129),
                           max_iter=10, random_state=129)
imputed_array = imputer.fit_transform(impute_features_encoded)
full['Age'] = imputed_array[:, list(impute_features_encoded.columns).index('Age')]

# Child and Mother features
full['Child'] = full['Age'].apply(lambda age: 'Child' if age < 18 else 'Adult')
full['Mother'] = 'Not Mother'
full.loc[(full['Sex'] == 'female') & (full['Parch'] > 0) & (full['Age'] > 18) & (full['Title'] != 'Miss'), 'Mother'] = 'Mother'

# Split data
train_df = full.iloc[:891].copy()
test_df = full.iloc[891:].copy()
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FsizeD', 'Child', 'Mother']
X_train = pd.get_dummies(train_df[features])
y_train = train_df['Survived'].astype(int)
X_test = pd.get_dummies(test_df[features])
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=754)
rf_model.fit(X_train, y_train)

# --- Visualizations ---
train_df = train_df[train_df['Survived'].notnull()].copy()
train_df['Survived'] = train_df['Survived'].astype(int).astype(str)
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Fsize', hue='Survived', palette='pastel')
plt.title('Family Size vs Survival')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.show()

mosaic(train_df, ['FsizeD', 'Survived'])
plt.title('Family Size Category vs Survival')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=full[full['PassengerId'] != 62], x='Embarked', y='Fare', hue='Pclass')
plt.axhline(80, color='red', linestyle='--')
plt.title('Fare vs Embarked and Pclass')
plt.show()

fare_data = full[(full['Pclass'] == 3) & (full['Embarked'] == 'S') & (full['Fare'].notnull()) & (np.isfinite(full['Fare']))]
plt.figure(figsize=(8, 6))
plt.hist(fare_data['Fare'].astype(float), bins=30, color='#99d6ff', alpha=0.6, density=True)
plt.axvline(fare_median, color='red', linestyle='--', label='Median Fare')
plt.title('Fare Distribution (Pclass 3, Embarked S)')
plt.xlabel('Fare')
plt.ylabel('Density')
plt.legend()
plt.show()

# Age histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.histplot(full['Age'], bins=30, kde=False, color='darkgreen', ax=axes[0])
axes[0].set_title('Age: Original Data')
sns.histplot(full['Age'], bins=30, kde=False, color='lightgreen', ax=axes[1])
axes[1].set_title('Age: After MICE Imputation')
plt.tight_layout()
plt.show()

# Age vs Survival by Sex
g = sns.FacetGrid(train_df, col='Sex', hue='Survived', height=5)
g.map(sns.histplot, 'Age', kde=False, bins=20, element='step', stat='count')
g.add_legend()
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Age Distribution by Sex and Survival')
plt.show()

# Child and Mother Crosstabs
print("Child vs Survival:\n", pd.crosstab(full['Child'], full['Survived']), "\n")
print("Mother vs Survival:\n", pd.crosstab(full['Mother'], full['Survived']), "\n")

# Variable Importance
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Variables': X_train.columns, 'Importance': np.round(importances, 4)})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df['Rank'] = ['#' + str(i + 1) for i in range(len(importance_df))]

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='Importance', y='Variables', palette='viridis')
for i in range(len(importance_df)):
    plt.text(x=0.005, y=i, s=importance_df['Rank'].values[i], color='red', va='center')
plt.title('Variable Importance (Random Forest)')
plt.tight_layout()
plt.show()

# OOB Score
print(f"OOB Score: {rf_model.oob_score_:.4f}")

# Simulated Error Curve
oob_errors = []
n_estimators_range = range(10, 301, 10)
for n in n_estimators_range:
    model_temp = RandomForestClassifier(
        n_estimators=n, 
        oob_score=True, 
        random_state=754,
        n_jobs=-1,
        bootstrap=True
    )
    model_temp.fit(X_train, y_train)
    oob_errors.append(1 - model_temp.oob_score_)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, oob_errors, marker='o')
plt.title('OOB Error Rate vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error Rate')
plt.grid(True)
plt.show()

# Predictions and Output
predictions = rf_model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions.astype(int)})
submission.to_csv("rf_mod_Solution_Py.csv", index=False)
print(submission.head())
