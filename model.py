import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_combat = pd.read_csv('combats.csv')
df_test = pd.read_csv('tests.csv')
df_pokemon = pd.read_csv('pokemon.csv')

# print(df_combat.head())
# print(df_combat.shape)
# print(df_test.head())
# print(df_test.shape)
# print(df_pokemon.head())
# print(df_pokemon.shape)

# 1st pokemon dataframe
df_combat['1st_pokemon_name'] = df_combat['First_pokemon'].apply(lambda x:df_pokemon['Name'][x-1])
df_combat['1st_pokemon_type1'] = df_combat['First_pokemon'].apply(lambda x:df_pokemon['Type 1'][x-1])
df_combat['1st_pokemon_hp'] = df_combat['First_pokemon'].apply(lambda x:df_pokemon['HP'][x-1])
df_combat['1st_pokemon_attack'] = df_combat['First_pokemon'].apply(lambda x:df_pokemon['Attack'][x-1])
df_combat['1st_pokemon_defense'] = df_combat['First_pokemon'].apply(lambda x:df_pokemon['Defense'][x-1])
df_combat['1st_pokemon_spatk'] = df_combat['First_pokemon'].apply(lambda x:df_pokemon['Sp. Atk'][x-1])
df_combat['1st_pokemon_spdef'] = df_combat['First_pokemon'].apply(lambda x:df_pokemon['Sp. Def'][x-1])
df_combat['1st_pokemon_speed'] = df_combat['First_pokemon'].apply(lambda x:df_pokemon['Speed'][x-1])

# 2nd pokemon dataframe
df_combat['2nd_pokemon_name'] = df_combat['Second_pokemon'].apply(lambda x:df_pokemon['Name'][x-1])
df_combat['2nd_pokemon_type1'] = df_combat['Second_pokemon'].apply(lambda x:df_pokemon['Type 1'][x-1])
df_combat['2nd_pokemon_hp'] = df_combat['Second_pokemon'].apply(lambda x:df_pokemon['HP'][x-1])
df_combat['2nd_pokemon_attack'] = df_combat['Second_pokemon'].apply(lambda x:df_pokemon['Attack'][x-1])
df_combat['2nd_pokemon_defense'] = df_combat['Second_pokemon'].apply(lambda x:df_pokemon['Defense'][x-1])
df_combat['2nd_pokemon_spatk'] = df_combat['Second_pokemon'].apply(lambda x:df_pokemon['Sp. Atk'][x-1])
df_combat['2nd_pokemon_spdef'] = df_combat['Second_pokemon'].apply(lambda x:df_pokemon['Sp. Def'][x-1])
df_combat['2nd_pokemon_speed'] = df_combat['Second_pokemon'].apply(lambda x:df_pokemon['Speed'][x-1])

# print(df_combat.head())
# print(df_combat.isnull().sum())

# create target
df_combat['First_win'] = df_combat.apply(lambda col: 1 if col['Winner'] == col['First_pokemon'] else 0, axis=1)

# label pokemon type1
from sklearn.preprocessing import LabelEncoder

label_1st_type1 = LabelEncoder()
label_2nd_type1 = LabelEncoder()
label_type1_bio = LabelEncoder()

label_1st_type1.fit(df_combat['1st_pokemon_type1'])
label_2nd_type1.fit(df_combat['2nd_pokemon_type1'])
label_type1_bio.fit(df_pokemon['Type 1'])


df_combat['label_1st_type1'] = label_1st_type1.transform(df_combat['1st_pokemon_type1'])
df_combat['label_2nd_type1'] = label_2nd_type1.transform(df_combat['2nd_pokemon_type1'])
df_pokemon['Label Type 1'] = label_type1_bio.transform(df_pokemon['Type 1'])

# print(df_combat.head())
# print(df_combat.columns)

# define features(X) and target(y)
X = df_combat.drop(['First_pokemon', 'Second_pokemon', 'Winner', '1st_pokemon_name',
       '1st_pokemon_type1', '2nd_pokemon_type1', '2nd_pokemon_name', 'First_win'],axis=1)
y = df_combat['First_win']

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# create model - Random Forest Classifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators=100)

rfc_model.fit(X_train,y_train)
prediction_rfc = rfc_model.predict(X_test)

# model evaluation
# print(classification_report(y_test,prediction_rfc))
# print(confusion_matrix(y_test,prediction_rfc))


pokemon1 = 'Charmander'
pokemon2 = 'Bulbasaur'

def battle(pokemon1, pokemon2):
    """
    Give two pokemon name to predict winner
    pokemon1 : str,
    pokemon2 : str,
    """

    if pokemon1 in df_pokemon['Name'].values and pokemon2 in df_pokemon['Name'].values:
        pokemon1 = df_pokemon[df_pokemon['Name']==pokemon1][['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed','Label Type 1']]
        pokemon2 = df_pokemon[df_pokemon['Name']==pokemon2][['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed','Label Type 1']]
        battle = np.concatenate([pokemon1.values, pokemon2.values], axis=1)
        prediction = rfc_model.predict(battle)[0] 
        if prediction == 1:
            prob = rfc_model.predict_proba(battle)[0][1] * 100
            print('{}% {} Wins!'.format(prob, pokemon1))
        else:
            prob = rfc_model.predict_proba(battle)[0][0] * 100
            print('{}% {} Wins!'.format(prob, pokemon2))
    else:
        print('Pokemon Not Found')

# battle(pokemon1,pokemon2)

import joblib
joblib.dump(rfc_model, 'Model_RFC_Deploy')