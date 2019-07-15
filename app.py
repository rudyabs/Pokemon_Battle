from flask import Flask, render_template, request, abort
import requests
import json
import numpy as np
import pandas as pd
import joblib
import io
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import base64


matplotlib.use('agg')
app = Flask(__name__)

df_pokemon = pd.read_csv('pokemon.csv')
label_type1_bio = LabelEncoder()
label_type1_bio.fit(df_pokemon['Type 1'])
df_pokemon['Label Type 1'] = label_type1_bio.transform(df_pokemon['Type 1'])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        pokemon1 = request.form['name1']
        pokemon2 = request.form['name2']
        
        if pokemon1 in df_pokemon['Name'].values and pokemon2 in df_pokemon['Name'].values:
            pokemon1 = df_pokemon[df_pokemon['Name']==pokemon1][['Name' ,'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Label Type 1']]
            pokemon2 = df_pokemon[df_pokemon['Name']==pokemon2][['Name' ,'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Label Type 1']]
            versus = np.concatenate([pokemon1.drop('Name', axis=1).values, pokemon2.drop('Name', axis=1).values], axis=1)
            prediction = model.predict(versus)[0] 

            # visualization
            
            compare = pd.concat([pokemon1, pokemon2])
            plt.close()
            plt.figure(figsize=(12,8))
            plt.subplot(161)
            plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['HP'], color=['red', 'blue'])
            plt.title('HP')
            plt.subplot(162)
            plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Attack'], color=['red', 'blue'])
            plt.title('Attack')
            plt.subplot(163)
            plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Defense'], color=['red', 'blue'])
            plt.title('Defense')
            plt.subplot(164)
            plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Sp. Atk'], color=['red', 'blue'])
            plt.title('Sp. Attack')
            plt.subplot(165)
            plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Sp. Def'], color=['red', 'blue'])
            plt.title('Sp. Defense')
            plt.subplot(166)
            plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Speed'], color=['red', 'blue'])
            plt.title('Speed')
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            graph = 'data:image/png;base64,{}'.format(graph_url)
            
            url1='https://pokeapi.co/api/v2/pokemon/'+pokemon1['Name'].values[0].lower()
            url2='https://pokeapi.co/api/v2/pokemon/'+pokemon2['Name'].values[0].lower()
            web1=requests.get(url1)
            web2=requests.get(url2)

            if str(web1)=='<Response [404]>':
                abort(404)
            else:
                gambar1=web1.json()['sprites']['front_default']
            
            if str(web2)=='<Response [404]>':
                abort(404)
            else:
                gambar2=web2.json()['sprites']['front_default']

            if prediction == 1:
                prob = model.predict_proba(versus)[0][1] * 100
                win = pokemon1
                result = {'prob':prob, 'win':win, 'graph':graph}
                return render_template('hasil.html', result=result, gambar1=gambar1, gambar2=gambar2)
            else:
                prob = model.predict_proba(versus)[0][0] * 100
                win = pokemon2
                result = {'prob':prob, 'win':win, 'graph':graph}
                return render_template('hasil.html', result=result, gambar1=gambar1, gambar2=gambar2)
        else:
            abort(404)

@app.errorhandler(404)
def error(error):
    return render_template('error.html')

if __name__ == "__main__":
    model = joblib.load('Model_RFC_Deploy')
    app.run(debug=True)