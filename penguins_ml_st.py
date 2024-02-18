#  /* Load modules */
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using a model"
    " built on the Palmer Penguins dataset. Use the form bellow to get started")

password_guess = st.text_input('The uppercase abreviattion of the most beautiful and wonderful city in the world (hint: tree letters)', type='password')


if password_guess != 'MAP':  
    st.write('waiting the correct password...')
    st.stop()



penguin_file = st.file_uploader("Upload you own penguin data")
if penguin_file is None:
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
    penguin_df = pd.read_csv('penguins.csv')

else:
    # /* Load and preprocess data */
    penguin_df = pd.read_csv('penguins.csv')
    penguin_df.dropna(inplace=True)

    # /* Organize data for model input */
    output = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    features = pd.get_dummies(features)

    output, uniques = pd.factorize(output)
    x_train, x_text, y_train, y_test = train_test_split(features, output, test_size=0.8) 

    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values, y_train)
    y_pred = rfc.predict(x_text.values)

    score = round(accuracy_score(y_pred, y_test), 2)
    st.write(
        f"""We trained a Random Forest model on these
        data, it has a score of {score}! Use the
        inputs bellow to try out the model"""
    )

with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgersen'])
    sex = st.selectbox('Sex', options=["Female" , "Male"])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)

    submitted = st.form_submit_button()
    if submitted:
        island_biscoe, island_dream, island_torgerson = 0,0,0
        if island == 'Biscoe':
            island_biscoe = 1
        elif island == 'Dream':
            island_dream = 1
        elif island == 'Torgerson':
            island_torgerson = 1
        
        sex_female, sex_male = 0,0
        if sex == 'Female':
            sex_female = 1
        elif sex == 'Male':
            sex_male = 1

        new_predciton = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_biscoe, island_dream, island_torgerson, sex_female, sex_male]])

        prediction_species = unique_penguin_mapping[new_predciton][0]

        st.subheader("Predicting your penguin's species")
        
        st.write(f"We predict your penguin is of the {prediction_species} species")
        st.write(
            """We used a machine learning (Random Forest)
            model to predict the spcies, the features
            used in this prediction are ranked by
            relative importance bellow."""
        )
        st.image('feature_importance.png')
        st.write(
            """Bellow are histograms for each continuous variable separeted by penguin
            species. The vertical line represents your inputted value."""
        )

        fig, ax = plt.subplots()
        ax = sns.displot(x=penguin_df['bill_length_mm'], hue=penguin_df['species'])
        plt.axvline(x=bill_length, color='black', linestyle='--')
        plt.title('Bill Length (mm) by Species')
        st.pyplot(ax)

        fig, ax = plt.subplots()
        ax = sns.displot(x=penguin_df['bill_depth_mm'], hue=penguin_df['species'])
        plt.axvline(x=bill_depth, color='black', linestyle='--')
        plt.title('Bill Depth (mm) by Species')
        st.pyplot(ax)

        fig, ax = plt.subplots()
        ax = sns.displot(x=penguin_df['flipper_length_mm'], hue=penguin_df['species'])
        plt.axvline(x=flipper_length, color='black', linestyle='--')
        plt.title('Flipper Length (mm) by Species')
        st.pyplot(ax)        
        

