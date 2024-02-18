import streamlit as st
import pickle

# Replace 'your_pickle_file.pickle' with the path to your actual pickle file
pickle_file_path = st.text_input('Type the name of your_pickle_file.pickle, then press Enter')


if pickle_file_path != '':
    st.write(pickle_file_path)

    # Open the pickle file in binary read mode
    with open(pickle_file_path, 'rb') as file:
        # Load the content
        loaded_object = pickle.load(file)

        # Print or inspect the loaded object
        st.write(loaded_object)
        # If it's a model, you can also print its parameters or other relevant attributes
        # For example, if the object is a scikit-learn model:
        st.write(f"Model Parameters: \n{loaded_object.get_params()}")
