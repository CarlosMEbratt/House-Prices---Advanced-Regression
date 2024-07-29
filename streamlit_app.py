import time
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import pickle


#'''Streamlit settings------------------------------------------------------------------------------------------------------------- '''

# Page title
st.set_page_config(page_title='House Price Prediction App', page_icon='🏠', layout='centered', initial_sidebar_state='auto')




#'''Main Function------------------------------------------------------------------------------------------------------------- '''
    

def main():
    
     
    with st.sidebar:

        with st.expander('About this app / Instructions'):
                    st.markdown('**What can this app do?**')
                    st.info('This app allow users to load a housedata.csv file and use it to build a machine learning model to predict house prices.')

                    st.markdown('**How to use the app?**')
                    st.warning('1. Select a data set and 2. Click on "Run the model". As a result, this would initiate the ML model and data processing.')

                    st.markdown('**Under the hood**')
                    st.markdown('Data sets:')
                    st.code('''- You can upload your own data set or use the example data set provided in the app.
                    ''', language='markdown')
                    
                    st.markdown('Libraries used:')
                    st.code('''
                            * Pandas for data wrangling  
                            * Scikit-learn
                            * RandomForestRegressor for machine learning
                            * Streamlit for user interface
                    ''', language='markdown')


    st.header('Input data')
    st.markdown("**1. Load the house' data**")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)      

    
    #'''--------------------------------------------------------------------------------------

    # Initiate the model building process
    if uploaded_file:  
        st.subheader('Processing the data')
        st.write('Processing in progress...')

        # Placeholder for model building process
        with st.spinner('Wait for it...'):
            time.sleep(2)

        #st.write('Customer predictions are now complete!')
        st.markdown(''':blue[House data has been loaded!]''')

        st.dataframe(data=df, use_container_width=True)


    #'''--------------------------------------------------------------------------------------

    st.markdown('**2. Load the saved model**')
    # Load the saved model
    uploaded_pkl = st.file_uploader("Upload .pkl file", type=["pkl"])

    # Check if a file is uploaded
    if uploaded_pkl is not None:
        st.write("File uploaded successfully!")

        try:
            # Load the model from the file
            loaded_model = pd.read_pickle(uploaded_pkl)
            st.success("Model loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading .pkl file: {e}")

    else:
        st.info("Please upload a .pkl file.")


    #'''--------------------------------------------------------------------------------------
    st.markdown('**3. Predict House Prices**')
    # Load the saved model
    if st.button('Predict'):
        # # Convert input data to numpy array
        #input_data_np = np.array(df)  # Adjust input data format as needed

        # Perform inference using the loaded model
        prediction = loaded_model.predict(df)
        df['predictions'] = prediction
        # Display prediction
        st.dataframe(data=prediction, use_container_width=True)

# Call the main function
if __name__ == '__main__':
    main()