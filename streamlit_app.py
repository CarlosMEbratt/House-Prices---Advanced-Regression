import time
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import pickle

# Page title
st.set_page_config(page_title='House Price Prediction App', page_icon='🏠', layout='centered', initial_sidebar_state='auto')

# Main Function
def main():
    with st.sidebar:
        with st.expander('About this app / Instructions', expanded=True):
            st.markdown('**What can this app do?**')
            st.info('This app allows users to load a house data CSV file and use it to build a machine learning model to predict house prices.')

            st.markdown('**How to use the app?**')
            st.warning('1. Upload a data set using the browse button, 2. Select an option to load the model, and 3. Click on "Predict". This will initiate the ML model and data processing.')

            st.markdown("""You can download a sample CSV file from the following link:
            [Download Sample CSV](https://github.com/CarlosMEbratt/House-Prices---Advanced-Regression/blob/main/df_to_load.csv)
            """, unsafe_allow_html=True)

            st.markdown('**Under the hood**')
            st.markdown('Data sets:')
            st.code('''- You can upload your own data set or use the example data set provided in the app.
            ''', language='markdown')
            
            st.markdown('Libraries used:')
            st.code('''
                    * Streamlit for user interface
                    * Pandas for data wrangling  
                    * Scikit-learn
                    * Picke with RandomForestRegressor for machine learning
                    
            ''', language='markdown')

    st.header('House Price Prediction App 🏠')

    st.markdown("**1. Load the house data**")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)      

    # Initiate the model building process
    if uploaded_file:  
        st.subheader('Processing the data')
        st.write('Processing in progress...')

        # Placeholder for model building process
        with st.spinner('Wait for it...'):
            time.sleep(2)

        st.markdown(''':blue[House data has been loaded!]''')
        st.dataframe(data=df, use_container_width=True)

    # Option to select how to load the model
    st.markdown('**2. Load the model**')
    option = st.radio("Choose how to load the model:", ('Automatically from GitHub', 'Manually by uploading a file'))

    @st.cache_data
    def load_pkl_file_from_url(url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful

            # Check if the response content type is correct
            if 'application/octet-stream' not in response.headers['Content-Type']:
                st.error(f"Unexpected content type: {response.headers['Content-Type']}")
                return None

            # Check the first few bytes of the file
            if response.content[:2] != b'\x80\x04':
                st.error("File does not appear to be a valid pickle file.")
                return None

            return pickle.loads(response.content)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            return None
        except pickle.UnpicklingError as e:
            st.error(f"Failed to unpickle the file: {e}")
            return None

    loaded_model = None

    if option == 'Automatically from GitHub':
        # URL of the .pkl file in your GitHub repository
        url = 'https://raw.githubusercontent.com/CarlosMEbratt/House-Prices---Advanced-Regression/main/best_rf.pkl'
        loaded_model = load_pkl_file_from_url(url)

        if loaded_model:
            st.write("Model loaded successfully!")
        else:
            st.error("Failed to load the model.")

    else:
        # Load the saved model manually
        uploaded_pkl = st.file_uploader("Upload .pkl file", type=["pkl"])

        # Check if a file is uploaded
        if uploaded_pkl is not None:
            st.write("File uploaded successfully!")

            try:
                # Load the model from the file
                loaded_model = pickle.load(uploaded_pkl)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading .pkl file: {e}")

        else:
            st.info("Please upload a .pkl file.")

    # Prediction
    st.markdown('**3. Predict House Prices**')

    if st.button('Predict'):
        if loaded_model is None:
            st.error("Model is not loaded. Please load a model first.")
        elif uploaded_file is None:
            st.error("House data is not loaded. Please upload a house data CSV file.")
        else:
            # Perform inference using the loaded model
            prediction = loaded_model.predict(df)
            df['Predicted_Price'] = prediction

            # Display prediction
            st.dataframe(data=df, use_container_width=True)

            # Plotting the histogram of predicted prices
            fig = px.histogram(df, x='Predicted_Price', nbins=50, title='Distribution of Predicted House Prices')
            fig.update_layout(xaxis_title='Predicted Sale Price', yaxis_title='Frequency')
            st.plotly_chart(fig)

# Call the main function
if __name__ == '__main__':
    main()
