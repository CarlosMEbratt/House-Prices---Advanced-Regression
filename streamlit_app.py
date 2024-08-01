import time
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import pickle

# Page title
st.set_page_config(page_title='House Price Prediction App', page_icon='🏠', layout='centered', initial_sidebar_state='auto')

# Define column names
column_names = [
    'OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond', 'TotalBsmtSF', 'HeatingQC', 'GrLivArea',
    'KitchenQual', 'Fireplaces', 'FireplaceQu', 'GarageCars', 'GarageQual',
    'GarageCond', 'WoodDeckSF', 'OpenPorchSF', 'TotalBath', 'TotalHalfBath'
]

# Main Function
def main():
    tabs = st.tabs(["Upload Data", "Enter Data"])

    uploaded_file = None
    loaded_model = None
    new_data = pd.DataFrame()

    with tabs[0]:
        st.header('House Price Prediction App 🏠')

        st.markdown("**1. Load the house data**")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, index_col=False)      

            # Initiate the model building process
            st.subheader('Processing the data')
            st.write('Processing in progress...')

            # Placeholder for model building process
            with st.spinner('Wait for it...'):
                time.sleep(2)

            st.markdown(''':blue[House data has been loaded!]''')
            st.dataframe(data=df, use_container_width=True)

    with tabs[1]:
        st.header('Enter Data Manually')

        st.markdown("**2. Enter data for prediction**")

        # Create a form for entering new data
        with st.form("Enter New Data"):
            input_data = {col: st.text_input(col, "") for col in column_names}
            predict_button = st.form_submit_button("Predict Price")

            if predict_button:
                # Create a new DataFrame from the input data
                try:
                    new_data = pd.DataFrame([input_data])
                    st.success('Data entered successfully!')
                    st.write('New Data for prediction:')
                    st.dataframe(new_data, use_container_width=True)

                    # Perform inference using the loaded model
                    if loaded_model is not None:
                        try:
                            prediction = loaded_model.predict(new_data)
                            new_data['Predicted_Price'] = prediction

                            # Display prediction
                            st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

                            # Plotting the histogram of predicted prices
                            fig = px.histogram(new_data, x='Predicted_Price', nbins=50, title='Distribution of Predicted House Prices')
                            fig.update_layout(xaxis_title='Predicted Sale Price', yaxis_title='Frequency')
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                    else:
                        st.error("Model is not loaded. Please load a model first.")
                except Exception as e:
                    st.error(f"Error creating DataFrame: {e}")

    st.markdown('**3. Load the model**')
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

# Call the main function
if __name__ == '__main__':
    main()
