

from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('final_catboost_model')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('top.jpg')
    image_office = Image.open('side_image.jpeg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict pricing of cars')
    st.sidebar.success('https://www.ipsr.edu.in')
    st.sidebar.image(image_office,use_column_width=True)
    st.title("Predicting car price")
    
    
    if add_selectbox == 'Online':
        Present_Price=st.number_input('Present price' , min_value=0.0, max_value=35.0, value=0.1)
        Kms_Driven=st.number_input('Kms_Driven',min_value=100, max_value=550000, value=100)
        Fuel_Type = st.selectbox('Fuel type of car i.e Diesel,Petrol,CNG', ['Diesal', 'Petrol','CNG'])
        Seller_Type = st.selectbox('Seller_Type : Defines whether the seller is a dealer or an individual', ['Dealer','Individual'])
        Transmission= st.selectbox('Transmission : Defines whether the car is manual or automatic',  ['Manual','Automatic'])
        Owner = st.selectbox('Owner : Defines the number of owners the car has previously had',  ['0','1','3'])
        yr_old= st.number_input('yr_old:how many years ',  min_value=1, max_value=4, value=1)
        output=""
        input_dict={'Present_Price':Present_Price,'Kms_Driven':Kms_Driven,'Fuel_Type':Fuel_Type,'Seller_Type':Seller_Type,'Transmission': Transmission,'Owner':Owner,'yr_old' :  yr_old}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))

        
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
