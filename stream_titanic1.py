

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('final_project_titanic')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('backchannel-how-to-escape-the-titanic.jpg')
    image_office = Image.open('images (1).jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict whether a person will survive or not?')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("Predicting employee leaving")
    if add_selectbox == 'Online':
        Pclass	=st.selectbox('Pclass' , ['1','2','3'])
        Sex =st.selectbox('Sex', ['male', 'female'])
        Age = st.number_input('Age', min_value=0, max_value=80, value=1)
        SibSp = st.number_input('SibSp', min_value=1, max_value=8, value=1)
        Parch = st.number_input('Parch',  min_value=0, max_value=6, value=1)
        Fare = st.number_input('Fare',  min_value=0, max_value=513, value=1)
        Embarked = st.selectbox('Embarked', ['S', 'C','Q'])
        output=""
        input_dict={'Pclass':Pclass,'Sex':Sex,'Age':Age,'SibSp': SibSp,'Parch':Parch,'Fare' : Fare,'Embarked' :Embarked}
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
