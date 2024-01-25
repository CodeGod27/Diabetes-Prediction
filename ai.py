import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.header("Daibetes Prediction System")
st.header("Enter Data")

#read file 
patient_df = pd.read_csv("C:/Users/91976/Desktop/diabetes.csv")

df_x = patient_df.iloc[:,0:8]

df_y = patient_df.iloc[:,8]

x_train, x_test, y_train, y_test=train_test_split(df_x, df_y, test_size=0.20, random_state=1)

rf_model = RandomForestClassifier (random_state = 1)

rf_model.fit(x_train,y_train)

rf_predictions = rf_model.predict(x_test)



st_Pregnancies = st.number_input("Enter Number of Pregnancies  : ")
st_Glucose = st.number_input("Enter Glucose (mg/dL)Value : ")
st_BloodPressure = st.number_input("Enter BloodPressure(mm Hg) Value : ")
st_SkinThickness = st.number_input("Enter SkinThickness (mm) Value : ")
st_Insulin = st.number_input("Enter Insulin (Mum/ml)Value : ")
st_BMI = st.number_input("Enter BMI Value : ")
st_DiabetesPedigreeFunction = st.number_input("Enter DiabetesPedigreeFunction Value : ")
st_Age = st.number_input("Enter Age Value(years) : ")

user_data = [[st_Pregnancies,st_Glucose,st_BloodPressure,st_SkinThickness,st_Insulin,st_BMI,st_DiabetesPedigreeFunction,st_Age]]
cols = [["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]

pd_test_df = pd.DataFrame(user_data,columns = cols)

st.subheader('User Input')
st.write(pd_test_df)

rf_predict_user_data = rf_model.predict(pd_test_df)

if rf_predict_user_data == 0:
    result = 'You Are NOT Diabetic'
else :
    result = 'You Are Diabetic'
st.header("Your Report")


st.subheader('Accuracy: ')
st.write(str(metrics.accuracy_score(y_test, rf_predictions)))

st.header(' Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()

# Scatterplot for patient data
ax1 = sns.scatterplot(x='Age', y='Glucose', data=patient_df, hue='Outcome', palette='magma')

# Scatterplot for user data
ax2 = sns.scatterplot(x=[st_Age], y=[st_Glucose], s=150, color='red')

plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 220, 10))
plt.title('0 - Healthy & 1 - Unhealthy')

# Display the plot
st.pyplot(fig_glucose)
st.header(' Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()

# Scatterplot for patient data
ax1 = sns.scatterplot(x='Age', y='BloodPressure', data=patient_df, hue='Outcome', palette='Reds')
# Scatterplot for user data
ax2 = sns.scatterplot(x=[st_Age], y=[st_BloodPressure], s=150, color='Green')

plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 130, 10))
plt.title('0 - Healthy & 1 - Unhealthy')

# Display the plot
st.pyplot(fig_bp)


st.header(' Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()

# Scatterplot for patient data
ax1 = sns.scatterplot(x='Age', y='Insulin', data=patient_df, hue='Outcome', palette='Reds')
# Scatterplot for user data
ax2 = sns.scatterplot(x=[st_Age], y=[st_Insulin], s=150, color='Green')

plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 900, 50))
plt.title('0 - Healthy & 1 - Unhealthy')

# Display the plot
st.pyplot(fig_i)

st.subheader('Result')
st.write(result)

st.header(' Thank You !!')
st.header(' Visit Next Time !!!')