import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image


bgimage_path = "CCP LOGO.png"  


page_bg_img = """
<style>
[data-testid="stVerticalBlock"]{
background: radial-gradient(circle, rgba(0,0,0,1) 70%, rgba(0,0,0,1) 140%);
padding: 30px;  /* Add padding inside */
max-width: 100%; 
border-radius: 25px;  /* Rounded corners */
margin: auto;  /* Center it */
padding: 30px;  /* Add padding inside */
}

[data-testid="stHeader"] {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            
        }

        [data-testid="stHeader"]::before {
            content: "";
            background: url('https://i.ibb.co/PsCnYd4S/CCP-LOGO.png') no-repeat center;
            background-size: contain;
            width: 300px; /* Adjust size */
            height: 100px;
            display: inline-block;
            margin-left: 10px;
        }

[data-testid="stMain"]{
background: background-color: #0d0d0d;
opacity: 1;
background: linear-gradient(135deg, #444cf755 25%, transparent 25%) -40px 0/ 80px 80px, linear-gradient(225deg, #444cf7 25%, transparent 25%) -40px 0/ 80px 80px, linear-gradient(315deg, #444cf755 25%, transparent 25%) 0px 0/ 80px 80px, linear-gradient(45deg, #444cf7 25%, #0d0d0d 25%) 0px 0/ 80px 80px;
bakground-position:top left;
background-attachment: local;        
        }

h1 {
            color: #fffff !important;  /* Light White */
            text-align: center;
        }

div[data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            width: 100% !important;
            opacity:.98;
        }

 div[data-testid="stSelectbox"],
 div[data-testid="stNumberInput"],
 div[data-testid="stTextInput"]
{
            background-color: rgba(0, 0, 0, .3) ;
            border-radius: 10px ;
            padding: 5px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            max-width: 80%;
            text-align: center ;
            margin: auto; 
            
        }

 
 



}
</style>



"""

st.markdown(page_bg_img, unsafe_allow_html=True)

image = Image.open("chrun.png")

st.image(image, width=1920)


def main():
    # Load the trained model and encoders

    model = pickle.load(open("lr_best_model.sav", "rb"))
    scaler = pickle.load(open("scaler.sav", "rb"))
    oh_gender = pickle.load(open("ohGender.sav", "rb"))
    oh_feedback = pickle.load(open("ohFeedback.sav", "rb"))
    oh_mcat = pickle.load(open("ohMcat.sav", "rb"))
    oh_sdis = pickle.load(open("ohSdis.sav", "rb"))
    
    st.title("Customer Churn Predictor")
    
    # User inputs

    avg_time_spent = st.number_input("Avg Time Spent", min_value=0.0, format="%.2f")
    avg_transaction_value = st.number_input("Avg Transaction Value", min_value=0.0, format="%.2f")
    points_in_wallet = st.number_input("Points In Wallet", min_value=0.0, format="%.2f")
    gender = st.selectbox("Gender", ["M", "F"])
    membership_category = st.selectbox("Membership Category", ["Basic Membership", "Gold Membership", "No Membership", "Platinum Membership",  "Premium Membership", "Silver Membership"])
    special_discounts = st.selectbox("Used Special Discounts", ["Yes", "No"])
    feedback = st.selectbox("Customer Feedback", ["No reason specified", "Poor Customer Service", "Poor Product Quality", "Poor Website", "Products always in Stock", "Quality Customer Care", "Reasonable Price", "Too many ads", "User Friendly Website"])
    
    if st.button("Predict"):
        try:
            # Prepare numerical input data

            input_data = pd.DataFrame({
                "avg_time_spent": [avg_time_spent],
                "avg_transaction_value": [avg_transaction_value],
                "points_in_wallet": [points_in_wallet]
            })
            
            # One-hot encode categorical variables

            df_gender = pd.DataFrame(oh_gender.transform([[gender]]),
                                     columns=oh_gender.get_feature_names_out(["gender"]))
            df_mcat = pd.DataFrame(oh_mcat.transform([[membership_category]]),
                                     columns=oh_mcat.get_feature_names_out(["membership_category"]))
            df_sdis = pd.DataFrame(oh_sdis.transform([[special_discounts]]),
                                     columns=oh_sdis.get_feature_names_out(["used_special_discount"]))
            df_feedback = pd.DataFrame(oh_feedback.transform([[feedback]]),
                                        columns=oh_feedback.get_feature_names_out(["feedback"]))
            
            # Combine all input data

            input_data = pd.concat([input_data, df_gender, df_mcat, df_sdis, df_feedback ], axis=1)
            
            # Ensure column order matches the trained scaler

            expected_columns = scaler.feature_names_in_
            input_data = input_data[expected_columns]  # Maintain correct order
            input_data_scaled = scaler.transform(input_data)
            
            # Make prediction

            churn_prediction = model.predict(input_data_scaled)
            
            # Display result
            
            if churn_prediction[0] == 1:
                st.error(f"The customer is likely to churn.")
            else:
                st.success(f"The customer is not likely to churn.")
        
        except Exception as e:
            st.error(f"Prediction error: {e}")

main()
