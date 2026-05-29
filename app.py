import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image

# --- 0. GLOBAL SETTINGS ---
LOG_FILE = "clinical_records.csv"
USER_FILE = "users.csv"



# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Biceph-Net Portal", layout="wide")
# --- 0. HIDE DEPLOY BUTTON AND FOOTER ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Initialize User Database if it doesn't exist
if not os.path.exists(USER_FILE):
    pd.DataFrame([
        {"username": "drsmith", "name": "Dr. Smith", "password": "pass123"},
        {"username": "admin", "name": "Administrator", "password": "admin123"}
    ]).to_csv(USER_FILE, index=False)

def log_diagnosis(p_id, doctor, diag):
    data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M")],
        "Patient_ID": [p_id],
        "Physician": [doctor],
        "Result": [diag]
    }
    df = pd.DataFrame(data)
    df.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))

# --- 2. AUTHENTICATION LOGIC ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def auth_portal():
    st.title("Biceph-Net Medical Portal")
    
    # Selection for Login or Sign Up
    mode = st.radio("Select Action", ["Sign In", "Sign Up"], horizontal=True)
    
    if mode == "Sign In":
        user_input = st.text_input("Username")
        pass_input = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            df_users = pd.read_csv(USER_FILE)
            # Ensure password comparison is string-based
            match = df_users[(df_users['username'] == user_input) & (df_users['password'].astype(str) == pass_input)]
            
            if not match.empty:
                st.session_state["authenticated"] = True
                st.session_state["name"] = match.iloc[0]['name']
                st.rerun()
            else:
                st.error("Invalid Username or Password")
                
    else:
        st.subheader("Create New Medical Account")
        new_name = st.text_input("Full Name (e.g. Dr. John Doe)")
        new_user = st.text_input("Choose Username")
        new_pass = st.text_input("Choose Password", type="password")
        
        if st.button("Register Account", use_container_width=True):
            df_users = pd.read_csv(USER_FILE)
            if new_user in df_users['username'].values:
                st.error("Username already exists. Please choose another.")
            elif new_name and new_user and new_pass:
                new_entry = pd.DataFrame([{"username": new_user, "name": new_name, "password": new_pass}])
                pd.concat([df_users, new_entry], ignore_index=True).to_csv(USER_FILE, index=False)
                st.success("Account created! You can now switch to 'Sign In'.")
            else:
                st.warning("Please fill in all fields.")

# --- 3. PROTECTED APP CONTENT ---
if not st.session_state["authenticated"]:
    # Center the login form slightly
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_portal()
else:
    # --- LOGOUT BUTTON ---
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()
        
    st.sidebar.success(f"Physician: {st.session_state['name']}")

 # --- 4. PROTECTED APP CONTENT ---


    # --- THE ADMIN SECURITY CHECK ---
    # Only the user with the exact name "Administrator" gets the 3rd tab
    is_admin = (st.session_state["name"] == "Administrator")

    if is_admin:
        tabs = st.tabs(["🔍 Diagnostic Portal", "📊 Clinical Insights", "🛡️ Admin Panel"])
    else:
        tabs = st.tabs(["🔍 Diagnostic Portal", "📊 Clinical Insights"])

    # --- TAB 1: NEW DIAGNOSIS ---
    with tabs[0]:
        st.title(" Parkinson's Disease Detection")
        
        # Patient Metadata Input
        col_id, col_file = st.columns([1, 2])
        with col_id:
            patient_id = st.text_input("Patient ID/Name:", placeholder="e.g. PT-405")
        with col_file:
            uploaded_file = st.file_uploader("Upload MRI Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

        if uploaded_file and patient_id:
            img = Image.open(uploaded_file)
            st.image(img, caption=f"Processing scan for {patient_id}", width=300)
            
            if st.button("Generate Report"):
                with st.spinner("Analyzing Triplet-Loss Features..."):
                    # Load Models (Cached)
                    @st.cache_resource
                    def load_ai_engine():
                        cnn = tf.keras.models.load_model("models/feature_extractor.h5", compile=False)
                        knn_model = joblib.load("models/knn_model.pkl")
                        return cnn, knn_model
                    
                    model, knn = load_ai_engine()

                    # Preprocessing & Prediction
                    img_prep = np.array(img.convert('RGB').resize((121, 121))) / 255.0
                    raw_pred = model.predict(np.expand_dims(img_prep, axis=0), verbose=0)
                    
                    # Handle Multi-output & Reshape
                    features = np.array(raw_pred[0]).reshape(1, -1) if isinstance(raw_pred, list) else raw_pred.reshape(1, -1)
                    
                    res_idx = knn.predict(features)[0]
                    prob = np.max(knn.predict_proba(features)) * 100
                    diagnosis = "Parkinson's Detected" if res_idx == 1 else "Normal / Healthy"

                    # Output UI
                    st.divider()
                    if res_idx == 1:
                        st.error(f"### DIAGNOSIS: {diagnosis}")
                    else:
                        st.success(f"### DIAGNOSIS: {diagnosis}")
                    
                    
                    # Save to database
                    log_diagnosis(patient_id, st.session_state["name"], diagnosis)
                    st.toast("Clinical record saved successfully.")

    # --- TAB 2: CLINICAL DASHBOARD ---
    with tabs[1]:
        st.title("📊 Clinical Statistics Dashboard")
        
        if os.path.exists(LOG_FILE):
            history_df = pd.read_csv(LOG_FILE)
            
            # KPI Metrics
            total = len(history_df)
            pd_cases = len(history_df[history_df['Result'] == "Parkinson's Detected"])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Diagnoses", total)
            m2.metric("PD Positive", pd_cases)
            m3.metric("Normal/Healthy", total - pd_cases)

            # Visual Distribution Chart
            st.divider()
            
            fig = px.pie(history_df, names='Result', 
                         title='Diagnosis Distribution',
                         color_discrete_map={"Normal / Healthy":"#28a745", "Parkinson's Detected":"#dc3545"},
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

            # Record Table
            st.subheader("📋 Recent Clinical Logs")
            st.dataframe(history_df.iloc[::-1], use_container_width=True)
            
            # Export Option
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Patient Records (CSV)", csv, "medical_report.csv", "text/csv")
        else:
            st.info("No records found. Complete a diagnosis in the first tab to populate the dashboard.")

   # --- TAB 3: ADMIN PANEL (HIDDEN FROM DOCTORS) ---
    if is_admin:
        with tabs[2]:
            st.title("🛡️ System Administration")
            
            # --- 1. MANAGE DOCTORS ---
            st.subheader("👥 Medical Staff Management")
            if os.path.exists(USER_FILE):
                df_users = pd.read_csv(USER_FILE)
                
                h1, h2, h3 = st.columns([2, 2, 2])
                h1.write("**Name**")
                h2.write("**Username**")
                h3.write("**Actions**")
                st.divider()

                for idx, row in df_users.iterrows():
                    if row['username'] == 'admin': continue 
                    
                    c1, c2, c3 = st.columns([2, 2, 2])
                    c1.write(row['name'])
                    c2.write(f"`{row['username']}`")
                    
                    with c3:
                        sub_col1, sub_col2 = st.columns(2)
                        if sub_col1.button("📝 Edit", key=f"edit_u_{idx}"):
                            st.session_state[f"editing_user_{idx}"] = True
                        
                        if sub_col2.button("🗑️ Del", key=f"del_u_{idx}"):
                            df_users.drop(idx).to_csv(USER_FILE, index=False)
                            st.rerun()
                    
                    if st.session_state.get(f"editing_user_{idx}", False):
                        with st.expander(f"Editing {row['username']}", expanded=True):
                            new_n = st.text_input("Full Name", value=row['name'], key=f"n_{idx}")
                            new_p = st.text_input("Password", value=row['password'], key=f"p_{idx}")
                            if st.button("Update Physician", key=f"save_u_{idx}"):
                                df_users.at[idx, 'name'] = new_n
                                df_users.at[idx, 'password'] = new_p
                                df_users.to_csv(USER_FILE, index=False)
                                del st.session_state[f"editing_user_{idx}"]
                                st.rerun()

            st.divider()

            # --- 2. MANAGE CLINICAL REPORTS (NO CONFIDENCE) ---
            st.subheader("📋 Clinical Report Management")
            if os.path.exists(LOG_FILE):
                df_reports = pd.read_csv(LOG_FILE)
                
                # Check if Confidence exists in old records and drop it if it does
                if 'Confidence' in df_reports.columns:
                    df_reports = df_reports.drop(columns=['Confidence'])

                for idx, row in df_reports.iterrows():
                    with st.container(border=True):
                        # Removed the confidence column from this layout
                        r1, r2, r3, r4 = st.columns([2.5, 2.5, 2.5, 1.5])
                        
                        r1.write(f"**Patient:** {row['Patient_ID']}")
                        r2.write(f"**Result:** {row['Result']}")
                        r3.write(f"**Date:** {row['Timestamp']}")
                        
                        with r4:
                            if st.button("Edit ✍️", key=f"edit_r_{idx}"):
                                st.session_state[f"editing_rep_{idx}"] = True
                            if st.button("Delete 🗑️", key=f"del_r_{idx}"):
                                df_reports.drop(idx).to_csv(LOG_FILE, index=False)
                                st.rerun()
                        
                        if st.session_state.get(f"editing_rep_{idx}", False):
                            new_id = st.text_input("Change Patient ID", value=row['Patient_ID'], key=f"id_{idx}")
                            new_res = st.selectbox("Change Diagnosis", ["Normal / Healthy", "Parkinson's Detected"], 
                                                 index=0 if row['Result'] == "Normal / Healthy" else 1, key=f"res_{idx}")
                            
                            if st.button("Save Changes", key=f"save_r_{idx}"):
                                df_reports.at[idx, 'Patient_ID'] = new_id
                                df_reports.at[idx, 'Result'] = new_res
                                df_reports.to_csv(LOG_FILE, index=False)
                                del st.session_state[f"editing_rep_{idx}"]
                                st.rerun()
            else:
                st.info("No records found.")

st.divider()
st.caption("Biceph-Net AI Research Tool v1.2 | Authorized Personnel Only")