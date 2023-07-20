import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Define function to create signup form
def signup():
    st.write("# Signup")
    # Create input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    # Create button to submit signup form
    if st.button("Signup"):
        # Add code to check if username already exists
        user_data = pd.read_csv("user_data.csv")
        if (user_data["username"] == username).any():
            st.error("Username already exists. Please choose a different username.")
        else:
            # Add code to validate password
            if not any(char.isupper() for char in password):
                st.error("Password must contain at least one capital letter.")
            elif not any(char.isdigit() for char in password):
                st.error("Password must contain at least one number.")
            elif not any(char in "!@#$%^&*()_+-=[]{}|;:,.<>/?`~" for char in password):
                st.error("Password must contain at least one special character.")
            else:
                # Add code to save user data to database or file
                signup_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                user_data = pd.DataFrame({"username": [username], "password": [password], "signup_date": [signup_date]})
                user_data.to_csv("user_data.csv", index=False)
                st.success("You have successfully signed up")

# Define function to create login form
def login():
    st.write("# User Login")
    # Create input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    # Create button to submit login form
    if st.button("Login"):
        # Add code to check user data against database or file
        user_data = pd.read_csv("user_data.csv")
        if (user_data["username"] == username).any() and (user_data["password"] == password).any():
            st.session_state.logged_in = True
            st.success("You have successfully logged in")
            os.system("streamlit run app.py")
        else:
            st.error("Invalid username or password")

# Define function to create logout button
def logout():
    st.write("# Logout")
    # Create button to log out user
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.success("You have successfully logged out")

# Define main function
def main():
    st.set_page_config(page_title="Login/Signup App")
    st.title("Welcome to Stock Price Prediction web app")
    # Check if user is logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    # Create menu with options to login, signup, or logout
    if not st.session_state.logged_in:
        menu = ["Login", "Signup"]
        choice = st.sidebar.selectbox("Select an option", menu)
        # Show appropriate form based on user's choice
        if choice == "Login":
            login()
        else:
            signup()
    else:
        logout()

# Call main function
if __name__ == "__main__":
    main()
