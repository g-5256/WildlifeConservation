import streamlit as st
from counting import animal_counting
from classification import animal_classification
from information import animal_information
import pandas as pd
import numpy as np
import time


def display_homepage():
    """
    Display the homepage with a compelling slogan and introduction.
    """
    # st.markdown(
    #      f"""
    #      <style>
    #      .stApp {{
    #          background: url("https://happydays365.org/wp-content/uploads/2019/12/World-Wildlife-Conservation-Day-980x653.jpg");
    #          background-size: cover
    #      }}
    #      </style>
    #      """,
    #      unsafe_allow_html=True
    #  )

    st.image("/home/gaurav/Downloads/elephant-48423.svg", use_column_width=True)  # Add an attractive wildlife image
    st.markdown("<h1 style='text-align: lcenter; color: green;'>Preserve the Paws & Claws!</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Join hands in making the world a sanctuary for wildlife.</h3>", unsafe_allow_html=True)
    
    st.write("""
    ## About the Project
    This Wildlife Conservation Web Application leverages cutting-edge technology to champion the cause of wildlife conservation. 
    Here's what you can do:
    - **Count Animals**: Use advanced computer vision techniques to estimate animal populations from images.
    - **Classify Species**: Determine if a species is endangered or extinct using state-of-the-art AI models.
    - **Discover Animal Information**: Dive deep into the world of various species, understanding their habitats, migratory patterns, and much more.
    
    Let's harness the power of technology to make a lasting impact on our planet's biodiversity.
    """)
    
    with st.status("Downloading data...", expanded=True) as status:
        st.write("Searching for data...")
        time.sleep(2)


        st.write("Found URL.")
        time.sleep(1)
        st.write("Downloading data...")
        time.sleep(1)
        status.update(label="Download complete!", state="complete", expanded=False)

st.button('Rerun')


def main():
    st.title("Wildlife Conservation Web Application")


    



    # Sidebar with different functionalities
    st.sidebar.title("Features")
    choice = st.sidebar.radio("Select a feature", ["Home", "Animal Counting", "Animal Classification", "Animal Information"])

    if choice == "Home":
        display_homepage()

    elif choice == "Animal Counting":
        st.header("Animal Counting")
        animal_counting()

    elif choice == "Animal Classification":
        st.header("Animal Classification")
        animal_classification()

    elif choice == "Animal Information":
        st.header("Animal Information")
        animal_information()

    
    st.write ("See Map below for the location of the animals")
    df = pd.DataFrame({
    "col1": np.random.randn(1000) / 50 + 37.76,
    "col2": np.random.randn(1000) / 50 + -122.4,
    "col3": np.random.randn(1000) * 100,
    "col4": np.random.rand(1000, 4).tolist(),})

    st.map(df,
    latitude='col1',
    longitude='col2',
    size='col3',
    color='col4')

if __name__ == "__main__":
    main()

