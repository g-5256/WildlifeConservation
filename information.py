import streamlit as st
import pandas as pd

# Load the dataset
animal_data = pd.read_csv('/home/gaurav/Desktop/gpt4web/wildlife_conservation_app/data/AnimalDataset.csv')

def fetch_animal_info_from_dataset(animal_name):
    """
    Fetch animal information from the provided dataset.
    """
    #make for three words search
    animal_name = animal_name.lower().strip()
    animal_name = animal_name.split()
    if len(animal_name) == 1:
        info = animal_data[animal_data['Animal'] == animal_name[0]].squeeze()
        
    info = animal_data[animal_data['Animal'] == animal_name].squeeze()
    return info if not info.empty else None

def display_conservation_status(status):
    """
    Display conservation status with color coding.
    """
    status_colors = {
        'Least Concern': 'green',
        'Vulnerable': 'orange',
        'Endangered': 'red',
        'Critically Endangered': 'darkred'
    }
    color = status_colors.get(status, 'blue')
    st.markdown(f"<div style='color: {color}; font-size: 24px;'>{status}</div>", unsafe_allow_html=True)

def animal_information():
    """
    Display the animal information from the dataset with enhanced visual presentation.
    """
    st.subheader("Enter the name of the animal:")
    animal_name = st.text_input("", "")
    
    if animal_name:
        st.write("Fetching information for", animal_name)
        info = fetch_animal_info_from_dataset(animal_name)
        
        if info is not None:
            # Assuming dataset contains a column 'Image_URL' with links to images
            # If not, placeholder images can be used
            image_url = info.get('Image_URL', 'https://placekitten.com/200/200')  # Placeholder image
            st.image(image_url, use_column_width=True)
            
            st.markdown("**Height:** " + info['Height (cm)'])
            st.markdown("**Weight:** " + info['Weight (kg)'])
            st.markdown("**Color:** " + info['Color'])
            st.markdown("**Lifespan:** " + info['Lifespan (years)'])
            
            st.write("Diet:", info['Diet'])
            st.write("Habitat:", info['Habitat'])
            st.write("Predators:", info['Predators'])
            st.write("Average Speed (km/h):", int(info['Average Speed (km/h)']))
            st.write("Countries Found:", info['Countries Found'])
            
            st.write("Conservation Status:")
            display_conservation_status(info['Conservation Status'])
            
            st.write("Family:", info['Family'])
            st.write("Gestation Period (days):", info['Gestation Period (days)'])
            st.write("Top Speed (km/h):", info['Top Speed (km/h)'])
            st.write("Social Structure:", info['Social Structure'])
            st.write("Offspring per Birth:", info['Offspring per Birth'])
        else:
            st.write("No information found for this animal.")

