import streamlit as st
from dataset_characteristics import show_characteristics_page
# from dataset2_characteristics import show_characteristics_updated_page
from add_house_data import add_house_page
from prediction_house_price import show_prediction_page
from insights import show_insights
import base64
from IPython.display import HTML

image = 'UPF.png'
st.write("WEBAPP by CINTA ARNAU ARASA & ORIOL GALLEGO VAZQUEZ")
st.sidebar.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
page = st.sidebar.selectbox("Choose a section to visit:", ("Global Characteristics", "Add House Data", "Prediction", "Insights"))
show_background = st.sidebar.checkbox('Show background')
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    opacity: 0.9;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

if show_background:
    set_background('houses_background.png')

if page == "Global Characteristics":
    show_characteristics_page()
elif page == "Add House Data":
    add_house_page()
elif page == "Prediction":
    show_prediction_page()
else:
    show_insights()