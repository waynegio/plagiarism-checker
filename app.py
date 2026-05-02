import streamlit as st

home = st.Page("Home.py")
page1 = st.Page("pages/1_How_It_Works.py")
page2 = st.Page("pages/2_Dataset_Exploration.py")
page3 = st.Page("pages/3_Build_The_Model.py")
page4 = st.Page("pages/4_Let's_Try.py")
pg = st.navigation([home, page1, page2, page3, page4], position="sidebar")

st.sidebar.title("P A I R C H E C K")
st.sidebar.markdown("---")
pg.run()
st.sidebar.markdown("""
<div style='font-weight: 500; font-size:18px; color:#4E4E61; margin-bottom:4px;'>
    Made by (Group 3):
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='font-weight: 300; font-size:14px; color:#8D8D93;'>
    Hazel Div Alden<br>
    Venesia Arisaputri<br>
    Wayne Giovanno
</div>
""", unsafe_allow_html=True)