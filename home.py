import streamlit as st

def home():
    st.Page('home.py', title="Home")
    st.header('Home')


pages = {
    "Home": [
        st.Page(home)
    ],
    "Models": [
        st.Page('logistic.py', title='Cardiovascular - Logistic'),
        st.Page('svm.py', title='Cardiovascular - SVM')
    ]
}

pg = st.navigation(pages, position='top')
pg.run()