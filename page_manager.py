import streamlit as st
import SessionState

''' 

            Page manager

Streamlit will require multiple pages from this project so this .py file will handle that.


'''

class PageManager:
    def __init__(self):
        self.apps = []
        prev, _ ,next = st.columns([1, 10, 1])
        test = st.button('Enter')
