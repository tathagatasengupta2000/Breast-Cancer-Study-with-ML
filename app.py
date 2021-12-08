import streamlit as st
import Functions as Fn
def my_widget(key):
        return st.button(key)
with st.sidebar:
        clicked1 = my_widget("Theory")
        #clicked2 = my_widget("Types of Cancer")
        #clicked3 = my_widget("Breast Cancer")
        #clicked4 = my_widget("Responsible Genetic Mutation")
        #clicked5 = my_widget("Classification Algorithim used")
        #clicked6 = my_widget("Dataset Information")
        #clicked7 = my_widget("Model Selection")
        clicked8 = my_widget("Model Selection")

Fn.intro()

if clicked1:
    Fn.show_page()
    Fn.Cancer_types()
    Fn.Breast_Cancer()
    Fn.Genetic_Mutation()
    Fn.ClassificationAlgorithm()
    Fn.DatasetInfo()
if True:
    lis=Fn.get_data()
    #st.markdown(lis[9:])
    Fn.Model_Selection(lis[9:])

