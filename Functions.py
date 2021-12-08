import streamlit as st
import pandas as pd
import joblib
import Content as con
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import LRegression as LR

df=pd.read_csv("Data.csv")
KNN_Model=joblib.load("KNN_Model")
SVM_Model=joblib.load("SVM_Model")
NB_Model=joblib.load("NB_Model")
LR_Model=joblib.load("LR_Model")
KSVM_Model=joblib.load("KSVM_Model")
RF_Model=joblib.load("RF_Model")
DT_Model=joblib.load("DT_Model")
X = df.iloc[:,1:-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
def exp(Str,Loc):
    my_expander = st.expander(Str, expanded=False)
    with my_expander:
        st.markdown(Loc)
def Conclusion():
    st.title("Conclusion Reached")
    st.markdown(con.conclusion)
    exp("Logistic Regression",con.LR2)
    exp("Support Vector Machine",con.SVM2)
    exp("K Nearest Neighbour",con.KNN2)
    exp("Kernel SVM",con.KSVM2)
    exp("Decision Tree",con.DT2)
    exp("Random Forest",con.RF2)
    exp("Naive Bayes",con.NB2)
    st.markdown(con.Conlusion2)
    st.info('Created by Tathagata Sengupta')
def intro():
    st.header("WElCOME")
    st.header("Study on Breast Cancer with Machine Learning")
    st.text("Created By Tathagata Sengupta")
def show_page():
    st.title("What is Cancer?")
    st.markdown(con.definition)
    st.subheader("Cancer Cell Features")
    st.markdown(con.Cancer_Cell)

def expi(Str,Loc,img,cap):
    my_expander = st.expander(Str, expanded=False)
    with my_expander:
        st.markdown(Loc)
        variable=Image.open(img)
        st.image(variable,width=750,caption=cap)
def Cancer_types():
    st.title("Types of Cancer")
    st.markdown(con.Types_Cancer)
    # my_expander = st.expander("Carcinoma", expanded=True)
    # with my_expander:
    #     st.markdown(con.Carcinoma)
    exp('Carcinoma',con.Carcinoma)
    exp('Sarcoma',con.Sarcoma)
    exp('Leukemia',con.Leukemia)
    exp('Lymphoma',con.Lymphoma)
    exp('Multiple Myeloma',con.MultipleMyeloma)
    exp('Melanoma',con.Melanoma)
def Breast_Cancer():
    st.title("What Is Breast Cancer?")
    st.markdown(con.Breast_Cancer)
    st.markdown(con.Where_Breast_cancer_starts)
    st.markdown(con.types_breast_cancer)
    st.markdown(con.How_breast_cance_spreads)
    variable=Image.open("image..jpg")
    st.image(variable,width=500,caption='Lymph Nodes')
    st.markdown(con.How_breast_cance_spreads2)
def Genetic_Mutation():
    st.subheader('Inheriting certain gene changes')
    st.markdown(con.intro)
    st.subheader("Some of the most aggresive Genes responsible for Breast Cancer")
    exp('BRCA1 and BRCA2',con.gen1)
    exp('ATM',con.gen2)
    exp('TP53',con.gen3)
    exp('CHEK2',con.gen4)
    exp('PTEN',con.gen5)
    exp('CDH1',con.gen6)
    exp('STK11',con.gen7)
    exp('PALB2',con.gen8)
def ClassificationAlgorithm():
    st.title("Classification Algorithms")
    st.markdown(con.classification)
    expi("Logistic Regression",con.LogRegression,"LR.jpg","The Steps of Logistic Regression")
    expi("Support Vector Machine",con.SVM,"SVM.jpg","Support Vector Classifier Graph with Maximum Marginal Hyperplane")
    expi("K Nearest Neighbour",con.KNN,"KNN.jpg","Before and After KNN implementation")
    expi("Kernel SVM",con.KSVM,"Kernel.jpg","Kernel SVM Heigher Dimension Classification")
    expi("Decision Tree",con.DT,"Decision.jpg","Decision Tree Architecture")
    expi("Random Forest",con.RF,"Random.jpg","Random Forest Architecture")
    expi("Naive Bayes",con.NB,"Naive.jpg","Bayes Theorem")
def DatasetInfo():
    st.title("Data Set Information")
    exp("Attributes Info",con.attribte)
    st.dataframe(df)
lis1=[]
def get_data():
    st.header("Data Required For Prediction")
    lis=["Clump Thickess","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]
    for n in range(0,9):
        st.markdown(f"Enter the value for {lis[n]}")
        m=st.slider(f"{lis[n]}",1,10)
        lis.append(m)
    # a=int(st.slider("Clump Thickess",1,10))
    # b=int(st.slider("Uniformity of Cell Size",1,10))
    # c=int(st.slider("Uniformity of Cell Shape",1,10))
    # d=int(st.slider("Marginal Adhesion",1,10))
    # e=int(st.slider("Single Epithelial Cell Size",1,10))
    # f=int(st.slider("Bare Nuclei",1,10))
    # g=int(st.slider("Bland Chromatin",1,10))
    # h=int(st.slider("Normal Nucleoli",1,10))
    # i=int(st.slider("Mitoses",1,10))
    
    return lis
def Model_Selection(lis):
    st.header("The Prediction is in Multiple Classification Algorithms")
    a=lis[0]
    b=lis[1]
    c=lis[2]
    d=lis[3]
    e=lis[4]
    f=lis[5]
    g=lis[6]
    h=lis[7]
    i=lis[8]
    ok=st.button("Predict")
    
    LR_pred=int(LR_Model.predict(sc.transform([[a,b,c,d,e,f,g,h,i]])))
    KNN_pred=int(KNN_Model.predict(sc.transform([[a,b,c,d,e,f,g,h,i]])))
    SVM_pred=int(SVM_Model.predict(sc.transform([[a,b,c,d,e,f,g,h,i]])))
    NB_pred=int(LR.pred(a,b,c,d,e,f,g,h,i))
    KSVM_pred=int(KSVM_Model.predict(sc.transform([[a,b,c,d,e,f,g,h,i]])))
    RF_pred=int(RF_Model.predict(sc.transform([[a,b,c,d,e,f,g,h,i]])))
    DT_pred=int(DT_Model.predict(sc.transform([[a,b,c,d,e,f,g,h,i]])))
    if ok:  
        st.subheader(f"The Logistic Regression Predict that the cell is ___ {'Malignent' if LR_pred == 4 else 'Benign'}")
        st.subheader(f"The KNN Predict that the cell is___ {'Malignent' if KNN_pred == 4 else 'Benign'}")
        st.subheader(f"The SVM Predict that the cell is___ {'Malignent' if SVM_pred == 4 else 'Benign'}")
        st.subheader(f"The Naive Bayes Predict that the cell is___ {'Malignent' if NB_pred == 4 else 'Benign'}")
        st.subheader(f"The KSVM that the cell is___ {'Malignent' if KSVM_pred == 4 else 'Benign'}")
        st.subheader(f'''The Random Forest Predict that the cell is___ {'Malignent' if RF_pred == 4 else 'Benign'}''')
        st.subheader(f"The Decision Tree Predict that the cell is___ {'Malignent' if DT_pred == 4 else 'Benign'}")
        Conclusion()
        