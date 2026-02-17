import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphviz as graphviz

st.title("Prueba")
st.write("Hello ,let's learn how to build a streamlit app together")




#st.title("This is the app title")
#st.header("This is the header")
#st.markdown("This is the markdown")
#st.subheader("This is the subheader")
#st.caption("This is the caption")
#st.code("x = 2021")
#st.latex(r''' a+a r^1+a r^2+a r^3 ''')




st.image("fp.jpeg", caption="Duko")
#st.audio("audio.mp3")
#st.video("video.mp4")




st.checkbox('Yes')
st.button('Click Me')
st.radio('Pick your gender', ['Male', 'Female'])
st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0, 50)



#st.progress(10)  
#with st.spinner('Wait for it...'):
#    time.sleep(10)


st.sidebar.title("Sidebar Title")
st.sidebar.markdown("This is the sidebar content")





#container = st.container()
#
#container.write("hi")



#with st.container():    
#    st.write("This is inside the container")
#    st.write("other text")
#
#st.write("fuera")


rand= np.random.normal(1,2, size=20)
fig, ax = plt.subplots()
ax.hist(rand,bins=15, color="purple")
st.pyplot(fig)



df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.line_chart(df)

df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.bar_chart(df)





st.graphviz_chart('''    digraph {        Big_shark -> Tuna        Tuna -> Mackerel        Mackerel -> Small_fishes        Small_fishes -> Shrimp    }''')








df = pd.DataFrame(    np.random.randn(500, 2) / [50, 50] + [37.76, -122.4], columns=['lat', 'lon'])
st.map(df)