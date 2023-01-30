
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

st.title('Real-Time Blood Volume Pulse')



# eda=pd.read_csv('EDA.csv')
# eda_plot=eda.iloc[:,0:1]
# eda_plot=eda[1000:3000]

bvp=pd.read_csv('BVP.csv')
bvp_plot=bvp.iloc[:,0:1]
bvp_plot=bvp
bvp_plot_30=bvp[:2000]

# temp=pd.read_csv('TEMP.csv')
# temp_plot=temp.iloc[:,0:1]
# temp_plot=temp[1000:3000]

# hr=pd.read_csv('HR.csv')
# hr_plot=hr.iloc[:,0:1]
# hr_plot=hr_plot[1000:3000]

# st.subheader('Electrodermal Activity (EDA)')
# st.line_chart(eda_plot)

# st.subheader('Blood Volume Pulse (BVP)')
# st.line_chart(bvp_plot)

# st.subheader('Skin Temperature (ST)')
# st.line_chart(temp_plot)

# st.subheader('Heart Rate (HR)')
# st.line_chart(hr_plot)


# eda_plot=np.asarray(eda_plot)
# hr_plot=np.asarray(hr_plot)
# temp_plot=np.asarray(temp_plot)
bvp_plot=np.asarray(bvp_plot)

# eda_df=pd.DataFrame.from_records(eda_plot)
# hr_df=pd.DataFrame.from_records(hr_plot)
# temp_df=pd.DataFrame.from_records(temp_plot)
bvp_df=pd.DataFrame.from_records(bvp_plot)

# dataframe_signals=pd.concat([eda_df,hr_df,temp_df,bvp_df],axis=1)
# dataframe_signals.columns=['EDA','HR','TEMP','BVP']

dataframe_signals=pd.concat([bvp_df],axis=1)
dataframe_signals.columns=['BVP']

bvp_plot_30=np.asarray(bvp_plot_30)
bvp_plot_30_df=pd.DataFrame.from_records(bvp_plot_30)
dataframe_signals_30=pd.concat([bvp_plot_30_df],axis=1)
dataframe_signals_30.columns=['BVP']

# hr_plot=list(hr_plot)
# eda_plot=list(eda_plot)
# temp_plot=list(temp_plot)
# bvp_plot=list(bvp_plot)



# dataframe_signals=pd.DataFrame({
#     'HR':hr_plot,
#     'EDA':eda_plot,
#     'TEMP':temp_plot,
#     'BVP':bvp_plot})



# st.subheader('Your Real-Time Blood Volume Pulse')
dataframe_signals.plot()
plt.show()
st.pyplot()
st.subheader('30 second BVP')
dataframe_signals_30.plot()
plt.show()
st.pyplot()

# st.subheader('Statistics of Important Vitals')
# st.write('Mean Heart Rate:'+ str(np.mean(dataframe_signals['HR'])))
# st.write('Max Heart Rate:'+ str(np.max(dataframe_signals['HR'])))
# st.write('Min Heart Rate:'+ str(np.min(dataframe_signals['HR'])))

# st.write('Mean Skin Temperature:'+ str(np.mean(dataframe_signals['TEMP'])))
# st.write('Max Skin Temperature:'+ str(np.max(dataframe_signals['TEMP'])))
# st.write('Min Skin Temperature:'+ str(np.min(dataframe_signals['TEMP'])))

