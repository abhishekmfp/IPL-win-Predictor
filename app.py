import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

teams = ['Sunrisers Hyderbad','Mumbai Indians','Royal Challengers Banglore','Kolkata Knight Riders',
          'Punjab Kings', 'Chennai Super Kings','Rajsthan Royals','Delhi Capitals','Gujarat Titans','Lucknow Super Giants']
cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah',  'Mohali', 'Bengaluru']
pipe = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Match  Win Predictor:  ')
col1,col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling Team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')
col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs Completed')
with col5:
    wickets = st.number_input('Wicket out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + " Wining Probability: " + str(round(win * 100)) + "%")
    st.header(bowling_team + " Wining Probability: " + str(round(loss * 100)) + "%")

    # Plotting the prediction probabilities as a pie chart
    fig, ax = plt.subplots()
    ax.pie([loss, win], labels=[bowling_team, batting_team], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)