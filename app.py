import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import urllib.request
import requests

st.set_page_config(page_title='Churn Dashboard', page_icon='📊', layout='wide')

# load the telco dataset from github
def load_data():
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    raw = urllib.request.urlopen(url)
    # tried reading local csv first but kept getting path errors
    # so just loading directly from github - easier this way
    df = pd.read_csv(io.StringIO(raw.read().decode('utf-8')))
    
    # totalcharges was stored as text - had to convert it
    # found this during EDA when df.info() showed object type
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()

# i created this risk scoring logic based on EDA findings
# month to month customers had 42% churn - highest of all groups
# new customers (under 12 months) also churn a lot
# high charges add some extra risk too
def get_risk(row):
    score = 0
    # first i tried using just tenure but accuracy was low
    # adding contract type made more sense after seeing the EDA results
    if row['Contract'] == 'Month-to-month':
        score += 2
    
    if row['tenure'] < 12:
        score += 2
    
    if row['MonthlyCharges'] > 65:
        score += 1

    if score >= 4:
        return 'High Risk'
    elif score >= 2:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['RiskTier'] = df.apply(get_risk, axis=1)


# sidebar filters
st.sidebar.title('Filters')
st.sidebar.markdown('---')

contracts = st.sidebar.multiselect(
    'Contract Type',
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)

internet = st.sidebar.multiselect(
    'Internet Service',
    options=df['InternetService'].unique(),
    default=df['InternetService'].unique()
)

risk_filter = st.sidebar.multiselect(
    'Risk Tier',
    options=['High Risk', 'Medium Risk', 'Low Risk'],
    default=['High Risk', 'Medium Risk', 'Low Risk']
)

st.sidebar.markdown('---')
st.sidebar.markdown('**Dataset:** IBM Telco Churn')
st.sidebar.markdown('**Model:** Logistic Regression')
st.sidebar.markdown('**Accuracy:** 78.7%')
st.sidebar.markdown('**Train/Test Split:** 80/20')

# filter the dataframe based on sidebar selections
filtered_df = df[
    (df['Contract'].isin(contracts)) &
    (df['InternetService'].isin(internet)) &
    (df['RiskTier'].isin(risk_filter))
]

# show warning if too few rows after filtering
if len(filtered_df) < 50:
    st.sidebar.warning('Very few customers in this filter - results may not be accurate')


# main dashboard
st.title('Customer Churn Prediction Dashboard')
st.caption('IBM Telco Dataset | Logistic Regression | Accuracy: 78.7%')
st.markdown('---')

# summary numbers at the top
total = len(filtered_df)
churn_pct = round(filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100, 1)
# took me a while to figure out normalize=True gives percentage
# earlier i was doing manual division which was messier
rev_risk = int(filtered_df[filtered_df['Churn'] == 'Yes']['MonthlyCharges'].sum())
high_risk = len(filtered_df[filtered_df['RiskTier'] == 'High Risk'])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Churn Rate", f"{churn_pct}%")
c3.metric("Monthly Revenue at Risk", f"Rs.{rev_risk:,}")
c4.metric("High Risk Customers", f"{high_risk:,}")

st.markdown('---')


# charts - row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader('Churned vs Retained')
    fig, ax = plt.subplots(figsize=(5, 3))
    filtered_df['Churn'].value_counts().plot(
        kind='bar', ax=ax,
        color=['steelblue', 'tomato']
    )
    ax.set_xticklabels(['Retained', 'Churned'], rotation=0)
    ax.set_ylabel('Number of Customers')
    ax.set_title(f'Churn rate: {churn_pct}%', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

with col2:
    st.subheader('Churn Rate by Contract Type')
    # this was one of the biggest findings - month to month is way higher
    contract_churn = (
        filtered_df.groupby('Contract')['Churn']
        .apply(lambda x: round((x == 'Yes').mean() * 100, 1))
        .reset_index()
    )
    contract_churn.columns = ['Contract', 'Churn Rate (%)']
    contract_churn = contract_churn.sort_values('Churn Rate (%)')

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(
        contract_churn['Contract'],
        contract_churn['Churn Rate (%)'],
        color=['steelblue', 'orange', 'tomato']
    )
    ax.set_xlabel('Churn Rate (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, contract_churn['Churn Rate (%)']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center')
    st.pyplot(fig)

st.markdown('---')

# charts - row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader('Monthly Charges vs Churn')
    # higher charges = more churn - confirmed in EDA
    fig, ax = plt.subplots(figsize=(5, 3))
    for grp, clr, lbl in zip(['No', 'Yes'], ['steelblue', 'tomato'], ['Retained', 'Churned']):
        ax.hist(
            filtered_df[filtered_df['Churn'] == grp]['MonthlyCharges'],
            bins=30, alpha=0.6, color=clr, label=lbl
        )
    ax.set_xlabel('Monthly Charges')
    ax.set_ylabel('Customers')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

with col2:
    st.subheader('Churn by Tenure Group')
    # new customers churn the most - first 12 months is the critical period
    filtered_df = filtered_df.copy()
    filtered_df['Tenure Group'] = pd.cut(
        filtered_df['tenure'],
        bins=[0, 12, 36, 72],
        labels=['New (0-12m)', 'Mid (12-36m)', 'Loyal (36-72m)']
    )
    tenure_churn = (
        filtered_df.groupby('Tenure Group', observed=True)['Churn']
        .apply(lambda x: round((x == 'Yes').mean() * 100, 1))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(
        tenure_churn['Tenure Group'],
        tenure_churn['Churn'],
        color=['tomato', 'orange', 'steelblue']
    )
    ax.set_ylabel('Churn Rate (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, tenure_churn['Churn']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val}%', ha='center')
    st.pyplot(fig)

st.markdown('---')

# internet service churn chart
st.subheader('Churn by Internet Service')
internet_churn = (
    filtered_df.groupby('InternetService')['Churn']
    .apply(lambda x: round((x == 'Yes').mean() * 100, 1))
    .reset_index()
)
internet_churn.columns = ['Internet Service', 'Churn Rate (%)']

fig, ax = plt.subplots(figsize=(7, 3))
bars = ax.bar(
    internet_churn['Internet Service'],
    internet_churn['Churn Rate (%)'],
    color=['tomato', 'orange', 'steelblue'], width=0.4
)
ax.set_ylabel('Churn Rate (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, internet_churn['Churn Rate (%)']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val}%', ha='center')
st.pyplot(fig)

st.markdown('---')

# risk tier table
st.subheader('Risk Tier Summary')

risk_summary = (
    filtered_df.groupby('RiskTier')
    .agg(
        Customers=('RiskTier', 'count'),
        Revenue_at_Risk=('MonthlyCharges', 'sum')
    )
    .reset_index()
    .sort_values('Customers', ascending=False)
)
risk_summary['Revenue_at_Risk'] = risk_summary['Revenue_at_Risk'].astype(int)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(risk_summary, use_container_width=True, hide_index=True)

with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = {'High Risk': 'tomato', 'Medium Risk': 'orange', 'Low Risk': 'steelblue'}
    ax.bar(
        risk_summary['RiskTier'],
        risk_summary['Customers'],
        color=[colors.get(r, 'gray') for r in risk_summary['RiskTier']]
    )
    ax.set_ylabel('Customers')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

st.markdown('---')

# AI risk explainer section
st.subheader('AI Customer Risk Explainer')
st.caption('Enter customer details to get a risk explanation')

#GROQ segment -API
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

col1, col2, col3 = st.columns(3)
with col1:
    cust_tenure = st.slider('Tenure (months)', 0, 72, 12)
with col2:
    cust_charges = st.slider('Monthly Charges', 18, 120, 65)
with col3:
    cust_contract = st.selectbox('Contract Type',
        ['Month-to-month', 'One year', 'Two year'])

# quick risk estimate using same logic as the dataset scoring
pts = 0
if cust_contract == 'Month-to-month': pts += 2
if cust_tenure < 12: pts += 2
if cust_charges > 65: pts += 1

if pts >= 4: label = '🔴 High Risk'
elif pts >= 2: label = '🟡 Medium Risk'
else: label = '🟢 Low Risk'

st.markdown(f"**Risk Estimate: {label}**")

if st.button('Get AI Explanation'):
    # at first, forgot to make it short and to the point
    # then converted it to better format 
    prompt = f"""You are a telecom churn analyst.
A customer has been with us {cust_tenure} months,
pays Rs.{cust_charges} monthly, on a {cust_contract} contract.

Give:
- 3 bullet points on their churn risk
- 1 retention action to take

Keep it short and direct."""

    with st.spinner('Analysing customer...'):
        try:
            resp = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                # first tried gemini api but kept getting 429 errors so switched to groq
                # claude helped a lot to understand the errors especially one which restricted me to use api
                headers={
                    'Authorization': f'Bearer {GROQ_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'llama-3.3-70b-versatile',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 250
                },
                timeout=10
            )

            if resp.status_code == 200:
                st.info(resp.json()['choices'][0]['message']['content'])
            elif resp.status_code == 429:
                st.warning('Rate limit hit - wait a few seconds and try again')
            else:
                st.error(f'Something went wrong - status {resp.status_code}')

        except Exception as e:
            st.error(f'Error: {str(e)}')

st.markdown('---')
# just a small appreciation for all the efforts 
st.caption('Customer Churn Dashboard | Pratham Monga | 2026')