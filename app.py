from flask import Flask, render_template, request
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

from src.recommendation_system import match

def followers_to_number(followers_str):
    try:
        return float(followers_str.replace("Followers","").replace('K', 'e3').replace('M', 'e6').replace('B', 'e9'))
    except:
        return np.NaN

def service_mapping(service):
    mapping = {
        "blockchain_infrastructure": "Service_Blockchain Infrastructure",
        "blockchain_service": "Service_Blockchain Service",
        "cefi": "Service_CeFi",
        "chain": "Service_Chain",
        "defi": "Service_DeFi",
        "gamefi": "Service_GameFi",
        "social": "Service_Social",
        "stablecoin": "Service_Stablecoin"
    }
    return mapping.get(service, "")

column_weights = {
    'Total Raised': 0.1,
    'First Funding Year': 1,
    'First Funding Month': 1,
    'First Funding Day': 1,
    'Funding Round_Angel': 0.1,
    'Funding Round_Pre-Seed': 0.1,
    'Funding Round_Pre-Series A': 0.1,
    'Funding Round_Seed': 0.1,
    'Funding Round_Series A': 0.1,
    'Funding Round_Strategic': 0.1,
    'Funding Round_Undisclosed': 0.1
}


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            amount_raised = 0
            data = pd.read_csv("Data/data.csv")
            data.fillna(0, inplace=True)
            data.drop_duplicates(inplace=True)
            description = request.form['description']
            no_of_recommendation = request.form['no_of_recommendation']
            service = request.form['service']
            funding_round = request.form['funding_round']
            amount_raised = request.form['amount_raised']
            twitter_followers = request.form['twitter_followers']

            if amount_raised:
                min_value = data['Total Raised'].min()
                max_value = data['Total Raised'].max()
                amount_raised = (int(amount_raised) - min_value)/(max_value - min_value)

                column_weights['Total Raised'] = 0.01

            column_weights['Funding Round_Angel'] =0.01 if funding_round=="angel" else 0.1
            column_weights['Funding Round_Pre-Seed'] =0.01 if funding_round=="pre_seed" else 0.1
            column_weights['Funding Round_Pre-Series A'] =0.01 if funding_round=="pre_series_a" else 0.1
            column_weights['Funding Round_Seed'] =0.01 if funding_round=="seed" else 0.1
            column_weights['Funding Round_Series A'] =0.01 if funding_round=="series_a" else 0.1
            column_weights['Funding Round_Strategic'] =0.01 if funding_round=="strategic" else 0.1
            column_weights['Funding Round_Undisclosed'] =0.01 if funding_round=="undisclosed" else 0.1

            # Perform some processing on the input text (e.g., sentiment analysis)
            # Replace this with your actual processing code

            # if service:
            #     service = service_mapping(service)
            #     data = data[data['Service']==service.replace("Service_","")]


            processed_data = data.drop(columns=["Name", "Raised Amount", "First Funding Date", "Valuation Amount", "Links"])

            processed_data.fillna(0, inplace=True)
            processed_data = pd.get_dummies(processed_data, columns= ["Service", "Funding Round"], dtype = int)
            processed_data["Inverstors_and_desc"] = processed_data["Investors"].astype(str)+" "+processed_data["Description"].astype(str)
            processed_data = processed_data.drop(columns=["Investors","Description"])
            # Initialize the MinMaxScaler
            scaler = MinMaxScaler()

            # List of numeric column names
            numeric_columns = ['Total Raised', 'First Funding Year', 'First Funding Month',
                'First Funding Day', 'Service_Blockchain Infrastructure',
                'Service_Blockchain Service', 'Service_CeFi', 'Service_Chain',
                'Service_DeFi', 'Service_GameFi', 'Service_Social',
                'Service_Stablecoin', 'Funding Round_Angel', 'Funding Round_Pre-Seed',
                'Funding Round_Pre-Series A', 'Funding Round_Seed',
                'Funding Round_Series A', 'Funding Round_Strategic',
                'Funding Round_Undisclosed']

            # Apply min-max normalization to each numeric column
            processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
            if service:
                service = service_mapping(service)
                processed_data = processed_data[processed_data[service]==1]
            
            if twitter_followers:
                twitter_followers_df = pd.read_csv("Data/Twitter_followers.csv")
                twitter_followers_df['followers_num'] = twitter_followers_df["followers"].apply(followers_to_number)
                if twitter_followers == 1:
                    twitter_followers_df = twitter_followers_df[twitter_followers_df['followers_num']<100000]
                elif twitter_followers == 2:
                    twitter_followers_df = twitter_followers_df[twitter_followers_df['followers_num']<200000]
                elif twitter_followers == 3:
                    twitter_followers_df = twitter_followers_df[twitter_followers_df['followers_num']<500000]
                elif twitter_followers == 4:
                    twitter_followers_df = twitter_followers_df[twitter_followers_df['followers_num']>=500000]

                processed_data = processed_data[
                    processed_data["Crypto Name"].isin(twitter_followers_df["Crypto Name"])
                ]

            processed_data = processed_data.drop(columns=["Crypto Name"])
            top_indices = match(processed_data=processed_data,
                                top_n=no_of_recommendation,
                                column_weights=column_weights,
                                description=description,
                                amount_raised=amount_raised
                                # service=service,
                                # funding_round=funding_round
                                )

            # result = data.iloc[top_indices].values.tolist()
            result_data = data.iloc[top_indices]
            result_ = result_data.merge(twitter_followers_df[['Crypto Name', 'followers']], on='Crypto Name', how='left', suffixes=('', '_df2'))
            result = result_.values.tolist()
            with open("log.txt", "a") as f:
                f.write(str(result))
        except Exception as e:
            with open("log.txt", "a") as f:
                f.write(str(e) + "\n")
            pass
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

