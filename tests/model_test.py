import numpy as np
import mlflow
import pandas as pd

mlflow.set_tracking_uri("https://dagshub.com/pedromonnt/fiap-credit-score-classification-model.mlflow")

model_uri = "models:/credit-score-classification-model/latest"
model = mlflow.pyfunc.load_model(model_uri)

def prepare_data(data):

    data_processed = []

    data_processed.append(int(data["Age"]))
    data_processed.append(float(data["Annual_Income"]))
    data_processed.append(float(data["Monthly_Inhand_Salary"]))  
    data_processed.append(int(data["Num_Bank_Accounts"]))
    data_processed.append(int(data["Num_Credit_Card"]))
    data_processed.append(int(data["Interest_Rate"]))
    data_processed.append(int(data["Num_of_Loan"]))
    data_processed.append(int(data["Delay_from_due_date"]))
    data_processed.append(int(data["Num_of_Delayed_Payment"]))
    data_processed.append(float(data["Changed_Credit_Limit"]))
    data_processed.append(int(data["Num_Credit_Inquiries"]))
    data_processed.append(float(data["Outstanding_Debt"]))
    data_processed.append(float(data["Credit_Utilization_Ratio"]))
    data_processed.append(int(data["Credit_History_Age"]))
    data_processed.append(float(data["Total_EMI_per_month"]))
    data_processed.append(float(data["Amount_invested_monthly"]))   
    data_processed.append(float(data["Monthly_Balance"]))

    # Listas de valores possíveis para one-hot encoding
    occupations = [
        "Accountant", "Architect", "Desconhecido", "Developer", "Doctor",
        "Engineer", "Entrepreneur", "Journalist", "Lawyer", "Manager",
        "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"
    ]
    credit_mix_values = ["Bad", "Desconhecido", "Good", "Standard"]
    payment_min_amount_values = ["NM", "No", "Yes"]
    payment_behaviour_values = [
        "Desconhecido", "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"
    ]

    # One-hot encoding para 'Occupation'
    for occ in occupations:
        data_processed.append(1 if data["Occupation"] == occ else 0)

    # One-hot encoding para 'Credit_Mix'
    for mix in credit_mix_values:
        data_processed.append(1 if data["Credit_Mix"] == mix else 0)

    # One-hot encoding para 'Payment_of_Min_Amount'
    for pma in payment_min_amount_values:
        data_processed.append(1 if data["Payment_of_Min_Amount"] == pma else 0)

    # One-hot encoding para 'Payment_Behaviour'
    for pb in payment_behaviour_values:
        data_processed.append(1 if data["Payment_Behaviour"] == pb else 0)

    len(data_processed)

    return data_processed

def range_golden_data(prediction):
    lower = 0
    upper = 2
    return lower <= prediction <= upper


def test_golden_data():

    payload = {
        "Age": "40",
        "Annual_Income": "100000",
        "Monthly_Inhand_Salary": "8000",
        "Num_Bank_Accounts": "2",
        "Num_Credit_Card": "4",
        "Interest_Rate": "10",
        "Num_of_Loan": "2",
        "Delay_from_due_date": "4",
        "Num_of_Delayed_Payment": "11",
        "Changed_Credit_Limit": "6",
        "Num_Credit_Inquiries": "3",
        "Outstanding_Debt": "2000",
        "Credit_Utilization_Ratio": "30",
        "Credit_History_Age": "200",
        "Total_EMI_per_month": "50",
        "Amount_invested_monthly": "250",
        "Monthly_Balance": "400",
        "Occupation": "Doctor",
        "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "No",
        "Payment_Behaviour": "High_spent_Large_value_payments"
    }
 
    data_processed = prepare_data(payload)
    data_processed = np.array([data_processed])

    columns = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Occupation_Accountant", "Occupation_Architect", "Occupation_Desconhecido", "Occupation_Developer",
        "Occupation_Doctor", "Occupation_Engineer", "Occupation_Entrepreneur", "Occupation_Journalist",
        "Occupation_Lawyer", "Occupation_Manager", "Occupation_Mechanic", "Occupation_MediaManager", 
        "Occupation_Musician", "Occupation_Scientist", "Occupation_Teacher", "Occupation_Writer",
        "Credit_Mix_Bad", "Credit_Mix_Desconhecido", "Credit_Mix_Good", "Credit_Mix_Standard",
        "Payment_of_Min_Amount_NM", "Payment_of_Min_Amount_No", "Payment_of_Min_Amount_Yes",
        "Payment_Behaviour_Desconhecido", "Payment_Behaviour_High_spent_Large_value_payments",
        "Payment_Behaviour_High_spent_Medium_value_payments", "Payment_Behaviour_High_spent_Small_value_payments",
        "Payment_Behaviour_Low_spent_Large_value_payments", "Payment_Behaviour_Low_spent_Medium_value_payments",
        "Payment_Behaviour_Low_spent_Small_value_payments"
    ]

    int_columns = [
        "Occupation_Accountant", "Occupation_Architect", "Occupation_Desconhecido", "Occupation_Developer",
        "Occupation_Doctor", "Occupation_Engineer", "Occupation_Entrepreneur", "Occupation_Journalist",
        "Occupation_Lawyer", "Occupation_Manager", "Occupation_Mechanic", "Occupation_MediaManager", 
        "Occupation_Musician", "Occupation_Scientist", "Occupation_Teacher", "Occupation_Writer",
        "Credit_Mix_Bad", "Credit_Mix_Desconhecido", "Credit_Mix_Good", "Credit_Mix_Standard",
        "Payment_of_Min_Amount_NM", "Payment_of_Min_Amount_No", "Payment_of_Min_Amount_Yes",
        "Payment_Behaviour_Desconhecido", "Payment_Behaviour_High_spent_Large_value_payments",
        "Payment_Behaviour_High_spent_Medium_value_payments", "Payment_Behaviour_High_spent_Small_value_payments",
        "Payment_Behaviour_Low_spent_Large_value_payments", "Payment_Behaviour_Low_spent_Medium_value_payments",
        "Payment_Behaviour_Low_spent_Small_value_payments"
    ]

    df_input = pd.DataFrame(data_processed, columns=columns)
 
    for col in int_columns:
        df_input[col] = int(df_input[col].iloc[0])
    
    result = model.predict(df_input)

    result = int(result[0])

    print(f"Prediction: {result}")

    assert range_golden_data(result), "ensuring golden data range for prediction"

def test_model_load_call():

    payload = {
        "Age": "40",
        "Annual_Income": "100000",
        "Monthly_Inhand_Salary": "8000",
        "Num_Bank_Accounts": "2",
        "Num_Credit_Card": "4",
        "Interest_Rate": "10",
        "Num_of_Loan": "2",
        "Delay_from_due_date": "4",
        "Num_of_Delayed_Payment": "11",
        "Changed_Credit_Limit": "6",
        "Num_Credit_Inquiries": "3",
        "Outstanding_Debt": "2000",
        "Credit_Utilization_Ratio": "30",
        "Credit_History_Age": "200",
        "Total_EMI_per_month": "50",
        "Amount_invested_monthly": "250",
        "Monthly_Balance": "400",
        "Occupation": "Doctor",
        "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "No",
        "Payment_Behaviour": "High_spent_Large_value_payments"
    }

    data_processed = prepare_data(payload)
    data_processed = np.array([data_processed])

    columns = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Occupation_Accountant", "Occupation_Architect", "Occupation_Desconhecido", "Occupation_Developer",
        "Occupation_Doctor", "Occupation_Engineer", "Occupation_Entrepreneur", "Occupation_Journalist",
        "Occupation_Lawyer", "Occupation_Manager", "Occupation_Mechanic", "Occupation_MediaManager", 
        "Occupation_Musician", "Occupation_Scientist", "Occupation_Teacher", "Occupation_Writer",
        "Credit_Mix_Bad", "Credit_Mix_Desconhecido", "Credit_Mix_Good", "Credit_Mix_Standard",
        "Payment_of_Min_Amount_NM", "Payment_of_Min_Amount_No", "Payment_of_Min_Amount_Yes",
        "Payment_Behaviour_Desconhecido", "Payment_Behaviour_High_spent_Large_value_payments",
        "Payment_Behaviour_High_spent_Medium_value_payments", "Payment_Behaviour_High_spent_Small_value_payments",
        "Payment_Behaviour_Low_spent_Large_value_payments", "Payment_Behaviour_Low_spent_Medium_value_payments",
        "Payment_Behaviour_Low_spent_Small_value_payments"
    ]

    int_columns = [
        "Occupation_Accountant", "Occupation_Architect", "Occupation_Desconhecido", "Occupation_Developer",
        "Occupation_Doctor", "Occupation_Engineer", "Occupation_Entrepreneur", "Occupation_Journalist",
        "Occupation_Lawyer", "Occupation_Manager", "Occupation_Mechanic", "Occupation_MediaManager", 
        "Occupation_Musician", "Occupation_Scientist", "Occupation_Teacher", "Occupation_Writer",
        "Credit_Mix_Bad", "Credit_Mix_Desconhecido", "Credit_Mix_Good", "Credit_Mix_Standard",
        "Payment_of_Min_Amount_NM", "Payment_of_Min_Amount_No", "Payment_of_Min_Amount_Yes",
        "Payment_Behaviour_Desconhecido", "Payment_Behaviour_High_spent_Large_value_payments",
        "Payment_Behaviour_High_spent_Medium_value_payments", "Payment_Behaviour_High_spent_Small_value_payments",
        "Payment_Behaviour_Low_spent_Large_value_payments", "Payment_Behaviour_Low_spent_Medium_value_payments",
        "Payment_Behaviour_Low_spent_Small_value_payments"
    ]

    df_input = pd.DataFrame(data_processed, columns=columns)

    for col in int_columns:
        df_input[col] = int(df_input[col].iloc[0])

    result = model.predict(df_input)

    result = int(result[0])

    assert isinstance(result, int), "ensuring model prediction returns an integer"
    assert result >= 0, "ensuring model prediction is greater than or equal to zero"

test_golden_data()
test_model_load_call()