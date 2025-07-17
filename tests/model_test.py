import numpy as np
import mlflow
import pandas as pd

mlflow.set_tracking_uri("https://dagshub.com/pedromonnt/fiap-credit-score-classification-model.mlflow")

model_uri = "models:/credit-score-classification-model/latest"
model = mlflow.pyfunc.load_model(model_uri)

def prepare_data(data):

    data_processed = []

    data_processed.append(int(data["Month"]))
    data_processed.append(int(data["Age"]))
    data_processed.append(int(data["Occupation"]))
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
    data_processed.append(int(data["Credit_Mix"]))
    data_processed.append(float(data["Outstanding_Debt"]))
    data_processed.append(float(data["Credit_Utilization_Ratio"]))
    data_processed.append(int(data["Credit_History_Age"]))
    data_processed.append(int(data["Payment_of_Min_Amount"]))
    data_processed.append(float(data["Total_EMI_per_month"]))
    data_processed.append(float(data["Amount_invested_monthly"]))
    data_processed.append(int(data["Payment_Behaviour"]))
    data_processed.append(float(data["Monthly_Balance"]))

    len(data_processed)

    return data_processed

def range_golden_data(prediction):
    lower = 0
    upper = 2
    return lower <= prediction <= upper


def test_golden_data():

    payload = {
        "Month": "5",
        "Age": "30",
        "Occupation": "6",
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
        "Credit_Mix": "2",
        "Outstanding_Debt": "2000",
        "Credit_Utilization_Ratio": "30",
        "Credit_History_Age": "200",
        "Payment_of_Min_Amount": "2",
        "Total_EMI_per_month": "50",
        "Amount_invested_monthly": "250",
        "Payment_Behaviour": "4",
        "Monthly_Balance": "400"
    }
 
    data_processed = prepare_data(payload)
    data_processed = np.array([data_processed])

    df_input = pd.DataFrame(data_processed, columns=payload.keys())
    
    # Colunas que o modelo espera como inteiros (long)
    int_columns = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

    # Converte as colunas para o tipo int64
    for col in int_columns:
        df_input[col] = df_input[col].astype(np.int64)

    result = model.predict(df_input)

    result = int(result[0])

    print(f"Prediction: {result}")

    assert range_golden_data(result), "ensuring golden data range for prediction"

def test_model_load_call():

    payload = {
        "Month": "5",
        "Age": "30",
        "Occupation": "6",
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
        "Credit_Mix": "2",
        "Outstanding_Debt": "2000",
        "Credit_Utilization_Ratio": "30",
        "Credit_History_Age": "200",
        "Payment_of_Min_Amount": "2",
        "Total_EMI_per_month": "50",
        "Amount_invested_monthly": "250",
        "Payment_Behaviour": "4",
        "Monthly_Balance": "400"
    }

    data_processed = prepare_data(payload)
    data_processed = np.array([data_processed])

    df_input = pd.DataFrame(data_processed, columns=payload.keys())

    # Colunas que o modelo espera como inteiros (long)
    int_columns = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

    # Converte as colunas para o tipo int64
    for col in int_columns:
        df_input[col] = df_input[col].astype(np.int64)

    result = model.predict(df_input)

    result = int(result[0])

    assert isinstance(result, int), "ensuring model prediction returns an integer"
    assert result >= 0, "ensuring model prediction is greater than or equal to zero"

test_golden_data()
test_model_load_call()