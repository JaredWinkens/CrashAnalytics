import pandas as pd
import numpy as np

df1 = pd.read_csv("./data/Data_Final.csv")
print(df1.head())
print(len(df1))
for i, col in enumerate(df1.columns):
    print(i,". ", col, df1[col].iloc[0])

print("---------------------------------------")

df2 = pd.read_csv("./data/Statewide_Crashes_2024C.csv")
df2['CrashCategory'] = np.nan
print(df2.head())
print(len(df2))
for i, col in enumerate(df2.columns):
    print(i,". ", col, df2[col].iloc[0])
for i, row in df2.iterrows():
    print(f"Index: {i}, CaseNumber: {row['CaseNumber']}, CrashCategory: {row['CrashCategory']}, CrashType: {row['CrashType']}, LightCondition: {row['LightCondition']}, WeatherCondition: {row['WeatherCondition']}") 
    if row['CrashType'] == 'COLLISION WITH BICYCLIST' or row['CrashType'] == 'COLLISION WITH PEDESTRIAN' or row['CrashType'] == 'COLLISION WITH OTHER PEDESTRIAN':
        row['CrashCategory'] = "VRU"
    else:
        row['CrashCategory'] = "Non-VRU"
    print(f"Index: {i}, CaseNumber: {row['CaseNumber']}, CrashCategory: {row['CrashCategory']}, CrashType: {row['CrashType']}, LightCondition: {row['LightCondition']}, WeatherCondition: {row['WeatherCondition']}") 

print("---------------------------------------")

merged_df = pd.concat([df1, df2], ignore_index=True)
print(merged_df.head())
print(len(merged_df))
for i, col in enumerate(merged_df.columns):
    print(i,". ", col, merged_df[col].iloc[0])

merged_df.to_csv("./data/Data_Final_2024.csv", index=False)