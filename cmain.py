import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import ExtraTreeRegressor
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report



df = pd.read_csv('cds.csv')
# dealing with data in wrong format,for categorical variables, this step is ignored
df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
df['country'] = pd.to_numeric(df['country'], errors='coerce')
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['material_ref'] = df['material_ref'].str.lstrip('0')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')

df['material_ref'].fillna('unknown', inplace=True)
df = df.dropna()
df['quantity tons'] = df['quantity tons'].apply(lambda x: np.nan if x<=0 else x)
df['selling_price'] = df['selling_price'].apply(lambda x: np.nan if x<=0 else x)
df = df.dropna()
df1 = df.copy()
#def plot(df, column):
   # plt.figure(figsize=(20,5))
   # plt.subplot(1,3,1)
   # sns.boxplot(data=df, x=column)
   # plt.title(f'Box Plot for {column}')

   # plt.subplot(1,3,2)
   # sns.histplot(data=df, x=column, kde=True, bins=50)
   # plt.title(f'Distribution Plot for {column}')

    #plt.subplot(1,3,3)
   # sns.violinplot(data=df, x=column)
  #  plt.title(f'Violin Plot for {column}')
 #   plt.show()
#for i in ['quantity tons', 'thickness', 'width', 'selling_price']:
#    plot(df1, i)


df1['quantity tons_log'] = np.log(df1['quantity tons'])
df1['thickness_log'] = np.log(df1['thickness'])
df1['selling_price_log'] = np.log(df1['selling_price'])
#for i in ['quantity tons_log', 'thickness_log', 'width', 'selling_price_log']:
#    plot(df1, i)
OE = OrdinalEncoder()
df1['status_en'] = OE.fit_transform(df1[['status']])
df1['item type_en'] = OE.fit_transform(df1[['item type']])
item_type_mapping = pd.DataFrame({
    'item type': df1['item type'].unique(),
    'item type_en': df1['item type_en'].unique()
})

status_mapping = pd.DataFrame({
    'status': df1['status'].unique(),
    'status_en': df1['status_en'].unique()
})

X=df1[['quantity tons_log','status_en','item type_en','application','thickness_log','width','country','customer','product_ref']]
y=df1['selling_price_log']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)


lr = LinearRegression()
lr.fit(X_train, y_train)
#print(lr.score(X_train, y_train))

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print('Mean squared error:', mse)
#print('R-squared:', r2)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print('Mean squared error:', mse)
#print('R-squared:', r2)

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print('Mean squared error:', mse)
#print('R-squared:', r2)

etr = ExtraTreeRegressor()
etr.fit(X_train, y_train)
y_pred = etr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print('Mean squared error:', mse)
#print('R-squared:', r2)

new_quantity_tons_log = np.log(40)
status_en = 5.0
item_type_en = 4.0
application = 20.0
new_thickness_log = np.log(250)
new_width = 28
country = 25.0
new_customer = 30202938
new_product_ref = 1670798778
new_sample = np.array([[new_quantity_tons_log, status_en, item_type_en, application,
                         new_thickness_log, new_width, country, new_customer, new_product_ref]])
new_pred = rf.predict(new_sample)[0]

# Regression
with open('model.pkl', 'wb') as file:
    pickle.dump(rf, file)


df2 = df1.copy()
df3 = df2[df2['status'].isin(['Won', 'Lost'])]
df3["status_encoded"] = df3['status'].map({"Won":1, "Lost":0})
df4 = df3[['quantity tons_log','selling_price_log','item type_en', 'application','thickness_log',
           'width','country','customer','product_ref','status_encoded']]

X = df4.drop(["status_encoded"], axis=1)
y = df4["status_encoded"]


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.3, random_state = 5)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

result = confusion_matrix(y_test, y_pred)
result1 = classification_report(y_test, y_pred)


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)

# Define the values for the new sample
new_quantity_tons_log = np.log(40)
selling_price_log = np.log(50)
item_type_en = 4.0
application = 20.0
new_thickness_log = np.log(250)
new_width = 1500.0
country = 25.0
new_customer = 30202938
new_product_ref = 1670798778

# Create the new sample as a numpy array
new_sample = np.array([[new_quantity_tons_log, selling_price_log, item_type_en, application,
                         new_thickness_log, new_width, country, new_customer, new_product_ref]])

# Make predictions using the trained RandomForestRegressor model
new_pred = rfc.predict(new_sample)

if new_pred==1:
    print('The status is: Won')
else:
    print('The status is: Lost')

# Saving the model
import pickle
with open('classfier_model.pkl', 'wb') as file:
    pickle.dump(rfc, file)



