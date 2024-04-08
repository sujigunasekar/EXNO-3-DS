## EXNO-3-DS

### AIM:

To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

### ALGORITHM:

STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set. 

STEP 5:Save the data to the file.

### FEATURE ENCODING:
1.Ordinal Encoding An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2.Label Encoding Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3.Binary Encoding Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4.One Hot Encoding We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

### Methods Used for Data Transformation:
1. FUNCTION TRANSFORMATION
• Log Transformation • Reciprocal Transformation • Square Root Transformation • Square Transformation

2. POWER TRANSFORMATION
• Boxcox method • Yeojohnson method

### CODING AND OUTPUT:
Developed by : Suji G
Reg No : 212222230152
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![318691387-9a445ed3-f79e-46ed-8493-a0138abde135](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/f08766a0-1d45-4fc7-ab6a-bfe63522eca4)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![318692227-c5ae2314-6f2b-4d93-92b3-f44d1b74015a](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/47a7c50b-b3c2-4fad-8b1e-e412b4193ff6)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![318692322-4ae17d2a-aa22-4340-9faf-8567549250f6](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/b2af05d8-7b6d-4ba7-9d2a-e91cfadde14c)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![318692437-2249ccf3-4a16-462b-b745-677312c7fd42](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/0f2a8630-d9a1-4edd-be5e-7ec7fe2b40e2)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![318692763-d2714505-ceae-48c6-b428-fc421aaa735d](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/6f69f914-933f-43c1-9224-11df0ec7f03f)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![318692827-b4b4c5b2-9bc8-4f41-8649-096999696847](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/39517d83-8325-4fa5-9316-38c257fbe4cc)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![318692921-e56e11b0-9489-41a5-973c-e32fca8f9840](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/1bf885d6-2cce-44aa-be8d-40fa4c805e55)
```
pip install --upgrade category_encoders
```
![318693032-0711d42f-4456-4222-8334-f183bc7c2385](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/6f6f01f1-b96d-4273-a4f0-0a00b59339b9)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![318693230-3d2f8b4c-0ffc-4754-8c1b-ad637c727c9b](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/68245164-297f-423a-9826-7a9b57367e15)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![318897767-781ddd71-1fc6-499b-9234-b83778405580](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/0125908a-e86b-4d52-b36a-25d9e20a9696)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![318897871-6f1877a4-9ba9-45d6-8df2-38fdc103a0ef](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/fc41441d-ed8c-4d75-bbf0-321dd235902e)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![318897982-63cbb12a-e9eb-447e-855a-e56c706bbfa9](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/03457385-58fc-4283-a6bd-adf987a2adac)
```
df.skew()
```
![318898092-3d04bbce-76dc-4571-8c8d-5aad234c1766](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/45850f05-7d1e-4294-8dc0-89581ef3e29b)
```
np.log(df["Highly Positive Skew"])
```
![318898189-7247340c-6488-4b75-9deb-0ad3f10e03fd](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/f115ae2f-5d91-43bf-8f13-635dd72625a1)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![318898261-71ae0399-a828-406a-93a6-0e36cc31e249](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/52656ab8-7073-4629-a32d-e8dab7d92c2d)

```
np.sqrt(df["Highly Positive Skew"])
```
![318898327-9b500fd0-9b55-4397-b1e8-364652aca983](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/331d87b9-d30b-4ac6-908f-d95649e6241e)
```
np.square(df["Highly Positive Skew"])
```
![318898423-d243323b-c97e-4c55-a41f-f76d176e6461](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/22c73863-51e8-4ddd-9eb3-d0ae5935c732)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![318898509-758eaaba-b780-4fee-8487-d8242a9d6148](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/1ff5458f-650a-4498-aed3-83c1c221b4d3)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![318898927-4945b8c6-e27d-4526-9032-0c0aeb9ab576](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/c6a8d46c-b715-4f43-a13b-6e19ba97ccdc)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![318899248-52a7553c-c1bd-4489-a0cb-b13a27684c23](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/30431d58-6696-43d6-91c8-98aff9120967)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![318899545-3688ed78-4920-4cd4-9e33-4420fc790b8d](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/625680c4-7e46-4b44-b1c4-0ab2f4f796b0)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![318899696-9ef5152c-d766-48e1-857c-a7dbfde4e648](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/96bec20e-6050-4c54-b28d-7b9b9f91e34d)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![318899799-fde4b296-88ec-46ad-b6f3-2cf2b64a15f2](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/0ec51229-37ad-45f9-b52c-ca6d9267b24c)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![318899874-57bae70b-8ee0-4ab1-86bf-733d2597089d](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/eec121d9-cd13-41d2-a4fb-1c3875a74628)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![318900112-3987a28b-3816-41b2-9a9d-6a1cedf8382e](https://github.com/sujigunasekar/EXNO-3-DS/assets/119559822/03deff59-542f-4605-aacd-409fd69d714d)

### RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.



       
