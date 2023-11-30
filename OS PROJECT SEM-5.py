#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('placement.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df=df.drop(columns='StudentID')
df


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df.describe()


# In[11]:


df['PlacementStatus'].value_counts()


# In[12]:


df.groupby('PlacementStatus').size().plot(kind='pie')
plt.title('Placement Distribution')


# In[13]:


sns.countplot(data=df,x='CGPA')
plt.xticks(rotation=90)
plt.title('CGPA Analysis')


# In[14]:


sns.countplot(data=df.loc[(df.PlacementStatus=='Placed')],x='CGPA')
plt.xticks(rotation=90)
plt.title('CGPA wise Placement')


# In[15]:


plt.figure(figsize=(5,5))
sns.displot(df['HSC_Marks'])


# In[16]:


sns.countplot(data=df.loc[(df.PlacementStatus=='Placed')],x='HSC_Marks')
plt.xticks(rotation=90)
plt.title('HSC Marks wise Placement')


# In[17]:


sns.displot(df['SSC_Marks'])


# In[18]:


sns.countplot(data=df.loc[(df.PlacementStatus=='Placed')],x='SSC_Marks')
plt.xticks(rotation=90)
plt.title('SSC Marks wise Placement')


# In[19]:


sns.countplot(data=df.loc[(df.PlacementStatus=='Placed')],x='ExtracurricularActivities')
plt.xticks(rotation=90)
plt.title('ExtracurricularActivities wise Placement')


# In[20]:


sns.countplot(data=df.loc[(df.PlacementStatus=='Placed')],x='SoftSkillsRating')
plt.xticks(rotation=90)
plt.title('Softskills wise Placement')


# In[21]:


sns.countplot(data=df.loc[(df.PlacementStatus=='Placed')],x='Projects')
plt.xticks(rotation=90)
plt.title('Projects wise Placement')


# In[22]:


sns.countplot(data=df.loc[(df.PlacementStatus=='Placed')],x='AptitudeTestScore')
plt.xticks(rotation=90)
plt.title('AptitudeTestScore wise Placement')


# In[23]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
object_cols = df.select_dtypes(include=['object']).columns
for column in object_cols:
    df[column]=labelencoder.fit_transform(df[column])


# In[24]:


correlation_matrix=df.corr()
plt.figure(figsize=(12,6))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')


# In[25]:


X=df.drop(columns='PlacementStatus')
y=df.PlacementStatus


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay


# In[30]:


lr=LogisticRegression(max_iter=50000,penalty=None)
lr.fit(X_train,y_train)
prediction=lr.predict(X_test)
score1=accuracy_score(prediction,y_test)


# In[31]:


print('Accuracy Score is:',score1)


# In[32]:


target_names=['NotPlaced','Placed']


# In[33]:


print('Classification Report is : \n',classification_report(y_test,prediction,target_names=target_names))


# In[34]:


cm=confusion_matrix(y_test,prediction)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names).plot(cmap='Blues')


# In[35]:


from sklearn.ensemble import RandomForestClassifier


# In[36]:


rfc=RandomForestClassifier(criterion='entropy')
rfc.fit(X_train,y_train)
prediction=rfc.predict(X_test)
score2=accuracy_score(prediction,y_test)


# In[37]:


print('Accuracy Score is:',score2)


# In[38]:


target_names=['NotPlaced','Placed']
print('Classification Report is : \n',classification_report(y_test,prediction,target_names=target_names))


# In[39]:


cm=confusion_matrix(y_test,prediction)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names).plot(cmap='Blues')


# In[40]:


final_data = pd.DataFrame({'Models':['LR','RF'],
            'ACC':[score1*100,
                  score2*100,
                  ]})


# In[41]:


final_data


# In[42]:


sns.barplot(x=final_data['Models'],y=final_data['ACC'],width=0.4)
plt.title('Models vs Accuracy')


# In[53]:


new_data = pd.DataFrame({
    'CGPA': float(input("Enter CGPA: ")),
    'Internships': int(input("Enter the number of Internships: ")),
    'Projects': int(input("Enter the number of Projects: ")),
    'Workshops/Certifications': int(input("Enter the number of Workshops/Certifications: ")),
    'AptitudeTestScore': int(input("Enter Aptitude Test Score: ")),
    'SoftSkillsRating': float(input("Enter Soft Skills Rating: ")),
    'ExtracurricularActivities': int(input("Enter the number of Extracurricular Activities: ")),
    'PlacementTraining': int(input("Enter the Placement Training rating (1 or 0): ")),
    'SSC_Marks': float(input("Enter SSC Marks: ")),
    'HSC_Marks': float(input("Enter HSC Marks: ")),
}, index=[0])

print("\nEntered Data:")
print(new_data)


# In[52]:


lr= LogisticRegression()
lr.fit(X,y)







++a
+















































































































+++++++


# In[50]:


p=lr.predict(new_data)
prob=lr.predict_proba(new_data)
if p==1:
    print('Placed')
    print(f"You will be placed with probability of {prob[0][1]:.2f}")
else:
    print("You wont get placed,keep working hard")


# In[ ]:




