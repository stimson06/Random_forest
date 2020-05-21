import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

    
def Data_info(train_data, test_data):
    print(train_data.describe(include=['O']))
    missing_val_train=train_data.isna().sum()
    missing_val_test=test_data.isna().sum()
    print(missing_val_train)
    print("\n"*3)
    print(missing_val_test)
        
def Pivot_Analysis(train_data):
    pivot_class=train_data[['Pclass', 'Survived']].groupby(['Pclass'],
              as_index=False).mean().sort_values(by='Survived', ascending=False)
    pivot_sex=train_data[["Sex", "Survived"]].groupby(['Sex'], 
              as_index=False).mean().sort_values(by='Survived', ascending=False)   
    pivot_sibsp=train_data[["SibSp", "Survived"]].groupby(['SibSp'], 
              as_index=False).mean().sort_values(by='Survived', ascending=False) 
    pivot_parch=train_data[["Parch", "Survived"]].groupby(['Parch'], 
              as_index=False).mean().sort_values(by='Survived', ascending=False)
    print(pivot_class,"\n"*3,pivot_sex,"\n"*3,pivot_sibsp,"\n"*3,pivot_parch)
   
def Visualisation(train_data):
    
    #Figure 1
    plt.subplot(2,2,1)
    train_data.groupby("Sex")["Survived"].count().sort_values(ascending=False).plot.bar()
    plt.title("Sex", fontweight="bold")
    plt.subplot(2,2,2)
    train_data.groupby("Embarked")["Survived"].count().sort_values(ascending=False).plot.bar()
    plt.title("Embarked", fontweight="bold")
    
    #Figure 2
    g = sns.FacetGrid(train_data, col='Survived')
    g.map(plt.hist, 'Age', bins=15)
    g.add_legend()
    
    #Figure 3
    grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=10)
    grid.add_legend()
    
    #Figure 4
    grid= sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()
   
    #Figure 5
    grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()
    
    plt.show()

def Data_Wrangling(train_data, test_data):
    
    train_data = train_data.drop(['Ticket', 'Cabin','PassengerId'], axis=1)
    test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
    combine=[train_data,test_data]
    
    #Extracting the titles of the names
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    #Replacing the titles that best fits
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major',
 	                                                  'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    #Mapping the titles with values
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    
    #Creating a band for Continuous datatype
    train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
    age_band=train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], 
                as_index=False).mean().sort_values(by='AgeBand', ascending=True)
    ##print(age_band)
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']=4
    train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
    fare_band=train_data[['FareBand', 'Survived']].groupby(['FareBand'], 
               as_index=False).mean().sort_values(by='FareBand', ascending=True)
    ##print(fare_band)
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        
    #Creating new features
    train_data["Family"]=train_data["SibSp"]+train_data["Parch"]
    test_data["Family"]=test_data["SibSp"]+test_data["Parch"]
    
    #Dropping of features that doest contribute any to the system 
    train_data=train_data.drop(['Name','AgeBand','FareBand'],axis=1)
    test_data=test_data.drop(['Name'],axis=1)
    
    return train_data,test_data

def Imputation(train_data, test_data):
    
    #Impuation of numeric data in training data\
    col_numeric = list(train_data.select_dtypes(exclude="object")) # Columns with numerical datatype
    imputed_data=pd.DataFrame(train_data,columns=col_numeric)
    Impute=IterativeImputer(max_iter=10, random_state=610) # Iterative imputation
    imputed_train=pd.DataFrame(Impute.fit_transform(pd.DataFrame(imputed_data)),columns=col_numeric)
    train_data[col_numeric]=imputed_train[col_numeric]
    
    #Impuation of numeric data in test data
    col_numeric = list(test_data.select_dtypes(exclude="object")) # Columns with numerical datatype
    imputed_data=pd.DataFrame(test_data,columns=col_numeric)
    Impute=IterativeImputer(max_iter=10, random_state=610) # Iterative imputation
    imputed_test=pd.DataFrame(Impute.fit_transform(pd.DataFrame(imputed_data)),columns=col_numeric)
    test_data[col_numeric]=imputed_test[col_numeric]
    
    #Completing the Missing values in Embarked
    train_data['Embarked']=train_data['Embarked'].fillna('S')
    
    #Label Encoding
    label_encoder=LabelEncoder()
    train_data["Sex"]=label_encoder.fit_transform(train_data["Sex"])
    train_data["Embarked"]=label_encoder.fit_transform(train_data["Embarked"])
    test_data["Sex"]=label_encoder.fit_transform(test_data["Sex"])
    test_data["Embarked"]=label_encoder.fit_transform(test_data["Embarked"])
    return train_data, test_data

def Model(train_data, test_data):
    
    X_train = train_data.drop("Survived", axis=1)
    Y_train = train_data["Survived"]
    X_test  = test_data.drop("PassengerId", axis=1).copy()
    
    #Random forest classifier
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    acc_random_forest = random_forest.score(X_train, Y_train)
    print(acc_random_forest)
    
    #Submission
    submission = pd.DataFrame({"PassengerId": test_data["PassengerId"].astype('int'), "Survived": Y_pred})
    submission.to_csv('F:\ML\Git_repo\Decision_Tree\submission.csv', index=False)

train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")
Pivot_Analysis(train_data)
Visualisation(train_data)
train_data, test_data=Imputation(train_data, test_data)
##Data_info(train_data, test_data)
train_data, test_data=Data_Wrangling(train_data,test_data)
##Data_info(train_data, test_data)
Model(train_data, test_data)

