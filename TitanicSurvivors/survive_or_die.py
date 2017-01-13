#!/usr/bin/env python
from __future__ import print_function

# class passenger :
#     def __init__(self, row) :
#         print row
#         self.PassengerId=int(row[0])
#         self.Survived=int(row[1])
#         self.Pclass=int(row[2])
#         self.Name=row[3]
#         self.Sex=0
#         if row[4]=="female" : self.Sex=1
#         self.Age=-1
#         if len(row[5]) : self.Age=int(float(row[5]))
#         self.SibSp=row[6]
#         self.Parch=row[7]
#         self.Ticket=row[8]
#         self.Fare=float(row[9])
#         self.Cabin=row[10]
#         self.Embarked=row[11]
#         pass

def eff_survived(df, value, condition='Survived==1', condition_den='', bins=10, fname=''):
    if fname=='': fname=value+'_eff.png'
    if condition_den != '' :
        condition='(%s) and %s' %(condition, condition_den)
    num=df[df.eval(condition)][value].values
    try : den=df[df.eval(condition_den)][value].values
    except : den=df[value].values
    from hist import efficiency
    efficiency(num, den, bins=bins, xlabel=value, ylabel='% survived', fname=fname)

def prep_data(input_file_name):
    import csv
    import pandas as pd
    import numpy as np

    df = pd.read_csv(input_file_name)

    df['sex'] = df.Sex.map( lambda x: int(0) if x=='male' else int(1) )
    

    df['age'] = df['Age']
    for i in range(0,2): #range
        for j in range(1,4): #class            
            df.loc[ (df.sex==i) & (df.Pclass==j) & df.age.isnull(), 'age'] = df.query('sex==%i and Pclass==%i' %(i,j)).Age.median()
            continue
        continue

    #print df.Age.mean(), df.age.mean()

    df['unk_age'] = df['Age'].map( lambda x: 1 if x==x else 0 )
    #df.info()
    df['family'] = df['SibSp']+df['Parch']

    #for e,v in ( {'C':0, 'S':1, 'Q':2}).iteritems():
    for e,v in  {'C':0, 'S':1, 'Q':2}.items():
        df.loc[ (df.Embarked==e), 'embarked' ] = v
        continue
    df.loc[ (df.embarked.isnull()), 'embarked'] = 0
        

    df['cabin'] = df.Cabin.map( lambda x: 0 if x!=x else len(x))
    df['Fare'] = df.Fare.map( lambda x: 0 if x!=x else x)

    #    df = df.drop( ['Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked', 'embarked', 'cabin', 'unk_age', 'Parch', 'SibSp', 'family'], axis=1)
    df = df.drop( ['Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis=1)

#     for column in df.columns.unique():
#         print column, len(df.query('%s!=%s' %(column, column)))

        
    return df

def gender_model(df):
    df['Survived'] = df['sex'].map( lambda x: x )

def write_results(name, df):
    f=open(name,'w')
    f.write('PassengerId,Survived\n')
    for r in range(0,len(df)):
        f.write('%s,%s\n' %(df['PassengerId'][r],df['Survived'][r]))
        continue
    pass

def print_query(df, query, description):
    d=len(df.query(query))
    n=len(df.query(query+'and Survived==1'))
#     an=len(df.query('not('+query+') and Survived==1'))
#     ad=len(df.query('not('+query+') '))
    if d>0 : print (description, ': ', n*1./d, ' total query: ', d)#, ' flipped: ', an*1./ad, '(',ad,') Score: ', (n+ad-an)*1./len(df) 
    else : print (description, ': has no events' )

def custom_model(df, query, col='Survived', n=2):

    df[col] = df.eval(query).astype(int)    
    print ('success rate: ', len(df[df[col]==df['Survived']])*1./len(df))
    pass

def get_weight(df, query):
    return '%f*(%s)' %(len(df.query('('+query+') and Survived==1'))*1./len(df.query(query)), query)

def main(args) :

    try : input_file_name = args[1]
    except : input_file_name = 'train.csv'

    try : output_file_name = args[2]
    except : output_file_name = 'test.csv'

    df_train = prep_data(input_file_name)
    df_test = prep_data(output_file_name)

    import hist
#     hist.efficiency( df_train[df_train.Survived==1].age.values, df_train.age.values, bins=[0,10,20,30,40,60,100], xlabel='Age', fname='Age_eff.jpg')
#     hist.efficiency( df_train[df_train.Survived==1].Pclass.values, df_train.Pclass.values, bins=3, xlabel='Pclass', fname='Pclass_eff.jpg')
    eff_survived(df_train, 'age', bins=[0,10,20,30,40,60,100])
    eff_survived(df_train, 'Pclass', bins=3)
    eff_survived(df_train, 'sex', bins=2)
    eff_survived(df_train, 'Fare', bins=[0,20,40,60,80,100,1000])
    eff_survived(df_train, 'cabin', bins=[0,1,2,3,4,5,6,7,8,9,10])
    eff_survived(df_train, 'Parch', bins=[0,1,2,3,4,5,6,7,8,9,10])
    eff_survived(df_train, 'SibSp', bins=[0,1,2,3,4,5,6,7,8,9,10])
    eff_survived(df_train, 'family', bins=[0,1,2,3,4,5,6,7,8,9,10])
    eff_survived(df_train, 'unk_age', bins=2)

    hist.hist( df_train, 'age' )
    hist.hist( df_train, 'Fare', bins=20)
    hist.hist( df_train, 'cabin', bins=[0,1,2,3,4,5,6,7,8,9,10])
    hist.hist(df_train, 'Parch', bins=[0,1,2,3,4,5,6,7,8,9,10])
    hist.hist(df_train, 'SibSp', bins=[0,1,2,3,4,5,6,7,8,9,10])
    hist.hist(df_train, 'family', bins=[0,1,2,3,4,5,6,7,8,9,10])


    gender_model(df_test)
    write_results('gender_model.csv',df_test)

    print_query(df_train, 'sex==1', 'gender_model')
    condition='(sex==1 and Pclass<3) or (age<=10 and Pclass<3) or (sex==1 and Pclass==3 and Fare<9) or family==3'
    print_query(df_train, condition, 'all')
    custom_model(df_test,  condition)
    custom_model(df_train, condition, col='Survived2')
    write_results('custom_model.csv', df_test)

    from sklearn.ensemble import RandomForestClassifier 
    forest = RandomForestClassifier(n_estimators = 200)
        
    train_data=df_train.drop(['PassengerId'], axis=1).values
    train_data_test=df_train.drop(['Survived', 'PassengerId'], axis=1).values
    test_data=df_test.drop(['PassengerId'], axis=1).values
    forest = forest.fit(train_data[0::,1::],train_data[0::,0])
    
    output = forest.predict(test_data)
    df_test['Survived'] = output.astype(int)
    write_results('forest_model.csv', df_test)

    output = forest.predict(train_data_test)
    df_train['rf'] = output.astype(int)
    custom_model(df_train, 'rf==1', col='Survived2')

if __name__=="__main__" :
    import sys
    main(sys.argv[1:])
