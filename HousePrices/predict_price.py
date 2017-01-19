#!/usr/bin/env python
from __future__ import print_function

import pylab as plt
plt.ioff()
import sklearn
import math

def get_del( v, df):
    sum2=0
    #for i in range(0, len(v)): sum+=(abs(v[i]-df[i])/df[i])
    #return sum/len(df)
    for i in range(0, len(df)): sum2+=(math.log(v[i])-math.log(df[i]))**2
    return math.sqrt(sum2/len(df))

def train( df, regressor, veto_vars=None, observable='SalePrice_f', train_frac=1, n_estimators=200 ):
    n_veto_vars = 0
    if veto_vars is not None :  n_veto_vars=len(veto_vars)
    print ('training ', regressor, ' removing ', n_veto_vars, ' variables training frac: ', train_frac )
    if veto_vars is not None : 
        df = df.drop(veto_vars, axis=1)
        #print ('slimmed columns: ', df.columns)
        pass
    ntrain_data = int(train_frac*len(df))
    forest = None
    random_state=1234
    if regressor == sklearn.ensemble.AdaBoostRegressor :
        dt_stump = sklearn.tree.DecisionTreeRegressor(max_depth=None, max_features=0.5, min_samples_leaf=1,splitter='random')
        forest = regressor(n_estimators=500, random_state=random_state, learning_rate=1, loss='linear', base_estimator=dt_stump)
        pass
    elif regressor == sklearn.ensemble.RandomForestRegressor :
        forest = regressor(n_estimators=100, random_state=random_state)
        pass
    elif regressor == sklearn.ensemble.GradientBoostingRegressor : 
        forest = regressor(n_estimators=n_estimators, random_state=random_state, max_depth=3)
        pass
    else : # BaggingRegressor
        forest = regressor(n_estimators=n_estimators, random_state=random_state)
        pass

    forest = forest.fit( df.drop([observable], axis=1).values[0:ntrain_data], df[observable].values[0:ntrain_data] )

    ranking = []
    try : 
        importance = forest.feature_importances_
        #print ('len(importance)', len(importance))
        for i in range(0, len(importance)) :
            ranking.append( [importance[i], df.columns[i]] )        
            continue
    
        ranking = sorted(ranking)
        #     for i in range(len(ranking)-1, len(importance)-5, -1):
        #         print (ranking[i])
        #         continue;

        #if hasattr(forest, 'train_score_'): print ( (forest.train_score_[-1]-forest.train_score_[-2])/forest.train_score_[-1], (forest.train_score_[1]-forest.train_score_[0])/forest.train_score_[1])
    except : pass
    return forest, ranking
    
def evaluate_training( df, forest, train_frac=1, test_sample=None, vars=None, observable='SalePrice_f', name=''):
    ntrain = int(train_frac*len(df))
    ntest = int(len(df)-ntrain)
    self_test = True
    if test_sample is not None :
        ntrain=len(df)
        ntest=len(test_sample)
        self_test=False
    else:
        test_sample=df.drop([observable],axis=1)[ntrain:]
        pass

    if vars is not None : test_sample=test_sample.drop(vars, axis=1)
    output=forest.predict( test_sample.values )
    if self_test : 
        real_prices=df[observable][ntrain:].tolist()
        print ('del: ', get_del(output, df.SalePrice_f.values[ntrain:]) )
        f=open('my_test_submission_'+name+'.csv', 'w')
        f.write('Id,SalePrice,SalePriceActual\n')
        for i in range(0, len(test_sample)):
            f.write('%i,%f,%i\n' %(ntrain+1+i, output[i],int(real_prices[i]))) #fixme
            continue
        f.close()
    else :
        f=open('my_submission2_'+name+'.csv', 'w')
        f.write('Id,SalePrice\n')
        for i in range(0, len(test_sample)):
            f.write('%i,%f\n' %(1461+i, output[i])) #fixme
            continue
        f.close()
        
def decorrelate(df, col1, col2, v1, v2):
    df[col1+'_p'] = df[col1]/v1 + df[col2]/v2
    df[col2+'_p'] = df[col1]/v1 - df[col2]/v2
    df[col1] = df[col1+'_p']
    df[col2] = df[col2+'_p']
    df.drop( [col1+'_p', col2+'_p'], axis=1, inplace=True)

def findCorrelations(df, df_test, threshold=0.5):
    df2=df.drop(['SalePrice_f'], axis=1)
    count=0
    from math import sqrt
    while ( count<200 and True ):
        count+=1
        m=df2.corr()
        drop_out=True
        for col in m.columns:
            if m[col].drop(col).max()>threshold:
                col2=m[col].drop(col).idxmax()
                #print (col, col2, m[col][col2])
                v1=sqrt(df2[col].var())
                v2=sqrt(df2[col2].var())
                decorrelate(df, col, col2, v1, v2)
                decorrelate(df2, col, col2, v1, v2)
                decorrelate(df_test, col, col2, v1, v2)
                m2=df2.corr()
                #print (col, col2, m2[col][col2])
                drop_out=False
                break
            continue
        if drop_out: break
        continue
    print ('ending decorrelation after, ', count, ' iterations')

def dropUncorrelated(df, df_test, col, threshold=0.5):
    m=df.corr()
    todrop=[]
    for col2 in m.columns :
        if abs(m[col][col2])<threshold : todrop.append(col2)
        continue
    print ('dropping ', todrop)
    df.drop( todrop, axis=1, inplace=True)
    df_test.drop( todrop, axis=1, inplace=True)
    
    
        
def make_hist(df, col1, col2):
    means=[]
    labels=[]
    print ('making hist for, ', col1)
    if df[col1].dtype == 'O' :
        for s in df[col1].unique():
            if s != s: 
                labels.append('nan')
                means.append(df[df.eval('%s!=%s' %(col1,col1))][col2].mean())
                pass
            else : 
                labels.append(s)
                means.append(df[df.eval('%s=="%s"' %(col1,s))][col2].mean())
                pass
            continue
        

        plt.clf()
        plt.plot( [i for i in range(0, len(means))], means, 'b.', ms=15)
        plt.xticks( plt.arange(len(means)), df[col1].unique(), horizontalalignment='center' )
        plt.xlim(xmin=-0.2,xmax=len(means)+0.2)
        plt.ylim(ymin=0.5*min(means), ymax=1.2*max(means))
        plt.savefig(col1+"_"+col2+'.png')
        plt.clf()

    elif col1.endswith('_f') :
        
        nbins=10
        if col1.endswith('_o_f') : 
            nbins_tmp = sorted(df[col1].unique())
            nbins=[0.]
            for item in nbins_tmp:
                nbins.append( item*1.0001 )
                continue
            pass
        elif len(df[col1].unique())< nbins : nbins=len(df[col1].unique())
        h=plt.hist( df[col1], bins=nbins )
        means = []
        xs = []
        for i in range(0, len(h[1])-1):
            xs.append( (h[1][i]+h[1][i+1])/2 )
            val_up=h[1][i+1]
            if i+2==len(h[1]) : val_up*=1.0001
            mean = df[ df.eval('%s>=%f and %s<%f' %(col1, h[1][i], col1, val_up)) ][col2].mean()
            if col1 == 'BsmtHalfBath_f':
                print( h[1][i], h[1][i+1], xs[i], mean, h[0], h[1])
            if mean != mean : mean = 0
            means.append( mean )
            continue
        if col1 == 'BsmtHalfBath_f': print( xs, means)
        #print (col1, means, xs)
        plt.plot( xs, means, 'b.', ms=15 )
        plt.xlim(xmin=0.8*min(xs), xmax=max(xs)*1.2)
        plt.ylim(ymin=0.5*min(means), ymax=1.2*max(means))
        plt.savefig(col1+"_"+col2+'.png')
        plt.clf()

        pass#if =='O'
    pass#function

        

def make_price_per_column(df):
    for column in df.columns:
        if column=='SalePrice' : continue
        make_hist(df, column, 'SalePrice')

        continue
    pass

def addVars(df):
    #     df['GarageBltWHouse_f'] = (df.YearBuilt==df.GarageYrBlt).astype('float64')
    #df['TotalBaths_f'] = (df.BsmtFullBath_f+df.BsmtHalfBath_f+df.FullBath+df.HalfBath)
    #df['AreaPerRoom_f'] = df.GrLivArea_f/(df.TotRmsAbvGrd_f)
    #df['TotalFinBasementSF_f'] = df.BsmtFinSF1_f+df.BsmtFinSF2_f
    pass

def prep_data(df, df_test):
#     drop=df.columns.tolist()
#     drop.remove('SalePrice')
#     drop.remove('Id')

#     for i in ['OverallQual', 'GrLivArea','Neighborhood','2ndFlrSF','TotalBsmtSF','1stFlrSF','BsmtFinSF1','GarageCars']: drop.remove(i)

#     df.drop(drop, axis=1,inplace=True)
#     df_test.drop(drop, axis=1,inplace=True)
    

    #     #convert ints to strings
    #     ints=['MSSubClass',]
    #     prep_ints(df,ints)
    #     prep_ints(df_test,ints)

    #print(df_test.columns.tolist())

    for col in df.columns:
        if df[col].dtype == 'O' :
            prep_object(df, df_test, col, 'SalePrice')
        else : 
            prep_float(df, col)
            if col != 'SalePrice' : prep_float(df_test, col)
            #print (col, len(df[df[col]!=df[col]]))
            pass
        continue
    pass

def prep_ints(df, ints):
    for col in ints:
        df[col+'_i'] = df[col].map(lambda x: str(x))
        continue
    df.drop(ints, axis=1,inplace=True)

def prep_float(df, col):
    prepend=''
    if col[0].isdigit() : prepend='d_'
    df[prepend+col+'_f'] = df[col].map( lambda x: x if x==x else 0 )
    return 0

def prep_object(df, df_test, col, col2):
    prepend=''
    if col[0].isdigit() : prepend='d_'

    values = {}
    for s in df[col].unique():
        label = ''
        mean = -1
        if s != s: 
            label = 'nan'
            mean = df[df.eval('%s!=%s' %(col,col))][col2].mean()
            pass
        else : 
            label = s
            mean = df[df.eval('%s=="%s"' %(col,s))][col2].mean()
            pass
    
        while mean in values :
            mean *= 1.0001
            continue
        values[mean] = label
        continue

    max=sorted(values)[-1]
    min=sorted(values)[0]
    if max==0 : max+0.001

    new_values = {}
    #print (values, max, min)
    for m in sorted(values):
        v=values[m]
        new_values[v] = 1*m/max
        continue
    #print (new_values)
    if 'nan' not in new_values: new_values['nan'] = 0
#     intersection=len(set(df[col].unique()).intersection(set(df_test[col].unique())))
#     if col=='MSSubClass_i' and (intersection<len(df[col].unique()) or intersection<len(df_test[col].unique())) : print ('problem? ', col, intersection, len(df[col].unique()), len(df_test[col].unique()), sorted(df[col].unique().tolist()), sorted(df_test[col].unique().tolist()))
    df[prepend+col+'_o_f'] = df[col].map( lambda x: new_values[x] if x==x else new_values['nan'])
    if col=='BsmtFinSF1_i': 
        print (df_test['BsmtFinSF1_i'].unique())
    df_test[prepend+col+'_o_f'] = df_test[col].map( lambda x: new_values[x] if (x==x and x in new_values) else new_values['nan'])




def main(args):
    
    try : train_name=args[0]
    except : train_name='train.csv'

    try : test_name=args[1]
    except : test_name='test.csv'

    print (train_name, test_name)


    import pandas as pd
    df_train=pd.read_csv(train_name)
    df_test=pd.read_csv(test_name)

    prep_data(df_train, df_test)
    addVars(df_train)
    addVars(df_test)

    #print(df_train.columns)
    #make_price_per_column(df_train)

    trim = []
    for col in df_train.columns :
        if not col.endswith('_f'): trim.append(col)
        if col=='Id_f': trim.append(col)
        continue
    df_train=df_train.drop( trim, axis=1 )
    trim.remove('SalePrice')
    df_test=df_test.drop( trim, axis=1 )

    #dropUncorrelated(df_train, df_test, 'SalePrice_f', threshold=0.15)
    #findCorrelations(df_train, df_test,threshold=0.5)

    print (df_train.columns)

    df_train.to_csv('train_prep.csv')


    train_data = df_train.values
    train_data_test = df_train.drop( ['SalePrice_f'], axis=1).values
    test_data = df_test.values
    train_data_price = df_train.SalePrice_f.values

    for iregressor in range(0,4):
    #for iregressor in range(1,2):
        rtype = ''
        regressor = None
        if iregressor == 0 :
            rtype = 'AdaBoostRegressor'
            from sklearn.ensemble import AdaBoostRegressor 
            regressor = AdaBoostRegressor
            pass
        elif iregressor == 1 :
            rtype = 'GradientBoostingRegressor'
            from sklearn.ensemble import GradientBoostingRegressor
            regressor = GradientBoostingRegressor
            pass
        elif iregressor == 2 :
            rtype = 'RandomForestRegressor'
            from sklearn.ensemble import RandomForestRegressor 
            regressor = RandomForestRegressor
            pass
        elif iregressor == 3 :
            rtype = 'BaggingRegressor'
            from sklearn.ensemble import BaggingRegressor
            regressor = BaggingRegressor
            pass
            
        
        #print ('training ', rtype)
        train_frac=0.75
        forest, ranking = train( df_train, regressor)
        #         for var in ranking:
        #             print (var)
        forest_train_0p75, ranking_train_0p75 = train( df_train, regressor, train_frac=train_frac)

        #         veto_vars = []
        #         for i in sorted(ranking):
        #             #print (i, ranking[i])
        #             #if len(veto_vars)>=len(ranking)-40 : break
        #             if i[0]>0.001 : break
        #             veto_vars.append(i[1])
        #             continue
        
        #         forest2, ranking2 = train(df_train, regressor, veto_vars=veto_vars)
        #         forest2_train_0p75, ranking2_train_0p75 = train( df_train, regressor, train_frac=0.75, veto_vars=veto_vars)

        evaluate_training( df_train, forest_train_0p75, train_frac=train_frac, name=rtype)
        evaluate_training( df_train, forest_train_0p75, test_sample=df_test, name=rtype)
        #         evaluate_training( df_train, forest2_train_0p75, train_frac=0.75, vars=veto_vars)


        #         #from sklearn.ensemble import RandomForestRegressor #AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
        #         forest = regressor(n_estimators = 200)
        
        #         forest = forest.fit(train_data_test,train_data_price.ravel())
        #         output = forest.predict(train_data_test)
        #         print( 'performance on trained sample: ', test_performance( output, df_train, regressor ) )
        #         output_test = forest.predict(test_data)
        #         df_test2['SalePrice_f'] = output_test
        #         #         for i in range(0,20):
        #         #             print( output[i], ' ', df_train.SalePrice_f[i] )
        #         #             continue
        
        #         output_f=open('my_submission_'+rtype+'.csv', 'w')
        #         output_f.write('Id,SalePrice\n')
        #         for i in range(0, len(df_test2)):
        #             output_f.write('%i,%f\n' %(int(df_test.Id_f[i]), df_test2.SalePrice_f[i]))
        #             continue
    
        #         importance = forest.feature_importances_
        #         importance_dict = {}
        #         i=0
        #         for col in df_train.columns :
        #             if col.count('Unnamed') : continue
        #             if i >= len(importance) : break
        #             importance_dict[importance[i]] = col
        #             i+=1
        #             continue
        
        #         output_f.close()
        #         output_f = open ('my_ranking_'+rtype+'.txt', 'w')
        #         for i in sorted(importance_dict):
        #             output_f.write('%s: %f\n' %(importance_dict[i], i))
        #             continue
        #         output_f.close()
        
    #         #print(len(df_test), len(output_test), df_test2.SalePrice_f[0], output_test[0])


if __name__=="__main__":
    import sys
    main(sys.argv[1:])
