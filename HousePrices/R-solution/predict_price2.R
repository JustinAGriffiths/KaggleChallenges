library(MyRUtils)

#read in test and training
df_train = read.csv('../train.csv', header=TRUE, sep=',')
df_test = read.csv('../test.csv', header=TRUE, sep=',')

#fill NA with 0; convert strings to floats based on correlation to SalePrice
df_train <- prepData(df_train, observable="SalePrice")
df_test <- prepData(df_test, prepped_data=df_train)

#delete duplicate coluns created by above process
df_train <- cleanColumns(df_train)
df_test <- cleanColumns(df_test)

## print(colnames(df_train))
## print(colnames(df_test))

write.csv(df_train, file='train_prep.csv')
write.csv(df_test, file='test_prep.csv')

library(randomForest)

#slim all but the seven most correlated variables to SalePrice
df_train_slim <- df_train[names(sort(cor(df_train)["SalePrice",])[length(colnames(df_train))-7:0])]

n_train=1000
n=length(df_train)
rf_slim <- randomForest(SalePrice ~., data=df_train_slim[1:n_train,], importance=TRUE, proximity=TRUE, ntree=1000)
rf_full <- randomForest(SalePrice ~.-Id, data=df_train[1:n_train,], importance=TRUE, proximity=TRUE, ntree=1000)

y <- df_train[n_train:n,"SalePrice"]
yp_slim <- predict(rf_slim, df_train_slim[n_train:n,])
yp_full <- predict(rf_full, df_train[n_train:n,])

analyze_fit(y, yp_slim)
analyze_fit(y, yp_full)

rf_full <- randomForest(SalePrice ~.-Id, data=df_train, importance=TRUE, proximity=TRUE, sampsize=n)
yp_test <- predict(rf_full, df_test)

df_test["SalePrice"] = yp_test
df_test_write = df_test[ c("Id", "SalePrice") ]

write.csv(df_test_write, "result.csv", quote=F, row.names=F)

