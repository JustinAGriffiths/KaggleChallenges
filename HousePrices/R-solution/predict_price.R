df_train = read.csv('../train.csv', header=TRUE, sep=',')
df_test = read.csv('../test.csv', header=TRUE, sep=',')
df_train[is.na(df_train)] <- 0
df_test[is.na(df_test)] <- 0

print( nrow(df_train) )

for (col in colnames(df_train )) {
                                        #print( col )
    if ( lapply(df_train[col],class) == "character" ){
        cat ( 'char col ', col, '\n' )

        means<-c()
        col2<-paste(col,'_f', sep='')
        df_train[col2] <- -1
        df_test[col2] <- -1
        for( item in unique( df_train[[col]]) ) {
            m<-mean( df_train[df_train[col]==item,][["SalePrice"]] ) 
            means<-c(means, m)
            #cat( item, m, '\n' )
        }
        for( item in unique( df_train[[col]]) ) {
            m<-mean( df_train[df_train[col]==item,][["SalePrice"]] ) 
            df_train[ df_train[ col ] == item, ][[col2]]<-m/max(means)
            if(nrow(df_test[ df_test[col] == item, ])) {
                df_test[ df_test[ col ] == item, ][[col2]]<-m/max(means)
                #cat( 'yes for ', col, '\n')
            }
            #else cat( 'no for ', col, '\n')
        }

        #print (means)
        df_train[col] = NULL
        df_test[col] = NULL
    }    
}


print (colnames(df_train))
print (colnames(df_test))

write.csv(df_train, file='train_prep.csv')
write.csv(df_test, file='test_prep.csv')


library(randomForest)

df_train_slim <- df_train[names(sort(cor(df_train)["SalePrice",])[length(colnames(df_train))-7:0])]

rf_1 <- randomForest(SalePrice ~., data=df_train_slim, importance=TRUE, proximity=TRUE)
yp <- predict(rf_1, df_train_slim)

yres <- sum(((yp-df_train_slim$SalePrice)/df_train_slim$SalePrice)**2)/length(yp)

sum2 <- 0

for( i in seq_along(yp)){
    sum2 <- sum2 + ((yp[i]-df_train_slim$SalePrice[i])/df_train_slim$SalePrice[i])**2
}
sum2 <- sum2/length(yp)

cat( yres, sum2, '\n' )
