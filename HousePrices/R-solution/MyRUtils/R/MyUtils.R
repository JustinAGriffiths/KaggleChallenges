#' Prep data frame function for regression analysis
#'
#' This function takes a input
#' 'training' dataframe
#' and optional testing dataframe
#' removes nulls, converts string types to float
#'
#' keywords trainingSample observable
#' @keywords df_test
#' @export
#' @examples
#' prepData()

prepData <- function( df_train, observable=NULL, prepped_data=NULL ) {


    df_train <- removeNull(df_train)
    theColNames <- colnames(df_train )
    columnsToDelete = c()
    for (col in theColNames) {
        if ( lapply(df_train[col],class) == "character" ){
            columnsToDelete <- c(columnsToDelete, col)
            means<-c()
            col2<-paste(col,'_f', sep='')
            col3<-paste('delete_',col, sep='')
            df_train[col2] <- -1
            if ( !is.null(observable) ){
                items = unique( df_train[[col]])
                for( item in items  ) {
                    m<-mean( df_train[df_train[col]==item,][[observable]] ) 
                    means<-c(means, m)
                }
                for( i in seq_along(items)) {
                    item=unique(df_train[[col]])[i]
                    m=means[i]
                    df_train[ df_train[ col ] == item, ][[col2]]<-m/max(means)
                }
                df_train[col3] = df_train[col]
            }
            else {
                items = unique( df_train[[col]])
                prepped_items = unique( prepped_data[[col3]])
                for( item in items  ) {
                    if(nrow(df_train[ df_train[col] == item, ])) {
                        prepped_value <- 0
                        if (item %in% prepped_items) 
                            prepped_value <- unique(prepped_data[prepped_data[col3]==item,][[col2]]) 
                        df_train[ df_train[col] == item, ][[col2]] <- prepped_value
                    }
                }
            }
        }   
    }
    df_train[columnsToDelete]=NULL
    df_train <- df_train    
}

#' To be called after prep_data
#' @keywords df_test
#' @export
#' @examples
#' cleanColumns()
cleanColumns <- function(df) {
    for( col in colnames(df) ){
        if (substr(col,1,7)=='delete_'){
            df[col] <- NULL
        }
    }
    df<-df
}

#' To be called after prep_data
#' @keywords x,y
#' @export
#' @examples
#' analyze_fit(y, yfit)
analyze_fit <- function(y,yp) {
    error <- sqrt( sum( ((y-yp)/y)**2) )/length(y)
    error2 <- sqrt( sum( (log(yp)-log(y))**2 )/length(y) )
    m <- mean( abs(y-yp) )
    std <- sd( y-yp )

    cat('error: ', error, ' error2: ', error2,', mean: ', m, ', sd: ', std, '\n')
}

#' Internal function
#' removeNull()
removeNull <- function(df) {
    df[is.na(df)] <- 0
    df <- df
}

