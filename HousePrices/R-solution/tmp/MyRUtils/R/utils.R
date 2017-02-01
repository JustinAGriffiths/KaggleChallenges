#' Prep data frame function for regression analysis
#'
#' This function takes a input
#' 'training' dataframe
#' and optional testing dataframe
#' removes nulls, converts string types to float
#'
#' keywords trainingSample observable
#' good luck!

prepData <- function( df_train, observable, df_test=NULL ) {

    cat('before, ', df_train$PoolQC[0], '\n')
    df_train[is.na(df_train)] <- 0
    if( !is.null(df_test) ) df_train_test[is.na(df_test)] <- 0
    cat('after, ', df_train$PoolQC[0], '\n')

}
