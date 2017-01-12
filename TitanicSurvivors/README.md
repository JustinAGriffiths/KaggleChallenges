Titanic Survivors
=================

This is a simple package that attempts to predict whether a given passenger will survive based on known data of the passenger.

See here https://www.kaggle.com/c/titanic

## Usage

* ./survive_or_die.py [train.csv] [test.csv] [--condition '']

By default local train.csv, test.csv files are read in.  Optionally pass a custom condition:

* --condition 'sex==1' #females survive
* --condition 'Pclass==1 or sex==1' #females survive or first class passengers survive

## Models

* Gender Model, women survive, mend die
* Modified simple model: note that 1st and 2nd class females have an extremely high probability of surviving, as do children.  Oddly enough, low fare 3rd class female passengers have a higher survival probility than high fare female passengers--probably low stats
* Random Forest--not high perfomant