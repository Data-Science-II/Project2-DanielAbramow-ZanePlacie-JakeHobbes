Group members: Daniel Abramow, Zane Placie, Jake Hobbes

DANIEL

How to run the Scala project: 

Navigate to Scala/p2/src/main/scala and open main.scala. For each dataset, feature selection can be run for each model. For example, on line 35, “t.run()” will perform feature selection for transformed regression and generate the necessary graphs. Simply uncomment the lines for each model you wish to run for each dataset. Then, enter the sbt command line tool, compile, and enter “runMain”.

ZANE

How to run the Python files:

Navigate to the Python folder. Then, to run either file, simply enter “python file_name.py” into command line.

The first two methods, makeModel and measureModel, are helper methods, the first of which builds neural networks of various types (perceptrons as well as neural nets with one and two fully connected hidden layers) and the second of which returns various measures of a particular neural net's fit on a dataset. Various hyperparameters can be changed in these methods, and these are noted in the comments

The next six methods implement the backward elimination, forward selection, and stepwise regression techniques of feature selection. The first three methods in this group implement one step of each of these three techniques, while the second three methods implement each recursively (by calling the first three) until no improvement in model fit can be found.

The next two methods plot the r-squared, adjusted r-squared, and cross-validated r-squared against the number of predictors in the  model, using both the backward elimination and forward selection techniques. 

The final method displays the output for each dataset. Specifically, for each type of neural net, it prints the models generated by  repeated use of backward elimination, forward selection, and stepwise regression for each of the three measures of model fit, then  prints the two plots generated by the previous two methods. 

The code after the methods is what should be run directly. The code for each of the five datasets is essentially the same. First, the dataset is read into Python. Then, summariesPrinter is called to print the output.

JAKE

After running the import statements, the first section is the makeModel method, which creates various neural net  models (Perceptron, NN3L, NN4L) based on input argument 'model_type'. This section also performs L1 and L2  (Lasso and Ridge) regularization based on input argument 'regularizer'.

The next section contains the measureModel method, which calls the makeModel method and returns various measures of  model fit (Rsq, Rsq-Adj, Rsq-CV) for the given dataset.

The code in the last second contains the summariesPrinter method, which takes in a dataset and the name of the response vector as input, and then calls the above methods such that the 3 model types and measures of model fit are all printed in an easily readable format. The code at the bottom of the file reads in the autompg, concrete, aquatic, wine, and concrete_strength datasets and runs the summariesPrinter method for each dataset.
