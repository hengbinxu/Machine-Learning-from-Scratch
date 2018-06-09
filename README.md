### Machine Learning from Scratch
***
This project implements some common machine learing algorithm from scratch.

Machine Learning can separate to three step.
 
 1. Function set.(Hypothesis)
 2. Goodness of a Function.(Define the loss function)
 3. Find the best function.(Optimization)

 In this project, I will follow this step to demonstrate how this model work.

- __Perceptron Learning Algorithm__

```pyhton
PLA = Perceptron()
PLA.training(x = training_x, y = training_y, max_iter = 1000)
pred = PLA.predict(testing_x = testing_x)
PLA.accuracy(prediction = pred, testing_y = testing_y) 
```
```python
plot_decision_boundary(data = final_iris, x_var= "sepal length (cm)", y_var = "petal length (cm)",target_variable = "target_class")
```
<img src = ".\\picture\\PLA_DecisionBoundary.png">

Animate fitting process

```python
animate_process(sub_data = subset_iris, x_var = 'sepal length (cm)', y_var = "sepal width (cm)")
```
<img src = ".\\picture\\PLA_fit.gif">

- __Naive Bayes Classifier__

```python
NB = Gaussian_Naive_Bayes()
NB.fit(training_data = training_data, target_variable= "Class")
pred = NB.predict(testing_x = testing_x)
acc = NB.accuracy(pred, y = testing_y)
```
```python
plot_decision_boundary(data = df, x_var = "BMI", y_var = "Age", target_variable = "Class")
```
<img src = ".\\picture\\NB_DB.png">

##### Iris dataset
```python
plot_decision_boundary(data = training_data, x_var = "sepal length (cm)", y_var = "petal length (cm)",target_variable = "class")
```
<img src = ".\\picture\\NB_iris.png">


- __Logistic Regression__ 
```python
lr = logistic_regression(num_iter = 10000, learning_rate = 0.001, fit_intercept = True)
lr.training(x = training_x, y = training_y)
pred = lr.predict(testing_x = testing_x, probability = False)
```

<img src = ".\\picture\\lr_DecBoun.png">