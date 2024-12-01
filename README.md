# CSE 151A - Taxi ML Group

## Milestone 3

The conclusion of our first model was that:
    - Using a Polynomial Regression model was more accurate than a Linear Regression model. This is becasue we were able to make the model more complexy by increasing the complexity of our "function" on the graph.
    - Our model's R2 correlation coefficient is decent for predicting fare price, but terrible for duration. We are not sure why. We can probably increase the polynomial degree but that may overfit the model.
    - We can improve it with more data, and possibly rounding the latitude and longitude to more decimals to ensure that we have more accuracy.
    - We notes that the decimal rounding was already quite accurate, with each point about less than 1-2 mi between each point in a lattice.

### Notebooks:
- Preprocessing in preprocess.ipynb
- Model in model.ipynb

## Milestone 4 Methods

To improve our model from polynomial regression, we opted to create a ANN with three layers. 

### Fare Price Calculation

For the fare model, we reduced our Test MSE from 11.70 (polynomial regression) to 9.571 (ANN), and we also increased our R-squared score from 0.835 (polynomial regression) to 0.869. 

### Duration Calculation

For the duration model, we also tried a ANN, but our model performed poorly, with an R-squared value of close to 0. After further testing, we believe that a neural network is too ocmplex of a model to accurately represent the duration prediction.

So, we decided to try a decision tree model instead. [TODO: explain a little about this model]. This model worked much better, reducing our Test MSE to 48118, and increasing our R-squared value to 0.472. 

## Conclusion

For our neural network, we could try to add some more layers. For now, it's a relatively simple neural network with only three dense layers. We could consider adding more layers, such as a dropout layer.

For our decision tree model, [TODO: how can we improve it?].