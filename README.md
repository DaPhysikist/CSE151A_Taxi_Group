# CSE 151A - Taxi ML Group

# Milestone 3

The conclusion of our first model was that:
    - Using a Polynomial Regression model was more accurate than a Linear Regression model. This is becasue we were able to make the model more complexy by increasing the complexity of our "function" on the graph.
    - Our model's R2 correlation coefficient is decent for predicting fare price, but terrible for duration. We are not sure why. We can probably increase the polynomial degree but that may overfit the model.
    - We can improve it with more data, and possibly rounding the latitude and longitude to more decimals to ensure that we have more accuracy.
    - We notes that the decimal rounding was already quite accurate, with each point about less than 1-2 mi between each point in a lattice.

### Notebooks:
- Preprocessing in preprocess.ipynb
- Model in model.ipynb

# Milestone 4 

## Methods

To improve our model from polynomial regression, we opted to create a ANN with three layers. 

### Fare Price Calculation

For the fare model, we reduced our Test MSE from 15.43 (polynomial regression) to 9.571 (ANN), and we also increased our R-squared score from 0.782 (polynomial regression) to 0.869. 

### Duration Calculation

For the duration model, we also tried a ANN, but our model performed poorly, with an R-squared value of close to 0. After further testing, we believe that a neural network is too complex of a model to accurately represent the duration prediction. So, we decided to try a decision tree model instead. A decision tree regressor splits the data into subsets based on feature values. This model worked much better, reducing our Test MSE to 48118, and increasing our R-squared value to 0.472. 

## Model Evaluation

For this milestone, we first implemented a neural network to predict the trip fare and duration, before implementing a decision tree model. The new decision tree model produced the largest improvement over our previous models . 

Our training versus test error for both the trip fare and trip duration prediction models has improved from the prior models we used. Initially with our first model, a Polynomial Regression model, our trip fare training data Mean Squared Error (MSE) was 14.398 and test data was 15.43, which is a 7.2% percent difference between the two. This indicates that our model generalized to the data relatively well, although with some signs of slight overfitting. With the newer neural network model applied, the train MSE lowered to 8.789 and the test MSE dropped to 9.571, which is a 8.9% difference between the two. The overall MSE had decreased, but at the cost of a larger difference between the training and test error due to an increase in model complexity. In addition, another metric we used to compare the regression models for trip fare is the R² score, or the coefficient of determination, which represents the proportion of the variance in the target of the model that is explained by the predictors in the model, and is used to evaluate how well the model fits the data. For the first model the R2 score was 0.782 and for the second model it was 0.869, which is a good improvement, as the highest possible R2 score is 1.0, and a higher R2 score means that our model explains the variance in possible taxi fare data better, with about 86.9% of variance in taxi fare data covered by our model. We hypothesize that achieving a higher R2 value would require a lot more data with many more features. In particular, time of day, weekday/weekend (Sunday, Monday, etc.) so that the model can generalize workdays, holidays, and other events that would cause traffic congestion. This in turn would require a more complex neural network.

For trip duration, with our first model (Polynomial Regression), our train data MSE was 32765.832 and our test data MSE was 32733.465, which is a  0.1% difference between the two, and the test MSE was lower than the train MSE, which might have been a result of underfitting. We noticed that our R2 value was very bad, and we struggled to create a new model that performed better.  We tried a Neural Network approach, similar to what we had implemented to predict the fares, but the results were poor (sometimes even having a negative R2 value). We decided that we had to create a more sophisticated model. 

After some iteration, we landed on a decision tree approach. Our new model resulted in a train data MSE of 47754.426 and a test data MSE of 48118.063, which is a 0.8% difference between the two. There was an increase in the difference between the train and test MSE due to an increase in the model complexity, but one aspect of our model’s performance improved significantly, the R2 score. The R2 score for the polynomial regression duration model was 0.343, while the R2 score for the Decision Tree model was 0.472, which is a significant improvement. These two metrics of MSE and R2 also apply to question 6 for evaluating our model performance, since we used a regression model instead of a classification model.

## Fitting Graph

Overall, our trip fare model has increased in model complexity, placing it further out on the fitting graph in terms of model complexity than the first model, but predictive error has also decreased, which lines up with the curve of the fitting graph. Since the difference in predictive error between the training and test data is relatively low at 8.9% and the overall predictive error of both training and test data is lower, the model’s complexity is just right and the model still falls within the ideal range for model complexity. Our trip duration model has also increased in model complexity and falls further out on the fitting graph than the first model, since it has progressed past the underfitting stage. The difference between the predictive error of the training and test data is quite low at 0.4%, but the predictive errors for each are still large values, which can definitely be improved further. This means that the model complexity needs to be increased, and the current model falls in the simple model section of the fitting graph, to the left of the ideal range of model complexity.

## Future Models to Try

The next model we are thinking of using is some kind of gradient boosting model, such as a XGBoost, since these types of models are good for handling data with lots of noise/outliers. We could also try using K-Nearest Neighbors - while this model is normally used for classification, a variation of it can also be used for regression, especially as the data has many dimensions that all contribute to how long a taxi ride can be. In particular, a KNN model might be able to generalize certain parts of NYC (with the latitude-longitude) as having more traffic and thus would have higher fares. In particular, the clusters can be intervals of trip duration, although this would imply that we would be trying to classify the duration and not necessarily predict a continuous value for the trip duration. One other issue we came across is that some of our models would predict a negative duration, which is impossible. We tried to use different activation functions in the Neural Network to try and eliminate this. In particular, we used the relu activation function in parts of the model to try and rectify this. We would also try a random forest model, which is similar to our decision tree model, but better at avoiding noise and outliers. A random forest model is also better at avoiding overfitting, something which decision trees can sometimes suffer from. Especially if we try to improve our model by obtaining data about more features, we could then use feature importance to prioritize features.

## Conclusion

**Fare:**

When predicting fare price, the neural network performed well, with a R2 value of 0.869. However, this neural network performed poorly when trying to predict trip duration, with an R2 value of -0.005, indicating that this model performed worse than just predicting the mean. We believe this was because the neural network was too simple to predict the duration we were looking for with the data we were using. We decided to explore other models for predicting duration for this reason. 
    
For our neural network, we could try to add some more layers. For now, it's a relatively simple neural network with only three dense layers. After adding more layers, we can add some dropout layers to ensure the model is able to propagate the loss properly through the model. We could also use a different optimizer, more high quality data, and more epochs.
We also tried to do hyperparameter search to optimize the neural network, but we found that no matter what, we could not break the 0.869 R2 value. We propose that higher quality data could be added, and that we could try to get more layers for the neural network to get it to generalize to the data better. One could argue that this might be the computational limit already as we will eventually asymptote and plateau performance as we approach 1.0, a theoretical perfect performance. Thus, we would get severe diminishing returns with more data, compute, and complexity.


**Duration:**

We found duration to be quite difficult to model. We believe this is because duration has a lot of random factors which can affect it while fare is most likely determined by distance travelled and time of the day. We think the random factors which affect the duration could have included traffic, rush hour, large events, etc. A decision tree would have an easier time generalizing for these random factors. As such, we tried a decision tree, and achieved a much better R2 value of 0.472.
    
For our decision tree model, we can improve it by tuning hyperparameters like max_depth, min_samples_split, and min_samples_leaf. We also employed feature engineering, especially in the Haversine distance function which ensured that the distances actually were able to convert from geographical location to a measurable distance the model could understand. With better feature engineering and more high quality data, we can enhance its performance and better measure it with cross-validation.

Another improvement which we could have tried to improve our model would have been to use gradient boosting. This is because a gradient boosting would have used more than one tree to combine predictions and find more complex relationships in the data. Gradient boosting would theoretically work well with our data since it is better able to work with categorical and numeric data due to its ability to find the features which matter the most.
