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
