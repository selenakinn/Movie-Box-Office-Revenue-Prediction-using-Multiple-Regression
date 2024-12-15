import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Movie data
data = {
    "Budget": [200000000, 185000000, 165000000, 150000000, 100000000],
    "Genre": [1, 1, 2, 2, 3],  # 1: Action, 2: Sci-Fi, 3: Drama
    "Year": [2019, 2018, 2014, 2010, 2008],
    "Revenue": [2797800564, 2048359754, 677471339, 829895144, 1004558444]  # Box office revenue
}

# Converting data to DataFrame
df = pd.DataFrame(data)

# Features (X) and target variable (y)
X = df[["Budget", "Genre", "Year"]]
y = df["Revenue"]

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training models
model = LinearRegression()
model.fit(X_train, y_train)

# Make a guess
y_pred = model.predict(X_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Revenue estimate for a new movie
new_movie = [[150000000, 1, 2022]]  # Budget: 150M, Genre: Action, Year: 2022
predicted_revenue = model.predict(new_movie)
print("Predicted Revenue for the new movie:", predicted_revenue[0])
