import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# One-hot encoding the DataFrame insurance
insurance_one_hot= pd.get_dummies(insurance)
insurance_one_hot.head()

# Create features and labels

X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

# Create training and test data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Make a model

# Set random seed
tf.random.set_seed(42)

insurance_model=tf.keras.Sequential([
                                       tf.keras.layers.Dense(100),
                                       tf.keras.layers.Dense(10),
                                       tf.keras.layers.Dense(1)
])
insurance_model.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])
history = insurance_model.fit(X_train, y_train, epochs=300, verbose=0)

#Evaluate the model

insurance_model.evaluate(X_test, y_test)


# Plot loss curve
pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel("loss")

