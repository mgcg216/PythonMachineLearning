import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.

    Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
    Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
    """
    selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
    return processed_features


def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.

    Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
    Returns:
    A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets

# For the training set, we'll choose the first 12000 examples, out of the total of 17000


training_examples = preprocess_features(california_housing_dataframe.head(12000))
print(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
print(training_targets.describe())

# For the validation set, we'll chosse the last 5000 examples, out of the total of 17000.


validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
print(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
print("validation_targets")
print(validation_targets.describe())

# Let's check our data against some baseline expectations:
#
# * For some values, like `median_house_value`, we can check to see if these values fall within reasonable ranges (
# keeping in mind this was 1990 data — not today!).
#
# * For other values, like `latitude` and `longitude`, we can do a quick check to see if these line up with expected
# values from a quick Google search.
#
# If you look closely, you may see some oddities:
#
# * `median_income` is on a scale from about 3 to 15. It's not at all clear what this scale refers to—looks like maybe
#  some log scale? It's not documented anywhere; all we can assume is that higher values correspond to higher income.
#
# * The maximum `median_house_value` is 500,001. This looks like an artificial cap of some kind.
#
# * Our `rooms_per_person` feature is generally on a sane scale, with a 75th percentile value of about 2. But there are
#  some very large values, like 18 or 55, which may show some amount of corruption in the data.
#
# We'll use these features as given for now. But hopefully these kinds of examples can help to build a little intuition
#  about how to check data that comes to you from an unknown source.

# Lets take a close look at two feature in particular: latitude and longitude. These are geographical coordinates of the
# city block in question. This might make a nice visualization - let's plot latitude and longitude, and use color to
# to show the median_house_value


def california_graph():
    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(validation_examples["longitude"],
                validation_examples["latitude"],
                cmap="coolwarm",
                c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

    ax = plt.subplot(1,2,2)
    ax.set_title("Training Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
    _ = plt.plot()

    plt.show()

# Looking at the tables of summary stats above, it's easy to wonder how anyone would do a useful data check. What's the
#  right 75th percentile value for total_rooms per city block?
#
# The key thing to notice is that for any given feature or column, the distribution of values between the train and
# validation splits should be roughly equal.
#
# The fact that this is not the case is a real worry, and shows that we likely have a fault in the way that our train
# and validation split was created.
#
# Take a look at how the data is randomized when it's read in.
#
# If we don't randomize the data properly before creating training and validation splits, then we may be in trouble if
#  the data is given to us in some sorted order, which appears to be the case here.

# Next, we'll train a linear regressor using all the features in the data set, and see how well we do.
# Let's define the same input function we've used previously for loading the data into the TensorFlow model


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# Because we're now working with mutiple input features, let's modularize our code for configuring feature columns into
# a seperate function

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
        input_features: The names of the numerical input features to use:
    Returns:
        A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


# Next, go ahead and complete the train_model() code below to set up the input functions and calculate predictions.
#
# NOTE: It's okay to reference the code from the previous exercises, but make sure to call predict() on the appropriate
# data sets.
#
# Compare the losses on training data and validation data. With a single raw feature, our best root mean squared error
# (RMSE) was of about 180.
#
# See how much better you can do now that we can use multiple features.
#
# Check the data using some of the methods we've looked at before. These might include:
#
# Comparing distributions of predictions and actual target values
#
# Creating a scatter plot of predictions vs. target values
#
# Creating two scatter plots of validation data using latitude and longitude:
#
# One plot mapping color to actual target median_house_value
# A second plot mapping color to predicted median_house_value for side-by-side comparison.

def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model of multiple features.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # 1. Create input functions.
    training_input_fn = lambda: my_input_fn(
        training_examples, training_targets["median_house_value"], batch_size=batch_size)# YOUR CODE HERE
    predict_training_input_fn = lambda: my_input_fn(
        training_examples, training_targets["median_house_value"], num_epochs=1, shuffle=False)# YOUR CODE HERE
    predict_validation_input_fn = lambda: my_input_fn(
        validation_examples, validation_targets["median_house_value"], num_epochs=1, shuffle=False)# YOUR CODE HERE

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # 2. Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)# YOUR CODE HERE
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)# YOUR CODE HERE
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return linear_regressor


linear_regressor = train_model(
    # TWEAK THESE VALUES TO SEE HOW MUCH YOU CAN IMPROVE THE RMSE
    learning_rate=0.00002,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

