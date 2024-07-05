import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score


def load_data(file_path, delimiter=','):
    df = pd.read_csv(file_path, delimiter=delimiter)
    data = df.to_numpy()
    num_rows = data.shape[0]
    header_list = list(df.columns)
    return num_rows, data, header_list


def filter_data(data):
    # Replace -99 to NaN values
    data_with_nan = np.where(data == -99, np.nan, data)

    # Filter out any rows that contain NaN values
    filtered_data = data_with_nan[~np.isnan(data_with_nan).any(axis=1)]

    return filtered_data


def statistics_data(data):
    # Calculate means for each feature (column)
    means = np.nanmean(data, axis=0)

    # Calculate standard deviations for each feature (column)
    std_devs = np.nanstd(data, axis=0)

    # Calculate coefficient of variation for each feature, handling division by zero
    np.seterr(divide='ignore', invalid='ignore')
    coefficient_of_variation = np.where(means != 0, std_devs / means, 0)

    return coefficient_of_variation


def split_data(data, test_size=0.3, random_state=1):
    X = data[:, :-1]  # all columns except the last one
    y = data[:, -1]  # the last column

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_decision_tree(x_train, y_train, ccp_alpha=0):
    # Create the decision tree classifier with the specified ccp_alpha
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)

    # Fit model to training data
    clf.fit(x_train, y_train)

    return clf


def make_predictions(model, X_test):
    # Use trained model to predict labels for the test set
    y_test_predicted = model.predict(X_test)
    return y_test_predicted


def evaluate_model(model, x, y):
    # Predict labels for the dataset
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)

    return accuracy, recall


def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    # Train a decision tree model without pruning to find its accuracy
    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(x_train, y_train)
    unpruned_accuracy = accuracy_score(y_test, tree.predict(x_test))

    # Generate a range of ccp_alpha values to evaluate
    path = tree.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # Exclude the maximum alpha which results in the trivial tree

    # Evaluate accuracy for each ccp_alpha and stop when the drop exceeds 1% of unpruned accuracy
    optimal_alpha = 0  # Start with no pruning
    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
        tree.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, tree.predict(x_test))

        # Check if the drop in accuracy exceeds 1%
        if accuracy < unpruned_accuracy - 0.01:
            break  # Stop training as soon as the accuracy drops more than 1% from the unpruned accuracy
        optimal_alpha = ccp_alpha  # Update the last ccp_alpha that did not exceed the accuracy drop

    return optimal_alpha


def tree_depths(model):
    # Return the maximum depth of the decision tree model
    return model.get_depth()


def important_feature(x_train, y_train, header_list):
    # Start with a basic tree to establish the ccp_alpha range
    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(x_train, y_train)

    # Generate the sequence of ccp_alpha values
    ccp_path = tree.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas = ccp_path.ccp_alphas

    # Iterate through ccp_alphas to find the smallest alpha that reduces tree depth to 1
    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
        tree.fit(x_train, y_train)
        if tree.get_depth() == 1:
            # When the tree depth is 1, identify the remaining feature
            # tree.feature_importances_ shows the importance of each feature
            # the index of the highest value corresponds to the most important feature
            most_important_index = tree.feature_importances_.argmax()
            return header_list[most_important_index]


if __name__ == '__main__':
    # Load the data
    file_path = 'DT.csv'
    num_rows, data, headers = load_data(file_path)

    # Filter the data
    filtered_data = filter_data(data)

    # Calculate the coefficient of variation for each feature in the filtered data
    cv = statistics_data(filtered_data)

    print("Coefficient of Variation for each feature:")
    for header, variation in zip(headers, cv):
        print(f"{header}: {variation:.2f}")

    # Split the filtered data
    X_train, X_test, y_train, y_test = split_data(filtered_data)

    clf = train_decision_tree(X_train, y_train)

    # Predict the labels for the test set
    y_test_predicted = make_predictions(clf, X_test)

    print("Predicted labels:")
    print(y_test_predicted)

    most_important_feature = important_feature(X_train, y_train, headers)
    print(f"The most important feature is: {most_important_feature}")

    # Evaluate the model's performance on the test data
    accuracy, recall = evaluate_model(clf, X_test, y_test)

    # Print the evaluation results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")

    # Find the optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(X_train, y_train, X_test, y_test)

    # Print the optimal ccp_alpha
    print(f"Optimal CCP Alpha: {optimal_alpha}")

    # Train a new tree using the optimal ccp_alpha and evaluate it
    best_tree = DecisionTreeClassifier(ccp_alpha=optimal_alpha)
    best_tree.fit(X_train, y_train)
    best_tree_accuracy = accuracy_score(y_test, best_tree.predict(X_test))
    print(f"Accuracy of the best pruned tree: {best_tree_accuracy:.2f}")

    # Training a decision tree without pruning
    unpruned_tree = DecisionTreeClassifier(random_state=1)
    unpruned_tree.fit(X_train, y_train)

    # Training a decision tree with optimal ccp_alpha
    pruned_tree = DecisionTreeClassifier(random_state=1, ccp_alpha=optimal_alpha)
    pruned_tree.fit(X_train, y_train)

    # Getting the depth of each tree
    unpruned_depth = tree_depths(unpruned_tree)
    pruned_depth = tree_depths(pruned_tree)

    # Print the depth of each tree
    print(f"Depth of unpruned tree: {unpruned_depth}")
    print(f"Depth of pruned tree: {pruned_depth}")


# References
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://scikit-learn.org/stable/modules/tree.html
# https://towardsdatascience.com/decision-tree-build-prune-and-visualize-it-using-python-12ceee9af752
