import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)
    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            evidence.append(dataConversion(row[:17], 'evidence'))
            labels.append(dataConversion(row[17], 'label'))
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positives = labels.count(1)
    negatives = labels.count(0)
    correctPositive = 0
    correctNegative = 0
    for actual, predicted in zip(labels, predictions):
        if actual == predicted:
            if actual == 1:
                correctPositive += 1
            else:
                correctNegative += 1
    sensitivity = correctPositive / positives
    specificity = correctNegative / negatives
    return (sensitivity,  specificity)

def dataConversion(row, dataType):
    convertedData = []
    if dataType == "evidence":
        convertedData.append(int(row[0]))
        convertedData.append(float(row[1]))
        convertedData.append(int(row[2]))
        convertedData.append(float(row[3]))
        convertedData.append(int(row[4]))
        convertedData.append(float(row[5]))
        convertedData.append(float(row[6]))
        convertedData.append(float(row[7]))
        convertedData.append(float(row[8]))
        convertedData.append(float(row[9]))
        if row[10] == "Jan":
            convertedData.append(0)
        if row[10] == "Feb":
            convertedData.append(1)
        if row[10] == "Mar":
            convertedData.append(2)
        if row[10] == "Apr":
            convertedData.append(3)
        if row[10] == "May":
            convertedData.append(4)
        if row[10] == "June":
            convertedData.append(5)
        if row[10] == "Jul":
            convertedData.append(6)
        if row[10] == "Aug":
            convertedData.append(7)
        if row[10] == "Sep":
            convertedData.append(8)
        if row[10] == "Oct":
            convertedData.append(9)  
        if row[10] == "Nov":
            convertedData.append(10)
        if row[10] == "Dec":
            convertedData.append(11)  
        convertedData.append(int(row[11]))
        convertedData.append(int(row[12]))
        convertedData.append(int(row[13]))
        convertedData.append(int(row[14]))
        if row[15] == "Returning_Visitor":
            convertedData.append(1)
        else:
            convertedData.append(0)
        if row[16] == 'TRUE':
            convertedData.append(1)
        else:
            convertedData.append(0)
        return convertedData
    elif dataType == "label":
        if row == "TRUE":
            return 1
        else:
            return 0

if __name__ == "__main__":
    main()
