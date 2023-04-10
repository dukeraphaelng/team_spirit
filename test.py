import os
from svm import svm_model
import pandas as pd

path_to_data = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test.csv"
)

# Load data
test_data = pd.read_csv(path_to_data)
# Get the first 5 texts and authors
test_list = test_data.text.tolist()
test_label = test_data.author.tolist()
test_list = test_list[0:5]
test_label = test_label[0:5]

svm_classifier = svm_model()

# Turn the labels into numbers
test_label = svm_classifier.label_data(test_label)

# Print the actual labels
print(test_label)
# Print the predicted labels
print(svm_classifier.predict(test_list))
# Print the probability distribution
print(svm_classifier.predict_proba(test_list))







