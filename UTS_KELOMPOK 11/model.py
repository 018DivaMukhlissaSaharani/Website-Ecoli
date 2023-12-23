# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.decomposition import PCA
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.pipeline import make_pipeline

# import numpy as np

# # Load The CSV File
# data = pd.read_csv("glass.csv")

# data.head()

# # Select Independent & Dependent Variabel
# X = data.drop(columns="Type", axis=1)
# Y = data["Type"]

# # Random Sampling
# resamp = RandomOverSampler()
# balX, balY = resamp.fit_resample(X, Y)

# # Data Normalization
# scaler = StandardScaler()
# standardized = scaler.fit_transform(balX)

# # Data Dimension Reduction
# n_components = 9  # Ganti dengan jumlah komponen yang diinginkan
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(standardized)

# # Create Decision Tree classifer object
# clf = DecisionTreeClassifier()

# # Tentukan parameter yang akan diuji dalam Grid Search
# param_grid = {'criterion':['gini', 'entropy', 'log_loss'], 'max_depth':np.arange(1,10)}

# # Inisialisasi Grid Search
# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='accuracy')
# # Latih model dengan data pelatihan
# grid_search.fit(X_pca, balY)

# print("Best Parameters (Accuracy):", grid_search.best_params_)
# print("Best Score (Accuracy):", grid_search.best_score_)




import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("ecoli.csv")
data.head()

del data['Sequence Name']
data.head()

# Drop missing values
data = data.dropna()

# Split data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Support Vector Machine Classifier
svm_classifier = SVC(kernel='linear')  # You can change the kernel if needed (e.g., 'rbf', 'poly', etc.)

# Fit the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Generate classification report
report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')

# Make pickle file of our model
pickle.dump(svm_classifier, open("model2.pkl", "wb"))