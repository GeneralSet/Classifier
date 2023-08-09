import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

## Load
data = joblib.load('train/data.pkl')

print('number of samples: ', len(data['data']))
print('labels:', np.unique(data['label']))

X = np.array(data['data'])
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)

# TRAIN
classifier = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
classifier.fit(X_train, y_train)

# TEST
y_pred = classifier.predict(X_test)
print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

# EXPORT
joblib.dump(classifier, 'app/classifier.pkl')
