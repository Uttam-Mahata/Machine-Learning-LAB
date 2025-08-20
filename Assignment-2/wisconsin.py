df = pd.read_csv('breast_cancer_wisconsin_diagnostic.csv')
print(df.head())
from sklearn.linear_model import LogisticRegression
model1 =  LogisticRegression(penalty='l2',solver='newton-cg', max_iter=1000)
model2 =  LogisticRegression(penalty=None,solver='newton-cg', max_iter=1000)
model3 =  LogisticRegression(penalty='l2',solver='lbfgs', max_iter=5000)
model4 =  LogisticRegression(penalty=None,solver='lbfgs', max_iter=10000)
model5 =  LogisticRegression(penalty='l1',solver='liblinear', max_iter=1000)
model6 =  LogisticRegression(penalty='l2',solver='liblinear', max_iter=1000)

from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=5)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=5)