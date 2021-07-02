import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

class selftraining():
    def __init__(self, model):
        """
            Define the model here. Directly put parameters for the model during initiating the class.
            E.g. clf = selftraining(LogisticRegression(max_iter=250))
        """
        self.clf = model
        self.scores = []
    
    def _add_data(self, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test):
        """
            Adding data to the object class.
        """
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.X_unlabeled = X_unlabeled
        self.y_unlabeled = y_unlabeled
        self.X_test = X_test
        self.y_test = y_test
    
    def build_vectorizer(self, text):
        """
            Building vectorizer from text sing TfidfVectorizer. If not using text, then do not need to build vectorizer.
        """
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(text)
    
    def transform(self, text):
        """
            Transforming text using TF-IDF.
        """
        vectorize = self.vectorizer.transform(text)
        return vectorize
    
    def fit(self, transform=False):
        """
            Fit classifier.
            Use transform=True if using text so it can be changed by tfidf.
        """
        X_labeled_vectorized = self.transform(self.X_labeled)
        if transform:
            X_labeled_vectorized = self.transform(self.X_labeled)
        self.clf.fit(X_labeled_vectorized, self.y_labeled)
    
    def predict_and_evaluate(self, transform=False):
        """
            Predict and evaluate the model. Evaluation metrics is accuracy.
            Use transform=True if using text so it can be changed by tfidf.
        """
        X_labeled_vectorized = self.transform(self.X_labeled)
        if transform:
            X_labeled_vectorized = self.transform(self.X_labeled)
        X_test_vectorized = self.transform(self.X_test)
        train_hat = self.clf.predict(X_labeled_vectorized)
        test_hat = self.clf.predict(X_test_vectorized)
        
        print('Model Evaluation : ')
        print(f'Classifier 1 accuracy score : ')
        print(f'\tTrain : {accuracy_score(train_hat, self.y_labeled)}')
        print(f'\tTest : {accuracy_score(test_hat, self.y_test)}')
        
        self.scores.append(accuracy_score(test_hat, self.y_test))
    
    def iterations(self, loops, threshold, transform=False):
        """
            Iterate through the unlabeled data. Input requires loop and the threshold. Number of loops will affect the pool of unlabeled data
            taken. Threshold will affect to which minimum probability will be taken from the unlabeled data.
        """
        self.pool = int(len(self.X_unlabeled) / loops)
        for i in range(loops):
            print(f'Iteration - {i}')
            self.find_index_to_add(threshold, self.pool, transform)
            self.fit(transform)
            self.predict_and_evaluate(transform)
        
        sns.lineplot(x=range(loops+1), y=self.scores)
    
    def find_index_to_add(self, threshold, pool, transform=False):
        X_unlabeled_vectorized = self.transform(self.X_unlabeled)
        if transform:
            X_unlabeled_vectorized = self.transform(self.X_unlabeled)
        train_proba = self.clf.predict_proba(X_unlabeled_vectorized)
        train_pred = self.clf.predict(X_unlabeled_vectorized)
        
        self.df = pd.DataFrame({
            'proba_0' : train_proba[:,0],
            'proba_1' : train_proba[:,1],
            'prediction' : train_pred
        })
        added_idx_from_clf = []
        counter = 0
        for i in range(len(self.df)):
            if self.df.iloc[i]['proba_0'] > threshold or self.df.iloc[i]['proba_1'] > threshold:
                added_idx_from_clf.append(i)
                counter += 1
                if counter >= pool:
                    break
        
        self.X_labeled = pd.concat([self.X_labeled, self.X_unlabeled.iloc[added_idx_from_clf]]).reset_index(drop=True)
        self.y_labeled = pd.concat([self.y_labeled, self.df.iloc[added_idx_from_clf, 2]])
        self.X_unlabeled.drop(added_idx_from_clf, inplace=True)
        self.X_unlabeled = self.X_unlabeled.sample(frac=1).reset_index(drop=True)

class Cotraining():
    def __init__(self, model1, model2):
        """
            Define the 2 models here. Directly put parameters for the model during initiating the class.
            E.g. clf = selftraining(LogisticRegression(max_iter=250), RandomForestClassifier(n_estimators=250))
        """
        self.h1 = model1
        self.h2 = model2
    
    def _add_data(self, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test):
        """
            Adding data to the object class.
        """
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.X_unlabeled = X_unlabeled
        self.y_unlabeled = y_unlabeled
        self.X_test = X_test
        self.y_test = y_test
    
        self.X_L1 = self.X_labeled
        self.y_L1 = self.y_labeled
        self.X_L2 = self.X_labeled
        self.y_L2 = self.y_labeled
        self.X_U = self.X_unlabeled
        self.y_U = self.y_unlabeled
    
    def build_vectorizer(self, text):
        """
            Building vectorizer from text sing TfidfVectorizer. If not using text, then do not need to build vectorizer.
        """
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(text)
    
    def transform(self, text):
        """
            Transforming text using TF-IDF.
        """
        vectorize = self.vectorizer.transform(text)
        return vectorize
    
    def fit(self, transform=False):
        """
            Fit classifier h1 and h2. 
            Use transform=True if using text so it can be changed by tfidf.
        """
        X_L1_vectorized = self.transform(self.X_L1)
        X_L2_vectorized = self.transform(self.X_L2)
        if transform:
            X_L1_vectorized = self.transform(self.X_L1)
            X_L2_vectorized = self.transform(self.X_L2)
        self.h1.fit(X_L1_vectorized, self.y_L1)
        self.h2.fit(X_L2_vectorized, self.y_L2)
    
    def predict_and_evaluate(self, transform=False):
        """
            Predict and evaluate the model. Evaluation metrics is accuracy.
            Use transform=True if using text so it can be changed by tfidf.
        """
        X_L1_vectorized = self.transform(self.X_L1)
        X_L2_vectorized = self.transform(self.X_L2)
        X_test_vectorized = self.transform(self.X_test)
        if transform:
            X_L1_vectorized = self.transform(self.X_L1)
            X_L2_vectorized = self.transform(self.X_L2)
            X_test_vectorized = self.transform(self.X_test)
        train1_hat = self.h1.predict(X_L1_vectorized)
        train2_hat = self.h2.predict(X_L2_vectorized)
        test1_hat = self.h1.predict(X_test_vectorized)
        test2_hat = self.h2.predict(X_test_vectorized)
        
        print('Model Evaluation : ')
        print(f'Classifier 1 f1 score : ')
        print(f'\tTrain : {accuracy_score(train1_hat, self.y_L1)}')
        print(f'\tTest : {accuracy_score(test1_hat, self.y_test)}')
        print(f'Classifier 2 f1 score : ')
        print(f'\tTrain : {accuracy_score(train2_hat, self.y_L2)}')
        print(f'\tTest : {accuracy_score(test2_hat, self.y_test)}')
    
    def iterations(self, loops, threshold, transform=False):
        """
            Iterate through the unlabeled data. Input requires loop and the threshold. Number of loops will affect the pool of unlabeled data
            taken. Threshold will affect to which minimum probability will be taken from the unlabeled data.
            Use transform=True if using text so it can be changed by tfidf
        """
        self.pool = len(self.X_unlabeled) / loops
        for i in range(loops):
            print(f'Iteration - {i}')
            self.find_index_to_add(threshold, self.pool, transform)
            self.fit(transform)
            self.predict_and_evaluate(transform)
    
    def find_index_to_add(self, threshold, pool, transform=False):
        X_unlabeled_vectorized = self.transform(self.X_unlabeled)
        if transform:
            X_unlabeled_vectorized = self.transform(self.X_unlabeled)
        train1_proba = self.h1.predict_proba(X_unlabeled_vectorized)
        train2_proba = self.h2.predict_proba(X_unlabeled_vectorized)
        train1_pred = self.h1.predict(X_unlabeled_vectorized)
        train2_pred = self.h2.predict(X_unlabeled_vectorized)
        
        df = pd.DataFrame({
            'proba_0' : train1_proba[:,0],
            'proba_1' : train1_proba[:,1],
            'prediction' : train1_pred
        })
        df2 = pd.DataFrame({
            'proba_0' : train2_proba[:,0],
            'proba_1' : train2_proba[:,1],
            'prediction' : train2_pred
        })
        
        added_idx_from_h1 = []
        added_idx_from_h2 = []
        counter = 0
        for i in range(len(df)):
            if self.df.iloc[i]['proba_0'] > threshold or self.df.iloc[i]['proba_1'] > threshold:
                added_idx_from_h1.append(i)
                counter += 1
                if counter >= pool/2:
                    break
        
        for i in range(len(df2)):
            if self.df2.iloc[i]['proba_0'] > threshold or self.df2.iloc[i]['proba_1'] > threshold:
                if i not in added_idx_from_h1:
                    added_idx_from_h2.append(i)
                    counter += 1
                    if counter >= pool/2:
                        break
        
        self.X_L1 = pd.concat([self.X_L1, self.X_unlabeled.iloc[added_idx_from_h1], self.X_unlabeled.iloc[added_idx_from_h2]]).reset_index(drop=True)
        self.y_L1 = pd.concat([self.y_L1, df.iloc[added_idx_from_h1, 2], df.iloc[added_idx_from_h2, 2]]).reset_index(drop=True)
        self.X_L2 = pd.concat([self.X_L2, self.X_unlabeled.iloc[added_idx_from_h1], self.X_unlabeled.iloc[added_idx_from_h2]]).reset_index(drop=True)
        self.y_L2 = pd.concat([self.y_L2, df2.iloc[added_idx_from_h1, 2], df2.iloc[added_idx_from_h2, 2]]).reset_index(drop=True)
        self.X_unlabeled.drop(added_idx_from_h1 + added_idx_from_h2, inplace=True)
        self.X_unlabeled = self.X_unlabeled.sample(frac=1).reset_index(drop=True, inplace=True)