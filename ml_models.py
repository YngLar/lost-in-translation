import pandas as pd
import re
import argparse
from sklearn.metrics import classification_report
import os
from html import unescape
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed

# Define the model configuration
NUM_EPOCHS = 8
BATCH_SIZE = 16
PREPROCESS = True
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01                   

### NATIVE DATA PATHS ###

#Spanish dataset paths
TRAIN_ES_PATH = "./spanish/train_spanish.csv"
TEST_ES_PATH = "./spanish/test_spanish.csv"
#Spanish dataset paths (LIWC GROUPED)
TRAIN_ES_LIWC_GROUPED_PATH = "./spanish/train_spanish_grouped_liwc.csv"
TEST_ES_LIWC_GROUPED_PATH = "./spanish/test_spanish_grouped_liwc.csv"
#RSDD dataset paths
TRAIN_EN_PATH = "./RSDD/filtered/rsdd_train_filtered.csv"
TEST_EN_PATH = "./RSDD/filtered/rsdd_test_filtered.csv"
#TRAIN_EN_PATH = "./RSDD/training.csv"
#TEST_EN_PATH = "./RSDD/testing.csv"
#RSDD dataset paths (LIWC GROUPED)
TRAIN_EN_LIWC_GROUPED_PATH = "./RSDD/filtered/train_grouped_liwc.csv"
TEST_EN_LIWC_GROUPED_PATH = "./RSDD/filtered/test_grouped_liwc.csv"

### TRANSLATED DATA PATHS ###

## ES_TRANSLATED ##
TRAIN_ES_TRANSLATED_PATH = "./spanish/translated/training.csv"
TEST_ES_TRANSLATED_PATH = "./spanish/translated/testing.csv"
# LIWC GROUPED
TRAIN_ES_TRANSLATED_LIWC_GROUPED_PATH = "./spanish/translated/train_grouped_liwc.csv"
TEST_ES_TRANSLATED_LIWC_GROUPED_PATH = "./spanish/translated/test_grouped_liwc.csv"

## EN_TRANSLATED ##
TRAIN_EN_TRANSLATED_PATH = "./RSDD/translated/training.csv"
TEST_EN_TRANSLATED_PATH = "./RSDD/translated/testing.csv"
# LIWC GROUPED
TRAIN_EN_TRANSLATED_LIWC_GROUPED_PATH = "./RSDD/translated/train_grouped_liwc.csv"
TEST_EN_TRANSLATED_LIWC_GROUPED_PATH = "./RSDD/translated/test_grouped_liwc.csv"

stop_words_en = set(stopwords.words('english'))
stop_words_es =set(stopwords.words('spanish'))

def preprocess_text_en(text):
    text = str(text)
    # Decode HTML entities
    text = unescape(text)
    
    # Replace URLs and numbers with specific tokens
    text = re.sub(r'https?://\S+|www\.\S+', 'weblink', text)
    text = re.sub(r'\d+', 'number', text)
    
    # Replace newlines and tabs with a space
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # Remove non-word characters and convert to lowercase
    text = re.sub(r'\W+', ' ', text).lower()
    
    # Tokenize the cleaned text
    tokens = text.split()
    
    # Filter out stop words
    tokens = [x for x in tokens if x not in stop_words_en]
    
    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def preprocess_text_es(text):
    text = str(text)

    # Decode HTML entities
    text = unescape(text)
    
    # Replace URLs and numbers with specific tokens
    text = re.sub(r'https?://\S+|www\.\S+', 'weblink', text)
    text = re.sub(r'\d+', 'number', text)
    
    # Replace newlines and tabs with a space
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # Remove non-word characters and convert to lowercase
    text = re.sub(r'\W+', ' ', text).lower()
    
    # Tokenize the cleaned text
    tokens = text.split()
    
    # Filter out stop words
    tokens = [x for x in tokens if x not in stop_words_es]
    
    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def group_df_by_user(df):
    # Data Preparation: Aggregate text per user
    df_grouped = df.groupby('id')['text'].apply(lambda texts: ' '.join(str(text) for text in texts)).reset_index()
    df_labels = df.groupby('id')['label'].first().reset_index()
    df_final = pd.merge(df_grouped, df_labels, on='id')
    return df_final


def find_best_feature(x):
    x_copy = x.drop(x.columns[:2], axis=1)
    correlations = x_copy.corr()['label'].sort_values()
    # create a list of features that are either positively or negatively correlated, with a threshold of 0.15
    features = []
    for i in range(len(correlations)):
        if correlations[i] > 0.15 or correlations[i] < -0.15:
            features.append(correlations.index[i])
    features.remove('label')
    # print features sorted by correlation
    print(correlations[correlations > 0.15])
    print(correlations[correlations < -0.15])
    return features

def remove_non_correlated_features(x, features):
    x = x[['text', 'label'] + features]
    return x
    

 
def create_dataset(train, test, LIWC):
    tfidf = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for efficiency
    if LIWC:
        features = find_best_feature(train)
        remove_non_correlated_features(train, features)
        remove_non_correlated_features(test, features)
        X_train = train.copy()
        X_train_text = tfidf.fit_transform(train['text'])
        X_train_text_df = pd.DataFrame(X_train_text.toarray(), columns=[f"tfidf_{i}" for i in range(X_train_text.shape[1])])
        X_train_LIWC = train[features]
        X_train = pd.concat([X_train_LIWC, X_train_text_df], axis=1)
        y_train = train['label']

        X_test = test.copy()
        X_test_text = tfidf.transform(test['text'])
        X_test_text_df = pd.DataFrame(X_test_text.toarray(), columns=[f"tfidf_{i}" for i in range(X_test_text.shape[1])])
        X_test_LIWC = test[features]
        X_test = pd.concat([X_test_LIWC, X_test_text_df], axis=1)
        y_test = test['label']
    
    else:
        X_train = tfidf.fit_transform(train['text']).toarray()
        y_train = train['label']

        X_test = tfidf.transform(test['text']).toarray()
        y_test = test['label']

    # print the shape of the datasets
    print(X_train.shape, y_train.shape)

    return X_train, y_train, X_test, y_test
 
def fix_df_types(df):
    df = df.dropna(subset=['text'])
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str.lower()

    if pd.api.types.is_integer_dtype(df["id"]):
        print("Column is already an integer type.")
    else:
        # Extract digits and convert to integer if the column is not already integer
        df["id"] = df["id"].astype(str).str.extract('(\d+)').astype(int)
    return df


# DATASET COMBINATIONS
# EN
# EN_LIWC
# EN+ES_translated
# EN+ES_translated_LIWC
# ES_translated
# ES_translated_LIWC

# ES
# ES_LIWC
# ES+EN_translated
# ES+EN_translated_LIWC
# EN_translated
# EN_translated_LIWC
def load_datasets(train_dataset, test_dataset):
    ### TRAIN ###
    if train_dataset == "EN":
        train_df = pd.read_csv(TRAIN_EN_PATH)
        train_df = fix_df_types(train_df)
        if PREPROCESS:
            train_df['text'] = train_df['text'].apply(preprocess_text_en)
        train_df = group_df_by_user(train_df)
    elif train_dataset == "EN_LIWC":
        train_df = pd.read_csv(TRAIN_EN_LIWC_GROUPED_PATH)
        train_df = fix_df_types(train_df)
        if PREPROCESS:
            train_df['text'] = train_df['text'].apply(preprocess_text_en)
    elif train_dataset == "EN+ES_translated":
        #native
        train_df_1 = pd.read_csv(TRAIN_EN_PATH)
        if PREPROCESS:
            train_df_1['text'] = train_df_1['text'].apply(preprocess_text_en)
        train_df_1 = fix_df_types(train_df_1)
        train_df_1 = group_df_by_user(train_df_1)
        #translated
        train_df_2 = pd.read_csv(TRAIN_ES_TRANSLATED_PATH)
        if PREPROCESS:
            train_df_2['text'] = train_df_2['text'].apply(preprocess_text_en)
        train_df_2 = fix_df_types(train_df_2)
        train_df_2 = group_df_by_user(train_df_2)
        #combine
        train_df = pd.concat([train_df_1, train_df_2], ignore_index=True)
    elif train_dataset == "EN+ES_translated_LIWC":
        #native
        train_df_1 = pd.read_csv(TRAIN_EN_LIWC_GROUPED_PATH)
        train_df_1 = fix_df_types(train_df_1)
        if PREPROCESS:
            train_df_1['text'] = train_df_1['text'].apply(preprocess_text_en)
        #translated
        train_df_2 = pd.read_csv(TRAIN_ES_TRANSLATED_LIWC_GROUPED_PATH)
        train_df_2 = fix_df_types(train_df_2)
        if PREPROCESS:
            train_df_2['text'] = train_df_2['text'].apply(preprocess_text_en)
        #combine
        train_df = pd.concat([train_df_1, train_df_2], ignore_index=True)
    elif train_dataset == "ES":
        train_df = pd.read_csv(TRAIN_ES_PATH)
        train_df = fix_df_types(train_df)
        if PREPROCESS:
            train_df['text'] = train_df['text'].apply(preprocess_text_es)
        train_df = group_df_by_user(train_df)
    elif train_dataset == "ES_LIWC":
        train_df = pd.read_csv(TRAIN_ES_LIWC_GROUPED_PATH)
        train_df = fix_df_types(train_df)
        if PREPROCESS:
            train_df['text'] = train_df['text'].apply(preprocess_text_es)
    elif train_dataset == "ES+EN_translated":
        #native
        train_df_1 = pd.read_csv(TRAIN_ES_PATH)
        if PREPROCESS:
            train_df_1['text'] = train_df_1['text'].apply(preprocess_text_es)
        train_df_1 = fix_df_types(train_df_1)
        train_df_1 = group_df_by_user(train_df_1)
        #translated
        train_df_2 = pd.read_csv(TRAIN_EN_TRANSLATED_PATH)
        if PREPROCESS:
            train_df_2['text'] = train_df_2['text'].apply(preprocess_text_es)
        train_df_2 = fix_df_types(train_df_2)
        train_df_2 = group_df_by_user(train_df_2)
        #combine
        train_df = pd.concat([train_df_1, train_df_2], ignore_index=True)
    elif train_dataset == "ES+EN_translated_LIWC":
        #native
        train_df_1 = pd.read_csv(TRAIN_ES_LIWC_GROUPED_PATH)
        train_df_1 = fix_df_types(train_df_1)
        if PREPROCESS:
            train_df_1['text'] = train_df_1['text'].apply(preprocess_text_es)
        #translated
        train_df_2 = pd.read_csv(TRAIN_EN_TRANSLATED_LIWC_GROUPED_PATH)
        train_df_2 = fix_df_types(train_df_2)
        if PREPROCESS:
            train_df_2['text'] = train_df_2['text'].apply(preprocess_text_es)
        #combine
        train_df = pd.concat([train_df_1, train_df_2], ignore_index=True)
    elif train_dataset == "EN_translated":
        train_df = pd.read_csv(TRAIN_EN_TRANSLATED_PATH)
        if PREPROCESS:
            train_df['text'] = train_df['text'].apply(preprocess_text_en)
        train_df = group_df_by_user(train_df)
    elif train_dataset == "EN_translated_LIWC":
        train_df = pd.read_csv(TRAIN_EN_TRANSLATED_LIWC_GROUPED_PATH)
        train_df = fix_df_types(train_df)
        if PREPROCESS:
            train_df['text'] = train_df['text'].apply(preprocess_text_en)
    elif train_dataset == "ES_translated":
        train_df = pd.read_csv(TRAIN_ES_TRANSLATED_PATH)
        train_df = fix_df_types(train_df)
        if PREPROCESS:
            train_df['text'] = train_df['text'].apply(preprocess_text_en)
        train_df = group_df_by_user(train_df)
    elif train_dataset == "ES_translated_LIWC":
        train_df = pd.read_csv(TRAIN_ES_TRANSLATED_LIWC_GROUPED_PATH)
        train_df = fix_df_types(train_df)
        if PREPROCESS:
            train_df['text'] = train_df['text'].apply(preprocess_text_en)
    
    ### TEST ###
    if test_dataset == "EN":
        test_df = pd.read_csv(TEST_EN_PATH)
        test_df = fix_df_types(test_df)
        if PREPROCESS:
            test_df['text'] = test_df['text'].apply(preprocess_text_en)
        test_df = group_df_by_user(test_df)
    elif test_dataset == "EN_LIWC":
        test_df = pd.read_csv(TEST_EN_LIWC_GROUPED_PATH)
        test_df = fix_df_types(test_df)
        if PREPROCESS:
            test_df['text'] = test_df['text'].apply(preprocess_text_en)
    elif test_dataset == "EN+ES_translated":
        #native
        test_df_1 = pd.read_csv(TEST_EN_PATH)
        if PREPROCESS:
            test_df_1['text'] = test_df_1['text'].apply(preprocess_text_en)
        test_df_1 = fix_df_types(test_df_1)
        test_df_1 = group_df_by_user(test_df_1)
        #translated
        test_df_2 = pd.read_csv(TEST_ES_TRANSLATED_PATH)
        if PREPROCESS:
            test_df_2['text'] = test_df_2['text'].apply(preprocess_text_en)
        test_df_2 = fix_df_types(test_df_2)
        test_df_2 = group_df_by_user(test_df_2)
        #combine
        test_df = pd.concat([test_df_1, test_df_2], ignore_index=True)
    elif test_dataset == "EN+ES_translated_LIWC":
        #native
        test_df_1 = pd.read_csv(TEST_EN_LIWC_GROUPED_PATH)
        test_df_1 = fix_df_types(test_df_1)
        if PREPROCESS:
            test_df_1['text'] = test_df_1['text'].apply(preprocess_text_en)
        #translated
        test_df_2 = pd.read_csv(TEST_ES_TRANSLATED_LIWC_GROUPED_PATH)
        test_df_2 = fix_df_types(test_df_2)
        if PREPROCESS:
            test_df_2['text'] = test_df_2['text'].apply(preprocess_text_en)
        #combine
        test_df = pd.concat([test_df_1, test_df_2], ignore_index=True)
    elif test_dataset == "ES":
        test_df = pd.read_csv(TEST_ES_PATH)
        test_df = fix_df_types(test_df)
        if PREPROCESS:
            test_df['text'] = test_df['text'].apply(preprocess_text_es)
        test_df = group_df_by_user(test_df)
    elif test_dataset == "ES_LIWC":
        test_df = pd.read_csv(TEST_ES_LIWC_GROUPED_PATH)
        test_df = fix_df_types(test_df)
        if PREPROCESS:
            test_df['text'] = test_df['text'].apply(preprocess_text_es)
    elif test_dataset == "ES+EN_translated":
        #native
        test_df_1 = pd.read_csv(TEST_ES_PATH)
        if PREPROCESS:
            test_df_1['text'] = test_df_1['text'].apply(preprocess_text_es)
        test_df_1 = fix_df_types(test_df_1)
        test_df_1 = group_df_by_user(test_df_1)
        #translated
        test_df_2 = pd.read_csv(TEST_EN_TRANSLATED_PATH)
        if PREPROCESS:
            test_df_2['text'] = test_df_2['text'].apply(preprocess_text_es)
        test_df_2 = fix_df_types(test_df_2)
        test_df_2 = group_df_by_user(test_df_2)
        #combine
        test_df = pd.concat([test_df_1, test_df_2], ignore_index=True)
    elif test_dataset == "ES+EN_translated_LIWC":
        #native
        test_df_1 = pd.read_csv(TEST_ES_LIWC_GROUPED_PATH)
        test_df_1 = fix_df_types(test_df_1)
        if PREPROCESS:
            test_df_1['text'] = test_df_1['text'].apply(preprocess_text_es)
        #translated
        test_df_2 = pd.read_csv(TEST_EN_TRANSLATED_LIWC_GROUPED_PATH)
        test_df_2 = fix_df_types(test_df_2)
        if PREPROCESS:
            test_df_2['text'] = test_df_2['text'].apply(preprocess_text_es)
        #combine
        test_df = pd.concat([test_df_1, test_df_2], ignore_index=True)
    elif test_dataset == "EN_translated":
        test_df = pd.read_csv(TEST_EN_TRANSLATED_PATH)
        test_df = fix_df_types(test_df)
        if PREPROCESS:
            test_df['text'] = test_df['text'].apply(preprocess_text_en)
        test_df = group_df_by_user(test_df)
    elif test_dataset == "EN_translated_LIWC":
        test_df = pd.read_csv(TEST_EN_TRANSLATED_LIWC_GROUPED_PATH)
        test_df = fix_df_types(test_df)
        if PREPROCESS:
            test_df['text'] = test_df['text'].apply(preprocess_text_en)
    elif test_dataset == "ES_translated":
        test_df = pd.read_csv(TEST_ES_TRANSLATED_PATH)
        test_df = fix_df_types(test_df)
        if PREPROCESS:
            test_df['text'] = test_df['text'].apply(preprocess_text_en)
        test_df = group_df_by_user(test_df)
    elif test_dataset == "ES_translated_LIWC":
        test_df = pd.read_csv(TEST_ES_TRANSLATED_LIWC_GROUPED_PATH)
        test_df = fix_df_types(test_df)
        if PREPROCESS:
            test_df['text'] = test_df['text'].apply(preprocess_text_en)
    return train_df, test_df


def train_and_tune(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_model(X_train, y_train, model_type="ALL"):

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(random_state=42)),
        "Naive Bayes": MultinomialNB(),
        "XGBoost": XGBClassifier(random_state=42),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(random_state=42, max_iter=1000))
    }
    param_grid = {
        "Logistic Regression": {'logisticregression__C': [0.01, 0.1, 1, 10, 100], 'logisticregression__penalty': ['l2'], 'logisticregression__solver': ['lbfgs', 'saga']},
        "Random Forest": {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30, 40]},
        "Support Vector Machine": {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']},
        "Naive Bayes": {'alpha': [0.01, 0.1, 1.0]},
        "XGBoost": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        "MLP": {'mlpclassifier__alpha': [0.0001, 0.001, 0.01]}
    }
    # Prepare list of models to be trained based on model_type
    tasks = [(model, param_grid[name], X_train, y_train) for name, model in models.items() if model_type == "ALL" or model_type == name]
    
    # Use Joblib to train models in parallel
    tuned_models = Parallel(n_jobs=-1)(delayed(train_and_tune)(*task) for task in tasks)

    # Collect the results in a dictionary
    return {name: model for name, model in zip(models.keys(), tuned_models) if model_type == "ALL" or model_type == name}

def make_predictions(trained_models, X_test, y_test):
    # Make predictions
    predictions = {}
    for name, model in trained_models.items():
        print(f"Making predictions with {name}")
        predictions[name] = model.predict(X_test)
        
    # Generate classification reports
    reports = {}
    for name, preds in predictions.items():
        reports[name] = classification_report(y_test, preds)
        print(f"Classification report for {name}:\n{reports[name]}")
    
    return reports

def write_reports_to_file(reports, train, test):
    # Write classification reports to file
    directory = "./classification_reports/ml/kjoh"  # Current directory, change as needed
    base_filename = "report"
    file_extension = ".txt"
    next_file_number = 1
    while os.path.exists(f"{directory}/{base_filename}_{next_file_number}{file_extension}"):
        next_file_number += 1
    full_path = f"{directory}/{base_filename}_{next_file_number}{file_extension}"
    with open(full_path, 'w') as f:
        for name, report in reports.items():
            f.write(f"Model: {name}\n")
            f.write(f"Train dataset: {train}\n")
            f.write(f"Test dataset: {test}\n")         
            f.write("--------------------------------------------------\n")
            f.write(report)
            f.write("\n \n")
    print(f"Classification report saved to {full_path}")

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    # Adding named command line arguments
    parser.add_argument('--train', type=str, help='Enter train dataset', required=True)
    parser.add_argument('--test', type=str, help='Enter test dataset', required=True)
    args = parser.parse_args()

    IS_LIWC = False
    # If liwc is in the name of the dataset, set the flag to true
    if "LIWC" in args.train:
        IS_LIWC = True

    print(f"Starting dataset creation with TRAIN:{args.train}, TEST:{args.test}")

    # Load data from csv
    train_df, test_df = load_datasets(args.train, args.test)

    # Tokenize with tf-idf
    X_train, y_train, X_test, y_test = create_dataset(train_df, test_df, LIWC=IS_LIWC)

    # Train the models
    trained_model = train_model(X_train, y_train)

    #Make predictions and write them to file
    reports = make_predictions(trained_model, X_test, y_test)

    # Write classification reports to file
    write_reports_to_file(reports, args.train, args.test)

main()


