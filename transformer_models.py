import pandas as pd
from transformers import BertTokenizer, Trainer, DataCollatorWithPadding, TrainingArguments
import torch
from torch import nn
import re
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset
import argparse
from sklearn.metrics import classification_report
import os
from tqdm import tqdm
import multiprocessing
from html import unescape
from nltk.corpus import stopwords
from transformers import AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.nn.functional import sigmoid


# Define the model configuration
NUM_EPOCHS = 50
BATCH_SIZE = 16
PREPROCESS = False
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1        
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

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

def tokenize_batch(texts, tokenizer):
    # Initialize the result storage
    batch_encodings = {"input_ids": [], "token_type_ids": [],"attention_mask": []}
 
    # Tokenize each text in the batch individually with tqdm for progress tracking
    for text in tqdm(texts, desc="Tokenizing"):
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512)
        batch_encodings['input_ids'].append(encoding['input_ids'])
        batch_encodings['token_type_ids'].append(encoding['token_type_ids'])
        batch_encodings['attention_mask'].append(encoding['attention_mask'])

    return batch_encodings
 
class Liwc_Dataset(Dataset):
    def __init__(self, encodings, labels, liwc_features):
        # Convert all data to tensors here, if memory allows
        self.encodings = {key: torch.tensor(vals) for key, vals in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.liwc_features = torch.tensor(liwc_features, dtype=torch.float)
        
    def __getitem__(self, idx):
        # Access tensors directly without conversion
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        item['liwc_features'] = self.liwc_features[idx]
        return item
   
    def __len__(self):
        return len(self.labels)
    
class Standard_dataset(Dataset):
    def __init__(self, encodings, labels):
        # Convert all data to tensors here, if memory allows
        self.encodings = {key: torch.tensor(vals) for key, vals in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __getitem__(self, idx):
        # Access tensors directly without conversion
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item
   
    def __len__(self):
        return len(self.labels)

def create_dataset(df, tokenizer, is_liwc):
    print("Token")
    
    # Retrieve text data
    text_list = df['text'].tolist()
    
    # Determine the number of available CPU cores and prepare data chunks
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(text_list) // num_cores
    text_chunks = [text_list[i:i + chunk_size] for i in range(0, len(text_list), chunk_size)]
    
    # Create a pool of workers to process data in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Map the tokenize_batch function over chunks of texts
        results = list(tqdm(pool.starmap(tokenize_batch, [(chunk, tokenizer) for chunk in text_chunks]), total=len(text_chunks)))
    
    # Combine results from all processes
    combined_input_ids = []
    combined_inputs_token_type_ids= []
    combined_attention_masks = []
    for result in results:
        combined_input_ids.extend(result['input_ids'])
        combined_inputs_token_type_ids.extend(result['token_type_ids'])
        combined_attention_masks.extend(result['attention_mask'])
    # make into single dictionary
    encodings = {"input_ids": combined_input_ids, "token_type_ids": combined_inputs_token_type_ids, "attention_mask": combined_attention_masks}
    encodings = {key: torch.tensor(vals) for key, vals in encodings.items()}
    # Prepare the dataset
    labels = df['label'].astype(int).tolist()  # Ensure labels are integers

    if is_liwc:
        liwc_features = df.drop(columns=['id', 'text', 'label']).values
        dataset = Liwc_Dataset(encodings, labels, liwc_features)
        sample_item = dataset[0]
        print(sample_item.keys())
    else:
        dataset = Standard_dataset(encodings, labels)
        sample_item = dataset[0]
        print(sample_item.keys())

    return dataset
 
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

class Bert_with_Liwc(nn.Module):
    def __init__(self, model_name, liwc_features_dim):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.liwc_processor = nn.Linear(liwc_features_dim, 128)
        self.extra_classifier = nn.Linear(128 + 1, 1)  

    def forward(self, input_ids, attention_mask, liwc_features, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  
        liwc_output = self.liwc_processor(liwc_features)  
        combined_features = torch.cat((logits, liwc_output), dim=1)  
        final_logits = self.extra_classifier(combined_features).squeeze(-1)  
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(final_logits, labels.float())
            return {"loss": loss, "logits": final_logits}

        return {"logits": final_logits}
    
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch['liwc_features'] = torch.stack([f['liwc_features'] for f in features])
        if 'labels' in features[0]:  # Ensure labels are also included
            batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.float)
        return batch

def train_model(train_dataset, model_type, tokenizer, is_liwc=False, is_spanish=False):
    # if train dataset contains liwc, use liwc model
    if is_liwc:
        print("Using LIWC model ...")
        if is_spanish:
            model = Bert_with_Liwc(model_name=model_type, liwc_features_dim = 87)
        else:
            model = Bert_with_Liwc(model_name=model_type, liwc_features_dim = 82)
        data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=2)

    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.15, random_state=42)
    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    valid_set = torch.utils.data.Subset(train_dataset, val_indices)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=NUM_EPOCHS,              
        per_device_train_batch_size=BATCH_SIZE,  
        per_device_eval_batch_size=BATCH_SIZE,              
        learning_rate = LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_ratio = WARMUP_RATIO,       
        weight_decay=WEIGHT_DECAY,        
        logging_dir='./logs',            
        logging_steps=10,
        evaluation_strategy="epoch",    
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=data_collator,
    )
    trainer.train()
    
    return trainer

def make_predictions(trained_model, test_dataset, is_liwc=False):
    predictions = trained_model.predict(test_dataset)
    if (is_liwc):
        probabilities = sigmoid(torch.from_numpy(predictions.predictions)) 
        predicted_labels = (probabilities > 0.7).long()
    else:
        #predicted_labels = predictions.predictions.argmax(-1)
        probabilities = sigmoid(torch.from_numpy(predictions.predictions)) 
        predicted_labels = (probabilities > 0.7).long()
    label_names = ['control', 'depressed']

    # Classification Report
    report = classification_report(predictions.label_ids, predicted_labels, target_names=label_names)
    print("Classification Report:\n", report)
    return report

def main():
    """
    print("Creating grouped dataset file for RSDD_train ...")
    RSDD_test = pd.read_csv(TEST_EN_PATH)
    RSDD_test = fix_df_types(RSDD_test)
    RSDD_test = group_df_by_user(RSDD_test)
    # Write to csv
    RSDD_test.to_csv('RSDD_test_grouped.csv', index=False)
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    # Adding named command line arguments
    parser.add_argument('--train', type=str, help='Enter train dataset', required=True)
    parser.add_argument('--test', type=str, help='Enter test dataset', required=True)
    parser.add_argument('--model', type=str, help='Enter BERT model', required=True)
    parser.add_argument('--save_tokens', action='store_true', help='Flag to save tokens')
    parser.add_argument('--load_tokens', action='store_true', help='Flag to load tokens from file')

    args = parser.parse_args()
    MODEL = args.model
    IS_SPANISH = False

    if args.model == "EN":
        MODEL = "bert-base-uncased"
    elif args.model == "ES":
        MODEL = 'dccuchile/bert-base-spanish-wwm-uncased'
        IS_SPANISH = True
    
    if "LIWC" in args.train:
        IS_LIWC = True
    else:
        IS_LIWC = False

    print(f"Starting dataset creation with TRAIN:{args.train}, TEST:{args.test} MODEL:{MODEL}")
    train_df, test_df = load_datasets(args.train, args.test)
    tokenizer = BertTokenizer.from_pretrained(MODEL)

    if args.load_tokens:
        print("Loading train dataset from file ... ")
        train = load_from_disk('./saved_tokens/train_dataset')
    else:
        train = create_dataset(train_df, tokenizer, IS_LIWC)

    if args.load_tokens:
        print("Loading test dataset from file ... ")
        test = load_from_disk('./saved_tokens/test_dataset')
    else:
        test = create_dataset(test_df, tokenizer, IS_LIWC)

    if args.save_tokens:
        # Save train dataset
        directory = "./tokenized_datasets/train"  # Current directory, change as needed
        base_filename = "dataset"
        next_file_number = 1
        while os.path.exists(f"{directory}/{base_filename}_{next_file_number}"):
            next_file_number += 1
        full_path = f"{directory}/{base_filename}_{next_file_number}{file_extension}"
        #train.save_to_disk(full_path)
        # Save test dataset
        directory = "./tokenized_datasets/test"  # Current directory, change as needed
        base_filename = "dataset"
        next_file_number = 1
        while os.path.exists(f"{directory}/{base_filename}_{next_file_number}"):
            next_file_number += 1
        full_path = f"{directory}/{base_filename}_{next_file_number}{file_extension}"
        #test.save_to_disk(full_path)


    # Train and save model

    trained_model = train_model(train, MODEL, tokenizer, IS_LIWC, IS_SPANISH)

    #Make predictions and write them to file
    c_report = make_predictions(trained_model, test, IS_LIWC)

    directory = "./classification_reports/bert/all_3"  # Current directory, change as needed
    base_filename = "report"
    file_extension = ".txt"
    next_file_number = 1
    while os.path.exists(f"{directory}/{base_filename}_{next_file_number}{file_extension}"):
        next_file_number += 1
    full_path = f"{directory}/{base_filename}_{next_file_number}{file_extension}"
    with open(full_path, 'w') as f:
        f.write(f"Model: {MODEL}\n")
        f.write(f"Train dataset: {args.train}\n")
        f.write(f"Test dataset: {args.test}\n")         
        f.write("--------------------------------------------------\n")
        f.write(f"NUM_EPOCHS = {NUM_EPOCHS}\n")
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"PREPROCESS = {PREPROCESS}\n")
        f.write(f"WEIGHT_DECAY = {WEIGHT_DECAY}\n")
        f.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
        f.write(f"WARMUP_RATIO = {WARMUP_RATIO}\n")
        f.write("--------------------------------------------------\n")
        f.write(c_report)
    print(f"Classification report saved to {full_path}")

    model_path = "./saved_models"
    #trained_model.save_model(model_path)
    #print(f"Model saved to {model_path}")

main()


