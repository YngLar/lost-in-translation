# Lost in Translation: Depression Detection in Multilingual Social Media
This is the code used in a master's thesis at the Norwegian University of Science and Technology (NTNU). The thesis investigates the universality of depression characteristics in texts across different languages, focusing on the effectiveness of depression detection models in multilingual social media contexts. The project involves analyzing, translating, and experimenting with datasets from various languages, including Spanish and English, to identify similarities and differences in the expression of depressive symptoms. The codebase includes scripts for data preprocessing, model training, and evaluation, utilizing both classical machine learning models and transformer-based models like BERT and BETO. Additionally, it incorporates the use of the Linguistic Inquiry and Word Count (LIWC) lexicon for feature extraction.

The datasets used have to be aquired before running the code. They need to be added to the correct paths, which can be found in the code.

## DATASET COMBINATIONS (for --train or --test)
EN refers to RSDD, whereas ES refers to MentalRiskES.
- English
    - EN
    - EN_LIWC
    - EN+ES_translated
    - EN+ES_translated_LIWC
    - ES_translated
    - ES_translated_LIWC
- Spanish
    - ES
    - ES_LIWC
    - ES+EN_translated
    - ES+EN_translated_LIWC
    - EN_translated
    - EN_translated_LIWC


## BERT model (for --model)
- "EN": "bert-base-uncased"
- "ES": "dccuchilebert-base-spanish-wwm-cased"
- For others: write their name on huggingface :)

# How to run
```
python BERT_generalized.py --train [TRAIN_DATASET] --test [TEST_DATASET] --model [MODEL]
```
