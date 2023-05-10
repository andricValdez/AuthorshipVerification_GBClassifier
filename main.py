import os
import re
import platform
import argparse
import subprocess
import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.split import getDataJSON
from utils import verif_evaluator

# TIRA exec cmd: python /main.py --input $inputDataset/input.jsonl --output $outputDir/answers.jsonl
# Local exec cmd: python main.py --mode=test --input=C:\Users\Qualtop\Desktop\andric\Projects\AuthorshipVerification-GBC\inputs\pan23\train.jsonl --output=C:\Users\Qualtop\Desktop\andric\Projects\AuthorshipVerification-GBC\outputs\answers.jsonl 

#tira-run --image registry.webis.de/code-research/tira/tira-user-pan23-cdav-2/gnc_clf:version1.1 --command 'python main.py --input-dataset-dir $inputDataset --output-dir $outputDir'

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
INPUTS_DIR = os.path.join(ROOT_DIR, "inputs")
DATASET_DIR = os.path.join(ROOT_DIR, INPUTS_DIR, 'pan23')
MODEL_FILE = os.path.join(OUTPUT_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(OUTPUT_DIR, "vectorizer.pkl")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.jsonl")
EVALUATION_FILE = os.path.join(OUTPUT_DIR, "evaluations.jsonl")
PAIRS_DATA_FILE = os.path.join(DATASET_DIR, "pairs.jsonl")
PAIRS_TRUTH_DATA_FILE = os.path.join(DATASET_DIR, "truth.jsonl")
TRAIN_DATA_FILE = os.path.join(DATASET_DIR, "train.jsonl")
TRAIN_TRUTH_DATA_FILE = os.path.join(DATASET_DIR, "train_truth.jsonl")
TEST_DATA_FILE = os.path.join(DATASET_DIR, "test.jsonl")
TEST_TRUTH_DATA_FILE = os.path.join(DATASET_DIR, "test_truth.jsonl")
VAL_DATA_FILE = os.path.join(DATASET_DIR, "val.jsonl")
VAL_TRUTH_DATA_FILE = os.path.join(DATASET_DIR, "val_truth.jsonl")


def read_dataset(data, truth_data, merge=True):
    dataframe = pd.DataFrame(getDataJSON(data)).set_index("id")
    dataframe[['text1','text2']] = pd.DataFrame(dataframe.pair.tolist(), index= dataframe.index)
    del dataframe["pair"]
    if merge:
        truth_dataframe = pd.DataFrame(getDataJSON(truth_data)).set_index("id")
        dataframe = pd.merge(dataframe, truth_dataframe, how='outer', left_index=True, right_index=True)
        del truth_dataframe
    return dataframe


def save_data(output_file, data):
    with open(output_file,"w+") as f:
        f.write(data)


def build_vectorizer(train_data, vect='count'):
    if vect == 'count':
        vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
    elif vect == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)

    vectorizer.fit(train_data["text1"] + train_data["text2"])
    save_obj(vectorizer, VECTORIZER_FILE)
    return vectorizer


def train_model(train_data, vectorizer):
    doc_vec = vectorizer.transform(train_data["text1"]) - vectorizer.transform(train_data["text2"])
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    model.fit(doc_vec, train_data["value"])
    save_obj(model, MODEL_FILE)
    return model


def save_obj(obj, file):
    joblib.dump(obj, file)


def load_obj(file_obj):
    return joblib.load(file_obj)


def gradientBoostingClassifier():
    parser = argparse.ArgumentParser(description='GBC script AA@PAN23')
    parser.add_argument("-m", "--mode", help="select from train | test | evaluate options. 'train' for only train the model, 'test' for only test the model", default='test', type=str, required=False)
    parser.add_argument("-i", '--input', help='The input data (expected in jsonl format). Must be full path.', type=str, required=False)
    parser.add_argument('-o', '--output',  help='The spoiled posts in jsonl format. Must be full path', type=str, required=True)
    args, _ = parser.parse_known_args()
    args = vars(args)

    #Entrenamiento GradientBoostingClassifier
    if args['mode'] == 'train':
        # Read and process train dataset
        print('*** Read and process train dataset')
        train_data = read_dataset(TRAIN_DATA_FILE, TRAIN_TRUTH_DATA_FILE, merge=True)
        print("train_data_shape: ", train_data.shape)

        # Vectorize data
        print('*** Building Vectorizer')
        vectorizer = build_vectorizer(train_data, vect='count')

        # Train model
        print('*** Training model')
        model = train_model(train_data, vectorizer)
    else:
        model = load_obj(MODEL_FILE)
        vectorizer = load_obj(VECTORIZER_FILE)

    # Leer y procesar el dataset de test
    print('*** Read and process test dataset')
    test_data = read_dataset(args['input'],  TEST_TRUTH_DATA_FILE, merge=False)
    print("test_data_shape: ", test_data.shape)

    # Predicción
    print('*** Doing predictions... ')  
    X_test = vectorizer.transform(test_data["text1"]) - vectorizer.transform(test_data["text2"])
    y_pred = model.predict(X_test)
    preds = ""
    for x,y in zip(test_data.index, y_pred):
        preds += str({"id":x, "value":int(y)}).replace("'",'"')+"\n"
    save_data(args['output'], preds)

    # Evaluate model
    if args['mode'] == 'evaluate':
        print('*** Evaluating model... ')  
        verif_evaluator.main(PAIRS_TRUTH_DATA_FILE, args['output'], OUTPUT_DIR)
    

if __name__ == "__main__":
    gradientBoostingClassifier()