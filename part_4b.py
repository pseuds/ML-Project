import pandas as pd
import numpy as np
import os
import subprocess

# Data Processing
def text_to_dataframe(text):
    text = text.strip().split('\n')
    data = [line.rsplit(" ", 1) for line in text if line.strip() != ""]
    df = pd.DataFrame(data, columns=['x', 'y'])
    return df

def vocabulary(df):
    vocabulary = df['x'].unique()
    return vocabulary

# Naive Bayes Prediction
def naive_bayes(train_path,file_in,file_out):
    file_path = os.getcwd()

    f = open(f"{file_path}/{train_path}", "r", encoding="utf8")
    raw_train = f.read()
    data = text_to_dataframe(raw_train)
    vocab = vocabulary(data)

    # Calculate prior class probabilities P(label)
    label_counts = data['y'].value_counts().to_dict()
    total_samples = data.shape[0]
    prior_probs = {label: count / total_samples for label, count in label_counts.items()}

    df =  data.groupby(['x', 'y']).size().reset_index(name='count')

    # Calculate likelihoods P(word | label)
    likelihoods = {}
    for label, count in label_counts.items():
        likelihoods[label] = {}

        for word in vocab:
            # Calculate the count of occurrences of the current word with the current label
            filtered_data = df[(df['x'] == word) & (df['y'] == label)]
            try:
                word_label_count = filtered_data['count'].iat[0]
            except:
                word_label_count = 0

            # Apply Laplace smoothing where alpha = 1
            smoothed_likelihood = (word_label_count + 1) / (count + len(vocab))

            likelihoods[label][word] = smoothed_likelihood


    # Test sentence
    with open(f"{file_path}/{file_in}", "r",encoding="utf8") as input_file:
        data_devin = input_file.readlines()

    # Cleaned data_devin
    sequence = [line.strip() for line in data_devin]

    # Perform Naive Bayes on test sentence
    predictions = []

    for word in sequence:
        if word != '':
            # simplified from: 
            # word_probs = {}
            
            # for label, count in label_counts.items():
            #     if word in likelihoods[label]:
            #         word_prob = np.log(prior_probs[label]) + np.log(likelihoods[label][word])
            #     else:
            #         word_prob = np.log(prior_probs[label]) + np.log(1 / (count + len(vocab)))
            #     word_probs[label] = word_prob
            
            word_probs = {label: np.log(prior_probs[label]) + np.log(likelihoods[label].get(word, 1 / (count + len(vocab))))
                          for label, count in label_counts.items()}

            predicted_label = max(word_probs, key=word_probs.get)
            predictions.append(predicted_label)
        else:
            predictions.append('')

    # Write the output to dev.p1.out file
    with open(f"{file_path}/{file_out}", 'w',encoding="utf8") as output_file:
        for word, sentiment in zip(data_devin, predictions):
            output_file.write(f"{word.strip()} {sentiment}\n")

def new_model():
    # language 
    languages = ['ES','RU']
    # Get the current working directory
    current_dir = os.getcwd()
    
    for language in languages:
        print(f"For Language: {language}")
        # Command to run
        command = ["python3", f"{current_dir}/EvalScript/evalResult.py", f"{current_dir}/Data/{language}/dev.out", f"{current_dir}/Data/{language}/dev.p4b.out"]
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)