import csv
import time
from distutils.log import Log

import numpy
import torch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import torch.nn.functional as F
from importlib_metadata import version
print(version('tokenizers'))
from transformers import BertTokenizer, BertForSequenceClassification
import trio
import preprocessor as pp
import emoji
from tqdm import tqdm
import pandas as pd

import numpy as np

d = DesiredCapabilities.CHROME
d['goog:loggingPrefs'] = { 'browser':'ALL' }

#TODO: Hash the URLs, build the pipeline

async def printConsoleLogs():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('debuggerAddress', 'localhost:9515')
    driver = webdriver.Chrome(options=chrome_options, desired_capabilities=d)
    df = pd.read_csv('tiktok-metas.csv')
    df = df.reset_index(drop=True)
    for index,row in tqdm(df.iterrows()):
        driver.get(row['webVideoUrl'])
        action_list = driver.find_elements(By.CLASS_NAME, "tiktok-1pqxj4k-ButtonActionItem")
        cmt_btn = action_list[1]
        cmt_btn.click()
        time.sleep(1)
        comments = (element.text for element in driver.find_elements(By.CLASS_NAME, "tiktok-q9aj5z-PCommentText"))
        comments = list(comments)
        input_ids = []
        attention_masks = []
        model_path = './model'
        tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', do_lower_case=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelog = BertForSequenceClassification.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2",
                                                              num_labels=2)
        modelnew = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        modelog.to(device)
        modelnew.to(device)
        comments.insert(0, row['text'])

        comments = [pp.tokenize(text) for text in comments]
        comments = [emoji.demojize(text) for text in comments]
        comments = [text.lower() for text in comments]

        # For every sentence...
        for sent in tqdm(comments, desc="Tokenizing Comments", colour="CYAN"):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=128,  # Pad & truncate all sentences.
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        val_labels = [1] * len(input_ids)
        labels = torch.tensor(val_labels)

        # Set the batch size.
        batch_size = 8

        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

        # Put model in evaluation mode
        modelog.eval()
        modelnew.eval()

        # Tracking variables
        predictionsog, predictionsnew, true_labels = [], [], []
        label_list = ["fake", "real"]

        # Predict
        for batch in tqdm(prediction_dataloader, desc="Predicting", colour="GREEN"):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputsog = modelog(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)
                outputsnew = modelnew(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)

            logitsog = outputsog[0]
            logitsnew = outputsnew[0]

            # Move logits and labels to CPU
            logitsog = logitsog.detach().cpu().numpy()
            logitsnew = logitsnew.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            for prediction in F.softmax(torch.from_numpy(logitsog), dim=-1):
                #predictions.append(label_list[prediction])
                predictionsog.append(prediction)
            for prediction in F.softmax(torch.from_numpy(logitsnew), dim=-1):
                #predictions.append(label_list[prediction])
                predictionsnew.append(prediction)

            # Store predictions and true labels
            #predictions.append(logits)
            true_labels.append(label_ids)
        totalog, totalnew = 0,0
        for i,pred in enumerate(predictionsog):
            if(pred.numpy()[0] > 0.55):
                if i == 0:
                    totalog += 5
                totalog += 1
        for i,pred in enumerate(predictionsnew):
            if(pred.numpy()[0] > 0.001):
                if i == 0:
                    totalnew += 5
                totalnew += 1
        totalog = totalog / (len(predictionsog) + 5)
        totalnew = totalnew / (len(predictionsog) + 5)
        print("Totals: " + str(totalog) + " / " + str(totalnew))
        totals = [totalog, totalnew]
        weights = [0.6, 0.4]
        total = numpy.average(totals, weights=weights)
        val = -1
        if(total > 0.35 and total < 0.65):
            val = 1
        elif(total > 0.65):
            val = 2
        else:
            val = 0
        row['label'] = val
    df.to_csv('model_results.csv')

trio.run(printConsoleLogs)