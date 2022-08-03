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

import numpy as np

d = DesiredCapabilities.CHROME
d['goog:loggingPrefs'] = { 'browser':'ALL' }

#TODO: Hash the URLs, build the pipeline

async def printConsoleLogs(url):
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_experimental_option("detach", True)
    chrome_options.add_experimental_option('debuggerAddress', 'localhost:9515')
    driver = webdriver.Chrome(options=chrome_options, desired_capabilities=d)
    #driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', service_args=['--verbose'], options=chrome_options, desired_capabilities=dc)
    #driver = Chrome(executable_path='chromedriver')
    #driver.get("https://www.tiktok.com/@evalicious_8910/video/7029269165189975301?_t=8TsbUUQw3XH&_r=1") #- fake
    #driver.get("https://www.tiktok.com/@thedisneygirlie/video/7115909702218812714?_t=8UUN7lXFNed&_r=1") #- real
    #driver.get("https://www.tiktok.com/@haysfordaysss/video/7052861684343557423?_t=8UUNy3Gb2g2&_r=1") #- real but misleading
    #driver.get("https://www.tiktok.com/@queen_marlene_/video/7049317613809192198?_t=8UUPtwEdLHK&_r=1")
    driver.get(url)
    action_list = driver.find_elements(By.CLASS_NAME, "tiktok-1pqxj4k-ButtonActionItem")
    cmt_btn = action_list[1]
    cmt_btn.click()
    time.sleep(1)
    print("Clicked!")
    time.sleep(1)
    print("Executing script...")
    '''for element in driver.find_elements(By.CLASS_NAME, "tiktok-q9aj5z-PCommentText"):
        print(element.text)'''
    comments = (element.text for element in driver.find_elements(By.CLASS_NAME, "tiktok-q9aj5z-PCommentText"))
    comments = list(comments)
    #print(comments)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    model_path = '/Users/josh/Documents/GitHub/Tiktok-OCR/model'
    tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', do_lower_case=True)
    modelog = BertForSequenceClassification.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2",
                                                          num_labels=2)
    modelnew = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)


    #comments.insert(0,"Eternally grateful for science #paxlovid #covid19")
    #comments.insert(0, "What are your guesses?? (Before the video ends) #covid19 #binaxnow #covidhometest #superhuman")
    #comments.insert(0, "Fully vaccinated and this is the second time I’ve caught covid.")
    comments.insert(0, "Why u worried if u had your jabs? $emoji$ $emoji$")

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
        batch = tuple(t.to(torch.device("cpu")) for t in batch)

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

    print('    DONE.')

    print(comments)
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
    print("Weighted average: " + str(round(total * 100,1)) + "%")
    if(total > 0.5):
        print("It's a real tiktok!")
    else:
        print("It's a fake tiktok!")

    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in range(0, len(comments)):
            myList = []
            myList.append(comments[row])
            #myList.append((predictions[row]).numpy()[0] > 0.001)
            myList.append(predictionsog[row].numpy()[0])
            myList.append(predictionsnew[row].numpy()[0])
            writer.writerow(myList)
    print("CSV done!")

trio.run(printConsoleLogs)