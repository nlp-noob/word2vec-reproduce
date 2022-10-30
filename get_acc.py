import numpy as np
import pandas as pd
import torch
import sys
from sklearn import svm

train_positive={}
train_negative={}
test_positive={}
test_negative={}

train_len=0
positive_train_len=0
negative_train_len=0
positive_test_len=0
negative_test_len=0
import json
with open('./test_data/yelp_academic_dataset_review.json') as f:
    for line in f:
        data = json.loads(line)
        review_data = data['text'].split()
        stars = data['stars']
        if(stars > 3 and positive_train_len<150000):
            train_positive[positive_train_len] = review_data
            positive_train_len += 1
        elif(stars <= 3 and negative_train_len<150000):
            train_negative[negative_train_len] = review_data
            negative_train_len += 1
        elif(stars > 3 and positive_test_len<15000):
            test_positive[positive_test_len] = review_data
            positive_test_len += 1
        elif(stars <= 3 and negative_test_len<15000):
            test_negative[negative_test_len] = review_data
            negative_test_len += 1
        elif(sum([positive_train_len, negative_train_len,
                  positive_test_len, negative_test_len])<330000):
            continue
        else:
            break

print(len(train_positive), len(train_negative))
print(len(test_positive), len(test_negative))

sys.path.append("../")
folder = "weights/cbow_Win5_WikiText2"
device = torch.device("cpu")
model = torch.load(f"{folder}/5epochs_model.pt", map_location=device)
vocab = torch.load(f"{folder}/vocab.pt")

# embedding from first model layer
embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()

# normalization
norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
norms = np.reshape(norms, (len(norms), 1))
embeddings_norm = embeddings / norms
embeddings_norm.shape
# get embeddings
embeddings_df = pd.DataFrame(embeddings)

# How to use embeddings
#emb1 = embeddings[vocab["king"]]
#emb2 = embeddings[vocab["man"]]
#emb3 = embeddings[vocab["woman"]]
dim_vec = len(embeddings[vocab["king"]]) 


# ignore words not in
def GetMeans(dataset):
    h={}
    for key, words in dataset.items():
        vec = np.zeros(dim_vec,)
        for i in range(0, len(words)):
            word = words[i]
            vec = np.add(vec, embeddings[vocab[word]])
        h[key] = [x/len(words) for x in vec]
    return h

positive_means = GetMeans(train_positive)
negative_means = GetMeans(train_negative)

def AppendMeans(m, val, mean_vals, Y_vals):
    for k,v in m.items():
        mean_vals.append(v)
        Y_vals.append(val)
    return [mean_vals, Y_vals]

[a, b] = AppendMeans(positive_means, 1, [], [])
[train_means,train_Y] = AppendMeans(negative_means, 0, a, b)

train_means=np.array(train_means)
print(train_means.shape)
print(len(train_Y))

clf = svm.SVC(gamma='scale', kernel='rbf')
print(clf.fit(train_means, train_Y))

# begin to test the model here
[c,d] = AppendMeans(GetMeans(test_negative), 0, [], [])
[test_means,test_Y] = AppendMeans(GetMeans(test_positive), 1, c, d)

predicted = np.array(clf.predict(test_means))

diff = [np.abs(a-b) for a,b in zip(test_Y, predicted)]

accuracy = 1-(diff.count(1)/len(predicted))

actual = test_Y

fneg = 0
fpos = 0
tpos =0
for i in range(len(actual)):
    if(actual[i]-predicted[i] == 1):
        fneg+=1
    if(predicted[i]-actual[i] == 1):
        fpos+=1
    if(predicted[i]+actual[i] == 2):
        tpos+=1
print('Pre-trained term embeddings:')
print('Accuracy', accuracy)
P= tpos/(tpos+fpos)
R= tpos/(tpos+fneg)
print('Precision', P)
print('Recall', R)








