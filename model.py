import pandas as pd
import random
import nltk
from nltk.tokenize import word_tokenize
import collections
from nltk.metrics import (precision, recall, f_measure)

#format csv files - search through and replace "ham ," with "ham<>"

text = open("spam 2.csv", "r", encoding = "ISO-8859-1")


#creating custom delimiter because there are commas in text messages
text = ''.join([i for i in text]).replace("ham,", "ham<>")
text = ''.join([i for i in text]).replace("spam,", "spam<>")
text = ''.join([i for i in text]).replace("v1,", "v1<>")
#accounting for some entries where there is a '.'
text = ''.join([i for i in text]).replace("spam.", "spam<>")
x = open("delspam2.csv","w")
x.writelines(text)
x.close



#print(text)

#reading
df = pd.read_csv('delspam2.csv', delimiter='<>', engine = 'python')

df.columns


df = df.sort_index()

hams = df.loc[df['v1'] == 'ham']
spams = df.loc[df['v1']=='spam']
#print(hams)
#print(spams)

labeledData = []


print(hams.columns)
AllWords = []
text = ""
for i in hams['v2,,,']:
    labeledData.append((i, 'ham'))
    text+= i ;
for i in spams['v2,,,']:
    labeledData.append((i, 'spam'))
    text += i;
random.shuffle(labeledData)

# take text containing all words and tokenize it
tokens =  nltk.word_tokenize(text)
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]


# get frequency dist of all words
all_words = nltk.FreqDist(words)
#print(all_words)
#take 2000 most prominent words
word_features = list(all_words)[:2000]
#print((word_features))



def document_features(document):
    document_words = set(document)
    #print(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# need to open the files and read at every labeled review
#featuresets = [(document_features(d), c) for (d,c) in labeledReviews]
# loop and append to featuresets
featuresets = []
for (d,c) in labeledData:
   # print(d,c)
    docToken = word_tokenize(d)
    feat = document_features(docToken)
    featuresets.append((feat, c))
  ## problem


#train_set, test_set = featuresets[400:], featuresets[:400]

### cross val split occurs here  -- 10 folds
num_folds = 10
subset_size = int(round(len(featuresets)/num_folds))

# for the Bayes model
foldAccuracies = []
foldNegativePrecisions = []
foldNegativeRecalls = []
foldNegativeFScores = []
foldPositivePrecisions = []
foldPositiveRecalls = []
foldPositiveFScores = []

for i in range(num_folds):
    cv_test = featuresets[i*subset_size:][:subset_size]
    cv_train = featuresets[:i*subset_size] + featuresets[(i+1)*subset_size:]
    # use NB classifier
    classifier = nltk.NaiveBayesClassifier.train(cv_train)
    print('  ')
    print('FOLD ' + str(i))
    print('For this fold:')
    print('Accuracy on Fold Test Set: ' + str(nltk.classify.accuracy(classifier, cv_test)))
    foldAccuracies.append(str(nltk.classify.accuracy(classifier, cv_test)));
    # most informative feauures
    # now get fold stats such as precison, recall, f score
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(cv_test):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    foldPositivePrecisions.append(str(precision(refsets['spam'], testsets['spam'])))
    foldPositiveRecalls.append(str(recall(refsets['spam'], testsets['spam'])))
    foldPositiveFScores.append(str(f_measure(refsets['spam'], testsets['spam'])))
    foldNegativePrecisions.append(str(precision(refsets['ham'], testsets['ham'])))
    foldNegativeRecalls.append(str(recall(refsets['ham'], testsets['ham'])))
    foldNegativeFScores.append(str(f_measure(refsets['ham'], testsets['ham'])))

    print('Positive Precision:', precision(refsets['spam'], testsets['spam']))
    print('Positive Recall:', recall(refsets['spam'], testsets['spam']))
    print('Positive F1-Score:', f_measure(refsets['spam'], testsets['spam']))
    print('Negative Precision:', precision(refsets['ham'], testsets['ham']))
    print('Negative Recall:', recall(refsets['ham'], testsets['ham']))
    print('Negative F1-Score:', f_measure(refsets['ham'], testsets['ham']))
    classifier.show_most_informative_features(5)

total = 0
totalPrecPos = 0
totalRecallPos = 0
totalFScorePos = 0
totalPrecNeg = 0
totalRecallNeg = 0
totalFScoreNeg = 0
for i in range(0, len(foldAccuracies)):
    total = total + float(foldAccuracies[i])
    totalPrecPos = totalPrecPos + float(foldPositivePrecisions[i])
    totalRecallPos = totalRecallPos + float(foldPositiveRecalls[i])
    totalFScorePos = totalFScorePos + float(foldPositiveFScores[i])
    totalPrecNeg = totalPrecNeg + float(foldNegativePrecisions[i])
    totalRecallNeg = totalRecallNeg + float(foldNegativeRecalls[i])
    totalFScoreNeg = totalFScoreNeg + float(foldNegativeFScores[i])

total_accuracy = total/num_folds
total_pos_prec = totalPrecPos/num_folds
total_pos_recall = totalRecallPos/num_folds
total_pos_fscore = totalFScorePos/num_folds
total_neg_precision = totalPrecNeg/num_folds
total_neg_recall = totalRecallNeg/num_folds
total_neg_fscore = totalFScoreNeg/num_folds
print('---------')
print('Averaged model performance over 10 folds: ')
print('   ')
print('Average accuracy over 10 folds: ' + str(total_accuracy))
print('Average precision for positive class: ' + str(total_pos_prec))
print('Average recall for positive class ' + str(total_pos_recall))
print('Average F-score for positive class ' + str(total_pos_fscore))
print('  ')
print('Average precision for negative class ' + str(total_neg_precision))
print('Average recall for negative class ' + str(total_neg_recall))
print('Average F-score for negative class ' + str(total_neg_fscore))



