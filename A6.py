####Assignment 6 
#Christian Burkhartsmeyer
#Caleb Kaiji Lu

import re
import numpy as np
import math

def buildArray (listOfIndices):
        returnVal = np.zeros (20000)
        for index in listOfIndices:
                returnVal[index] = returnVal[index] + 1
        return returnVal

trainingTxt = 's3://chrisjermainebucket/comp330_A6/20_news_same_line.txt'
trainingTxt_local = '/Users/CalebKaijiLu/Dropbox/COMP330/Assignment4/20_news_same_line.txt'
corpus = sc.textFile (trainingTxt_local)
corpusSize = corpus.count ();
# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)

# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = allCounts.top (20000, lambda x : x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pair
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords

###Taks1
dictionary = twentyK.map (lambda x:(topWords[x][0],x))

allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# and now join/link them, to get a bunch of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = dictionary.join (allWords)

# and drop the actual word itself to get a bunch of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))

# now get a bunch of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey ()
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map (lambda x: (x[0], buildArray (x[1])))

l = allDocsAsNumpyArrays

## 100 most common dictionary words appear in document 20 newsgroups/comp.graphics/37261.
t1_array= allDocsAsNumpyArrays.lookup('20_newsgroups/comp.graphics/37261')[0]
t1_index = t1_array.argsort()[-100:][::-1]
print dictionary.filter(lambda x:x[1] in t1_index).map(lambda x:x[0]).collect()
'''
>>> [u'the', u'to', u'of', u'a', u'and', u'in', u'is', u'for', u's', u'be', u'not', u'are', u'as', u'or', u'by', u'all', u'one', u'will', u'should', u'need', u'work', u'please', u'mail', u'point', u'information', u'thanks', u'must', u'david', u'software', u'long', u'available', u'day', u'however', u'group', u'possible', u'usa', u'call', u'above', u'research', u'code', u'fax', u'type', u'center', u'robert', u'contact', u'further', u'present', u'related', u'scientific', u'division', u'author', u'received', u'reality', u'alone', u'mil', u'stand', u'minutes', u'types', u'surface', u'navy', u'june', u'length', u'sick', u'virtual', u'addresses', u'authors', u'materials', u'taylor', u'dt', u'mar', u'visualization', u'vr', u'abstract', u'formerly', u'naval', u'presentation', u'structures', u'maryland', u'submission', u'distribute', u'attend', u'oasys', u'sixth', u'warfare', u'ocean', u'deadline', u'presentations', u'computational', u'seminar', u'reproduction', u'abstracts', u'bethesda', u'signatures', u'videotape', u'sponsor', u'lip', u'attendees', u'carderock', u'lipman', u'notification']

'''
##part2
def getProbs (checkParams, x, pi, log_allMus):
    #
    if checkParams == True:
            if x.shape [0] != log_allMus.shape [1]:
                    raise Exception ('Number of words in doc does not match')
            if pi.shape [0] != log_allMus.shape [0]:
                    raise Exception ('Number of document classes does not match')
            if not (0.999 <= np.sum (pi) <= 1.001):
                    raise Exception ('Pi is not a proper probability vector')
            for i in range(log_allMus.shape [0]):
                    if not (0.999 <= np.sum (np.exp (log_allMus[i])) <= 1.001):
                            raise Exception ('log_allMus[' + str(i) + '] is not a proper probability vector')
    #
    # to ensure that we don’t have any underflows, we will do
    # all of the arithmetic in “log space”. Specifically, according to
    # the Multinomial distribution, in order to compute
    # Pr[x | class j] we need to compute:
    #
    #       pi[j] * prod_w allMus[j][w]^x[w]
    # 
    # If the doc has a lot of words, we can easily underflow. So 
    # instead, we compute this as:
    #
    #       log_pi[j] + sum_w x[w] * log_allMus[j][w]
    #
    allProbs = np.log (pi)
    #
    # consider each of the classes, in turn
    for i in range(log_allMus.shape [0]):
            product = np.multiply (x, log_allMus[i])
            allProbs[i] += np.sum (product)
    #
    # so that we don’t have an underflow, we find the largest
    # logProb, and then subtract it from everyone (subtracting
    # from everything in an array of logarithms is like dividing
    # by a constant in an array of “regular” numbers); since we
    # are going to normalize anyway, we can do this with impunity
    #
    biggestLogProb = np.amax (allProbs)
    allProbs -= biggestLogProb
    #
    # finally, get out of log space, and return the answer
    #
    allProbs = np.exp (allProbs)
    return allProbs / np.sum (allProbs)


n_cluster = 20
n_words = 20000
n_iter = 200

alpha = np.zeros(n_cluster)+.1
beta = np.zeros(n_words)+.1

pi = np.random.dirichlet(alpha) #(20,)
mu = np.random.dirichlet(beta,20)#20*20000

for iter in range(n_iter):
	normalizedProbs = allDocsAsNumpyArrays.map(lambda x : (x[0],getProbs(False, x[1], pi, np.log(mu))))
	c = normalizedProbs.map(lambda x:(x[0],np.random.choice(n_cluster,1,p = x[1])[0]))
	c_ravel = np.array(c.flatMap(lambda x:x[1]).collect())
	c_count = np.unique(c_ravel,return_counts = True)[1]
	temp_alpha = alpha + c_count
	pi = np.random.dirichlet(temp_alpha)
	mu = np.zeros(n_cluster,n_words)

	####below code not finished
	haha = class_doc.map(lambda x:(x[1][0],x[1][1])) # get (class, original_count)
	for i in range(n_cluster):
		cnt = class_doc.map(lambda x:x[1][1])
		mu[i,:] = np.random.dirichlet(beta + cnt)


