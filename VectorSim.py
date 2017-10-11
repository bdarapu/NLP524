from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import entailment
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import Levenshtein as lv

#Load the word2vec model
model = Word2Vec.load_word2vec_format('vectors.txt', binary=False)

stop = stopwords.words('english')
wnl = WordNetLemmatizer()

#Entailment Score
def matcher(question, answer):

    ret = entailment.get_ai2_textual_entailment(answer, question)
    a_scores = list(map(lambda x: x['score'], ret['alignments']))
    if len(a_scores):
        mean_a_score = np.mean(a_scores)
    else:
        mean_a_score = 0

    confidence = ret['confidence'] if ret['confidence'] else 0
    score = mean_a_score * confidence

    # return confidence
    return score

def greedy_matching(triple_1, triple_2):

    #Get lemmatized tokens for each triple
    triple_1 = triple_1.lower().split(' ')
    triple_2 = triple_2.lower().split(' ')
    triple_1 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_1 if w not in stop]
    triple_2 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_2 if w not in stop]

    score = 0
    n = 0
    #Get word2vec / entailment(for Out-Of-Vocabulary words) score
    for w in triple_2:
        if w not in model:
            s = [matcher(w, ' '.join(triple_1))]
        else:
            s = []
            for t in triple_1:
                if t not in model:
                    s.append(matcher(w, t))
                else:
                    s.append((model.similarity(w, t)+1)*1.0/2)
        if s != []:
            score += max(s)
            n += 1
    return score/n if score > 0 else 0


def word_counting(triple_1, triple_2):
    triple_1 = triple_1.lower().split(' ')
    triple_2 = triple_2.lower().split(' ')
    triple_1 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_1 if w not in stop]
    triple_2 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_2 if w not in stop]

    count = 0

    for word_1 in triple_1:
        for word_2 in triple_2:
            if word_1 == word_2:
                count += 1

    return count

def adjacent_matching(triple_1, triple_2):
    triple_1 = triple_1.lower().split(' ')
    triple_2 = triple_2.lower().split(' ')
    triple_1 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_1 if w not in stop]
    triple_2 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_2 if w not in stop]
    triple_1 = [x for x in triple_1 if x != '']
    triple_2 = [x for x in triple_2 if x != '']

    count = 0
    index_1 = 0
    index_2 = 0

    #Searches for identical phrase of length 2 in both triples
    for word_1 in triple_1:
        for word_2 in triple_2:

            if word_1 == word_2:
                count += 1
                if count == 1:
                    index_1 = triple_1.index(word_1)
                    index_2 = triple_2.index(word_1)
                break
            elif count == 1:
                if (index_2 < len(triple_2)-1) and (word_1 == triple_2[index_2+1]):
                    count == 2
                    break
                break
            else:
                count = 0


    # If the phrases are both sorrounded by other words, they are compared using Greedy Matching
    # If not, they are automaticaly given the score of 1
    if count == 2:

        if (index_1 > 0 ) and (index_1 + 1 < len(triple_1) - 1 ):
            string_1 = triple_1[index_1 - 1] + " " + triple_1[index_1] + " " + triple_1[index_1 + 1]

        elif index_1 == 0 and len(triple_1) > 2:
            string_1 = triple_1[index_1] + " " + triple_1[index_1+1] + " " + triple_1[index_1+2]

        else:
            return 1

        if index_2 > 0 and (index_2 + 1 < len(triple_2) - 1):
            string_2 = triple_1[index_2 - 1] + " " + triple_1[index_2] + " " + triple_1[index_2 + 1]

        elif index_2 == 0 and len(triple_2) > 2:
            string_2 = triple_2[index_2] + " " + triple_2[index_2 + 1] + " " + triple_2[index_2 + 2]
        else:
            return 1
        return greedy_matching(string_1,string_2)
    #Searches for identical word in both triples
    else:
        count = 0
        for wrd_1 in triple_1:
            for wrd_2 in triple_2:
                if wrd_1 == wrd_2:
                    count += 1
                    index_1 = triple_1.index(wrd_1)
                    index_2 = triple_2.index(wrd_2)
                    break
            if count == 1:
                break

    # If the phrases are both sorrounded by other words, they are compared using Greedy Matching
    # If not, they are automaticaly given the score of 0

        if (index_1 > 0 ) and (index_1 < len(triple_1) - 1 ):
            string_1 = triple_1[index_1 - 1] + " " + triple_1[index_1] + " " + triple_1[index_1 + 1]
        elif index_1 == 0 and len(triple_1) > 2:
            string_1 = triple_1[index_1] + " " + triple_1[index_1+1] + " " + triple_1[index_1+2]
        else:
            return 0
        if index_2 > 0 and (index_2 < len(triple_2) - 1):
            string_2 = triple_2[index_2 - 1] + " " + triple_2[index_2] + " " + triple_2[index_2 + 1]
        elif index_2 == 0 and len(triple_2) > 2:
            string_2 = triple_2[index_2] + " " + triple_2[index_2 + 1] + " " + triple_2[index_2 + 2]
        else:
            return 0

        return greedy_matching(string_1, string_2)

def synset_counting(triple_1, triple_2):
    triple_1 = triple_1.lower().split(' ')
    triple_2 = triple_2.lower().split(' ')
    triple_1 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_1 if w not in stop]
    triple_2 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_2 if w not in stop]

    count = 0

    for w in triple_1:
        set_1 = wn.synsets(w)[:5]
        for word in triple_2:
            set_2 = wn.synsets(word)[:5]
            for syn_1 in set_1:
                for syn_2 in set_2:
                    if syn_1 == syn_2:
                            count += 1
    return count

def sense_extraction(triple_1, triple_2):

    triple_1 = triple_1.lower().split(' ')
    triple_2 = triple_2.lower().split(' ')
    triple_1 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_1 if w not in stop]
    triple_2 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_2 if w not in stop]
    triple_1 = [x for x in triple_1 if x != '']
    triple_2 = [x for x in triple_2 if x != '']

    #Extract keyword to use
    count = 0
    for wrd_1 in triple_1:
        if count == 1:
            break
        for wrd_2 in triple_2:
            if wrd_1 == wrd_2 or wrd_1 in wrd_2 or wrd_2 in wrd_1:
                count += 1
                index_1 = triple_1.index(wrd_1)
                index_2 = triple_2.index(wrd_2)
                break

    if count == 0:
        return 0

    #Take sorrounding words
    #Triple 1
    if (index_1 > 0) and (index_1 < len(triple_1) - 1):
        a,b = triple_1[index_1-1], triple_1[index_1+1]
    elif index_1 == 0 and len(triple_1) > 2:
        a, b = triple_1[index_1+1], triple_1[index_1+2]
    else:
        a,b =triple_1[index_1+1], ' '
    #Triple_2
    if (index_2 > 0) and (index_2 < len(triple_2) - 1):
        c, d = triple_2[index_2 - 1], triple_2[index_2 + 1]
    elif index_2 == 0 and len(triple_2) > 2:
        c, d = triple_2[index_2 + 1], triple_2[index_2 + 2]
    elif index_2 == len(triple_2)-1:
        c,d = triple_2[index_2-1], ' '
    else:
        c, d = triple_2[index_2 + 1], ' '

    #Find list of senses and find the most picked
    set_1 = set_2 = wn.synsets(triple_1[index_1])[:5]
    if set_1 == [] or set_2 == []:
        return 1
    for entry in set_1:
        if a in entry.definition():
            senses_1 = entry.definition()
            break
        else:
            senses_1 = ""
    for entry in set_2:
        if c in entry.definition():
            senses_2 = entry.definition()
            break
        else:
            senses_2 = ""
    if (senses_1 == senses_2) and (senses_1 != ""):
        return 1
    else:
        return 0

def lesk_similarity(triple_1, triple_2):
    triple_1 = triple_1.lower().split(' ')
    triple_2 = triple_2.lower().split(' ')
    triple_1 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_1 if w not in stop]
    triple_2 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_2 if w not in stop]

    triple_1 = [x for x in triple_1 if x != '']
    triple_2 = [x for x in triple_2 if x != '']

    count = 0
    for wrd_1 in triple_1:
        if count == 1:
            break
        for wrd_2 in triple_2:
            if wrd_1 == wrd_2 or wrd_1 in wrd_2 or wrd_2 in wrd_1:
                count += 1
                index_1 = triple_1.index(wrd_1)
                index_2 = triple_2.index(wrd_2)
                break
        if count == 0:
            return 0

    syn_1 = lesk(triple_1, triple_1[index_1])
    syn_2 = lesk(triple_2, triple_2[index_2])
    if syn_1 == syn_2:
        return 1
    else:
        return 0

#calculate Levenshtein Similarity for two triples
def MyLevenshteinRatio(triple_1,triple_2):
    triple_1 = triple_1.lower().split(' ')
    triple_2 = triple_2.lower().split(' ')
    triple_1 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_1 if w not in stop]
    triple_2 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_2 if w not in stop]
    score=0
    for w1 in triple_1:
        scorew1 = 0
        for w2 in triple_2:
            scorew1+=lv.ratio(w1,w2)
        score+=scorew1/len(triple_2)
    score=score/len(triple_1)
    return score

def MyLevenshteinSetRatio(triple_1,triple_2):
    triple_1 = triple_1.lower().split(' ')
    triple_2 = triple_2.lower().split(' ')
    triple_1 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_1 if w not in stop]
    triple_2 = [wnl.lemmatize(w.replace('^-1', '')) for w in triple_2 if w not in stop]

    return lv.setratio(triple_1,triple_2)




