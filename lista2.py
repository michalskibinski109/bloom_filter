from nltk.corpus import movie_reviews
import nltk
from bloom_filter2 import BloomFilter
from time import time
import matplotlib.pyplot as plt
from nltk.metrics.distance import edit_distance

class ValidityCheck:
    def __init__(self, words) -> None:
        self.bloomTime = None
        self.naiveTime = None
        self.mostCommonWords = []
        self.words = [word for word in words]
        self.bloomModel = BloomFilter(max_elements=len(words), error_rate=0.1)
        [self.bloomModel.add(word) for word in self.words]

    def prepareText(self, text):
        if type(text) == str:
            text = text.replace('.', '')
            text = text.replace(',', '')
            text = text.replace(';', '')
            text = text.replace(':', '')
            text = set(text.split(' '))
        text = set({word for word in text})
        return text

    def naiveMethod(self, object):
        text = self.prepareText(object.words())
        freq = nltk.FreqDist(object.words())
        invalidWords = {}
        start = time()
        for word in text:
            if word not in self.words:
                invalidWords[word] = freq[word]
        self.naiveTime = time() - start
        self.mostCommonWords = list(
            dict(sorted(invalidWords.items(), key=lambda item: item[1])).keys())[-10:]
        return len(invalidWords)

    def autoCorrect(self, words = []):
        if len(words)< 1:
            words = self.mostCommonWords
        bestFit = []
        for e, word in enumerate(words):
            bestFit.append([])

            temp = [edit_distance(w, word) for w in self.words]
            for _ in range(3):
                bestFit[e].append(self.words[temp.index(min(temp))])
                temp[temp.index(min(temp))] = 100
        return bestFit
        
    def bloomFilter(self, object):
        text = self.prepareText(object.words())
        freq = nltk.FreqDist(object.words())
        invalidWords = {}
        start = time()
        for word in text:
            if word not in self.bloomModel:
                invalidWords[word] = freq[word]
        self.bloomTime = time() - start
        return len(invalidWords)

    def plotTime(self):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(['naive time', 'bloom time'], (self.naiveTime, self.bloomTime))
        ax.set_ylabel('time [sec]')
        ax.set_title('times of finding invalid words')
        plt.show()


# words = set(nltk.corpus.abc.words())
# movies = nltk.corpus.movie_reviews
# model = ValidityCheck(words)

# print( f'total words: {len(set(movies.words()))}, bad words: {model.naiveMethod(movies)}, time: {model.naiveTime}')
# print(
#     f'total words: {len(set(movies.words()))}, bad words: {model.bloomFilter(movies)}, time: {model.bloomTime}')
# validWords = model.autoCorrect()
# invalidWords = model.mostCommonWords

# for i in range(10):
#     print(f'{invalidWords[i]} -> {validWords[i]}')