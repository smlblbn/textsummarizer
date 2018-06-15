from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from nltk.tokenize import sent_tokenize
import numpy as np
import re
import sys
from rouge.rouge_score import rouge_l_summary_level, rouge_n

class Summarizer():
    '''
    a text summarizer that uses lsa or lda
    '''
    def __init__(self):
        self.train_document = []
        self.label_document = []

        self.train_sentences = []
        self.label_sentences = []

        self.lsa_evaluated_sentences = []
        self.lda_evaluated_sentences = []

        self.count = CountVectorizer()
        self.tfidf = TfidfTransformer()

        self.lsa = TruncatedSVD(n_components=10, random_state=463)
        self.lda = LatentDirichletAllocation(n_components=10, random_state=463)

    def read_data(self, file_train, file_label):

        with open(file_train, 'r') as file:
            lines = file.readlines()

            for line in lines:

                if line.strip():
                    self.train_document.append(line.rstrip())
                    self.train_sentences.append(sent_tokenize(line.rstrip()))

        with open(file_label, 'r') as file:
            lines = file.readlines()

            for line in lines:
                self.label_document.append(line.rstrip())
                self.label_sentences.append(sent_tokenize(line.rstrip()))

    def lsa_summary(self, idx, type='test'):
        '''
        return a shorter version of the given text
        '''

        sentence_regex = []
        for sentence in self.train_sentences[idx]:
            sentence = re.sub('-lrb-', '', sentence)
            sentence = re.sub('-rrb-', '', sentence)
            sentence = re.sub('[^a-zA-Z_ ]', '', sentence)
            sentence_regex.append(sentence)

        count_matrix = self.count.fit_transform(sentence_regex)
        tfidf_matrix = self.tfidf.fit_transform(count_matrix)
        lsa_res = self.lsa.fit_transform(tfidf_matrix)

        val, count = np.unique(np.argmax(lsa_res, axis=1), return_counts=True)
        list1, list2 = zip(*sorted(zip(lsa_res[:, val[np.argmax(count)]].reshape(-1), np.arange(lsa_res.shape[0]))))

        indices = list(reversed(list2))

        if(type=='train'):
            indices = indices[0: len(self.label_sentences[idx])]
        else:
            indices = indices[0:2]

        evaluated = [self.train_sentences[idx][i] for i in tuple(indices)]

        self.lsa_evaluated_sentences.append(evaluated)

    def lda_summary(self, idx, type='test'):
        '''
        return a shorter version of the given text
        '''

        sentence_regex = []
        for sentence in self.train_sentences[idx]:
            sentence = re.sub('-lrb-', '', sentence)
            sentence = re.sub('-rrb-', '', sentence)
            sentence = re.sub('[^a-zA-Z_ ]', '', sentence)
            sentence_regex.append(sentence)

        count_matrix = self.count.fit_transform(sentence_regex)
        tfidf_matrix = self.tfidf.fit_transform(count_matrix)
        lda_res = self.lda.fit_transform(tfidf_matrix)

        val, count = np.unique(np.argmax(lda_res, axis=1), return_counts=True)
        list1, list2 = zip(*sorted(zip(lda_res[:, val[np.argmax(count)]].reshape(-1), np.arange(lda_res.shape[0]))))

        indices = list(reversed(list2))

        if(type=='train'):
            indices = indices[0: len(self.label_sentences[idx])]
        else:
            indices = indices[0:2]

        evaluated = [self.train_sentences[idx][i] for i in tuple(indices)]

        self.lda_evaluated_sentences.append(evaluated)

if __name__ == '__main__':

    print("Usage of script"
          "for training \n"
          "python summarizer train.txt train_short.txt \n"
          "for testing \n"
          "python summarizer \n"
          "paste the test document as input \n")

    if(len(sys.argv) == 1):

        document = input('Enter document: ')

        summarizer = Summarizer()

        summarizer.train_document.append(document)
        summarizer.train_sentences.append(sent_tokenize(document))

        summarizer.lsa_summary(0, type='test')
        summarizer.lda_summary(0, type='test')

        print('lsa summary sentences: ' + ' '.join(summarizer.lsa_evaluated_sentences[0]) + '\n')
        print('lda summary sentences: ' + ' '.join(summarizer.lda_evaluated_sentences[0]) + '\n')

    if(len(sys.argv) == 3):

        summarizer = Summarizer()
        summarizer.read_data(sys.argv[1], sys.argv[2])

        size_doc = len(summarizer.label_document)

        for idx in range(size_doc):
            summarizer.lsa_summary(idx, type='train')
            summarizer.lda_summary(idx, type='train')

        with open('train_summary.txt', 'w') as file:
            for idx in range(size_doc):
                file.write('lsa summary sentences ' + str(idx) + ': ' + ' '.join(summarizer.lsa_evaluated_sentences[idx]) + '\n')
                file.write('lda summary sentences ' + str(idx) + ': ' + ' '.join(summarizer.lda_evaluated_sentences[idx]) + '\n')
            file.close()

        sum_rouge_1_p = 0
        sum_rouge_1_r = 0
        sum_rouge_1_f = 0

        sum_rouge_2_p = 0
        sum_rouge_2_r = 0
        sum_rouge_2_f = 0

        sum_rouge_l_p = 0
        sum_rouge_l_r = 0
        sum_rouge_l_f = 0

        for idx in range(size_doc):
            score_1 = rouge_n(summarizer.lsa_evaluated_sentences[idx], summarizer.label_sentences[idx], n=1)
            score_2 = rouge_n(summarizer.lsa_evaluated_sentences[idx], summarizer.label_sentences[idx])
            score_l = rouge_l_summary_level(summarizer.lsa_evaluated_sentences[idx], summarizer.label_sentences[idx])

            #print(score_1)
            #print(score_2)
            #print(score_l)

            sum_rouge_1_p += score_1['p']
            sum_rouge_1_r += score_1['r']
            sum_rouge_1_f += score_1['f']

            sum_rouge_2_p += score_2['p']
            sum_rouge_2_r += score_2['r']
            sum_rouge_2_f += score_2['f']

            sum_rouge_l_p += score_l['p']
            sum_rouge_l_r += score_l['r']
            sum_rouge_l_f += score_l['f']

        sum_rouge_1_p /= size_doc
        sum_rouge_1_r /= size_doc
        sum_rouge_1_f /= size_doc

        sum_rouge_2_p /= size_doc
        sum_rouge_2_r /= size_doc
        sum_rouge_2_f /= size_doc

        sum_rouge_l_p /= size_doc
        sum_rouge_l_r /= size_doc
        sum_rouge_l_f /= size_doc

        print('lsa_rouge_1_p', sum_rouge_1_p)
        print('lsa_rouge_1_r', sum_rouge_1_r)
        print('lsa_rouge_1_f', sum_rouge_1_f)

        print('lsa_rouge_2_p', sum_rouge_2_p)
        print('lsa_rouge_2_r', sum_rouge_2_r)
        print('lsa_rouge_2_f', sum_rouge_2_f)

        print('lsa_rouge_l_p', sum_rouge_l_p)
        print('lsa_rouge_l_r', sum_rouge_l_r)
        print('lsa_rouge_l_f', sum_rouge_l_f)

        sum_rouge_1_p = 0
        sum_rouge_1_r = 0
        sum_rouge_1_f = 0

        sum_rouge_2_p = 0
        sum_rouge_2_r = 0
        sum_rouge_2_f = 0

        sum_rouge_l_p = 0
        sum_rouge_l_r = 0
        sum_rouge_l_f = 0

        for idx in range(size_doc):
            score_1 = rouge_n(summarizer.lda_evaluated_sentences[idx], summarizer.label_sentences[idx], n=1)
            score_2 = rouge_n(summarizer.lda_evaluated_sentences[idx], summarizer.label_sentences[idx])
            score_l = rouge_l_summary_level(summarizer.lda_evaluated_sentences[idx], summarizer.label_sentences[idx])

            #print(score_1)
            #print(score_2)
            #print(score_l)

            sum_rouge_1_p += score_1['p']
            sum_rouge_1_r += score_1['r']
            sum_rouge_1_f += score_1['f']

            sum_rouge_2_p += score_2['p']
            sum_rouge_2_r += score_2['r']
            sum_rouge_2_f += score_2['f']

            sum_rouge_l_p += score_l['p']
            sum_rouge_l_r += score_l['r']
            sum_rouge_l_f += score_l['f']

        sum_rouge_1_p /= size_doc
        sum_rouge_1_r /= size_doc
        sum_rouge_1_f /= size_doc

        sum_rouge_2_p /= size_doc
        sum_rouge_2_r /= size_doc
        sum_rouge_2_f /= size_doc

        sum_rouge_l_p /= size_doc
        sum_rouge_l_r /= size_doc
        sum_rouge_l_f /= size_doc

        print('lda_rouge_1_p', sum_rouge_1_p)
        print('lda_rouge_1_r', sum_rouge_1_r)
        print('lda_rouge_1_f', sum_rouge_1_f)

        print('lda_rouge_2_p', sum_rouge_2_p)
        print('lda_rouge_2_r', sum_rouge_2_r)
        print('lda_rouge_2_f', sum_rouge_2_f)

        print('lda_rouge_l_p', sum_rouge_l_p)
        print('lda_rouge_l_r', sum_rouge_l_r)
        print('lda_rouge_l_f', sum_rouge_l_f)