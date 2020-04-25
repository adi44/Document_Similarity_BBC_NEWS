from gensim import corpora
import codecs
from setting import DefaultSetting
import os


class Corpus:
    def __init__(self):
        self.key = []
        self.titles = []
        self.documents = []
        self.stoplist = []
        self.corpus = None
        self.corpus_path = ''
        self.dictionary = None
        self.dictionary_path = ''

    def get_docs(self, stopwords_file=None):
        allfiles=os.listdir("data/files/")
        j=0
        for i in allfiles:
            reader = codecs.open("data/files/"+i, 'r', 'utf8')
            content=reader.read()
            self.documents.append(content)
            self.titles.append(i)
            self.key.append(str(j))
            j=j+1

        stopwords = []
        if stopwords_file is not None:
            sreader = codecs.open(stopwords_file, 'r', 'utf8')
            for line in sreader.readlines():
                stopwords.append(line.replace('\n', ''))
        self.stoplist = set(stopwords)

    def set_docs(self, documents, stoplist=None):
        self.documents = documents
        self.stoplist = stoplist

    def _build_corpus(self, path="tmp", prefix_name="dic"):
        texts = [[word for word in document.split(u' ') if word not in self.stoplist] for document in self.documents]
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary_path = path + '/' + prefix_name + '.dict'
        self.dictionary.save(self.dictionary_path)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.corpus_path = path + '/' + prefix_name + '_bow.mm'
        corpora.MmCorpus.serialize(self.corpus_path, self.corpus)

    def build_corpus(self,stopwords_file=None,
                     path=DefaultSetting.DIRECTORY, prefix_name=DefaultSetting.PREFIX_NAME):
        self.get_docs( stopwords_file)
        self._build_corpus(path, prefix_name)
        

    def load_corpus(self, corpus_file, dictionary_file):
        self.corpus_path = corpus_file
        self.dictionary_path = dictionary_file
        self.corpus = corpora.MmCorpus(corpus_file)
        self.dictionary = corpora.Dictionary.load(dictionary_file)
