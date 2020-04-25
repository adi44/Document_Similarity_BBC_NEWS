from setting import DefaultSetting
from corpus import Corpus
import pickle
from sklearn.neighbors import LSHForest
import codecs


class DocSim:
    def __init__(self):
        self.model = None
        self.doc_corpus = Corpus()
        self.vectorized_docs = []
        self.vectorized_docs_path = ''
        self.lsh = None

    def set_model(self, model):
        self.model = model

    def set_doc(self, doc_corpus):
        self.doc_corpus = doc_corpus

    def vectorized(self, num_topics=DefaultSetting.NUMBER_TOPICS):
        self.lsh = LSHForest(n_estimators=DefaultSetting.HASH_SIZE, n_neighbors=10)
        docs_bow = [self.doc_corpus.dictionary.doc2bow(content.split(u' '))
                    for content in self.doc_corpus.documents]
        for doc_bow in docs_bow:
            vectorized_doc = [x[1] for x in self.model.get_document_topics(doc_bow, minimum_probability=0.0)]
            self.vectorized_docs.append(vectorized_doc)
        self.lsh.fit(self.vectorized_docs)

    def save(self, path=DefaultSetting.DIRECTORY, prefix_name=DefaultSetting.PREFIX_NAME):
        self.vectorized_docs_path = path + '/' + prefix_name + '.plk'
        with open(self.vectorized_docs_path, 'wb') as handle:
            pickle.dump(self.vectorized_docs, handle)

    def load(self, file):
        self.vectorized_docs_path = file
        with open(file, 'rb') as handle:
            self.vectorized_docs = pickle.load(handle)

    def query(self, documents, save=DefaultSetting.SAVE_RESULT_QUERY,
              path=DefaultSetting.DIRECTORY, prefix_name=DefaultSetting.PREFIX_NAME):
        vectorized_docs = []
        for document in documents:
            doc_bow = self.doc_corpus.dictionary.doc2bow(document.split(u' '))
            vectorized_doc = [x[1] for x in self.model.get_document_topics(doc_bow, minimum_probability=0.0)]
            vectorized_docs.append(vectorized_doc)
        distance, indices = self.lsh.kneighbors(vectorized_docs)
        if save:
            saved_file_path = path + '/' + prefix_name + '_res.txt'
            writer = codecs.open(saved_file_path, 'w', 'utf8')
            for i in range(len(documents)):
                writer.write('Input: \n\t' + documents[i] + '\n')
                writer.write('-'*100 + '\n')
                writer.write('Similar documents: \n')
                for idx in indices[i]:
                    writer.write('\tkey: ' + self.doc_corpus.key[idx] + '\n')
                    writer.write('\ttitle: ' + self.doc_corpus.titles[idx] + '\n')
                    writer.write('\tcontent: ' + self.doc_corpus.documents[idx] + '\n\n')
                writer.write('='*100 + '\n')
        else:
            for i in range(len(documents)):
                print('Input: ')
                print(documents[i])
                print('-' * 100)
                print('Similar documents: ')
                for idx in indices[0]:
                    print('\tkey: ', self.doc_corpus.key[idx])
                    print('\ttitle: ', self.doc_corpus.titles[idx])
                    print('\tcontent: ', self.doc_corpus.documents[idx])
                print('=' * 100)
