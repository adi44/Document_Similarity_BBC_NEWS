import logging
from setting import DefaultSetting
from corpus import Corpus
from ldamodel import LDAModel
from docsim import DocSim
import codecs
from datetime import datetime


def main():
    logging.basicConfig(format=DefaultSetting.FORMAT_LOG, level=logging.INFO)

    start_time = datetime.now()

    
    stopwords_file = 'data/stopwords.txt'
    num_topics = 100
    prefix_name = 'demo'
    directory = 'tmp'
    query = 'data/query.txt'

    corpus = Corpus()
    corpus.build_corpus( stopwords_file, directory, prefix_name)
    LDA = LDAModel()
    LDA.train(corpus.corpus, corpus.dictionary, num_topics, directory, prefix_name)
    LDA.show()

    docsim = DocSim()
    docsim.set_model(LDA.model)
    docsim.set_doc(corpus)
    docsim.vectorized(num_topics)
    # docsim.save(directory, prefix_name)

    print('Training time: ', datetime.now() - start_time)

    start_time = datetime.now()
    reader = codecs.open(query, 'r', 'utf8')
    documents = []
    for line in reader.readlines():
        documents.append(line.replace('\n', ''))
    docsim.query(documents, True, directory, prefix_name)
    docsim.query(documents, False, directory, prefix_name)
    print('Query time: ', datetime.now() - start_time)

if __name__ == '__main__':
    main()
