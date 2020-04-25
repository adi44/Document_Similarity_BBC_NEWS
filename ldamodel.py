from gensim import models
from setting import DefaultSetting


class LDAModel:
    def __init__(self):
        self.model = None
        self.model_path = ''

    def train(self, corpus, dictionary, num_topics=DefaultSetting.NUMBER_TOPICS,
              path=DefaultSetting.DIRECTORY, prefix_name=DefaultSetting.PREFIX_NAME):
        self.model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        self.model_path = path + '/' + prefix_name + '_lda.model'
        self.model.save(self.model_path)

    def load_file(self, path):
        self.model_path = path
        self.model = models.ldamodel.LdaModel.load(self.model_path)

    def show(self, top_words=DefaultSetting.NUMBER_TOP_WORDS, with_prob=DefaultSetting.SHOW_PROBABILITY):
        if with_prob:
            for i in range(self.model.num_topics):
                print('Topic ', i, ': ', self.model.print_topic(i, topn=top_words))
        else:
            for i in range(self.model.num_topics):
                words = [a[0] for a in self.model.show_topic(i, topn=top_words)]
                ss = 'Topic ' + str(i) + ':'
                for word in words:
                    ss += ' ' + word
                print(ss)
