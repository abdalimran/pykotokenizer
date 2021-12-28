from konlpy.tag import Komoran

class KoKomoran:
    def __init__(self):
        self.komoran = Komoran()
        
    def __call__(self, text):
        tokenized = self.komoran.morphs(text)
        tokenized_sent = ' '.join(tokenized)
        return tokenized_sent.strip(' \n\t')