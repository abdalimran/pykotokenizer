from konlpy.tag import Kkma

class KoKkma:
    def __init__(self):
        self.kkma = Kkma()
        
    def __call__(self, text):
        tokenized = self.kkma.morphs(text)
        tokenized_sent = ' '.join(tokenized)
        return tokenized_sent.strip(' \n\t')