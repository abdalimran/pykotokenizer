from konlpy.tag import Hannanum

class KoHannanum:
    def __init__(self):
        self.hannanum = Hannanum()
        
    def __call__(self, text):
        tokenized = self.hannanum.morphs(text)
        tokenized_sent = ' '.join(tokenized)
        return tokenized_sent.strip(' \n\t')