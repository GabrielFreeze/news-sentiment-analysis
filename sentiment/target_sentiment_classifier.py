from tqdm import tqdm
import multiprocessing
from NewsSentiment import TargetSentimentClassifier
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class NER():
    def __init__(self, model_name:str='dslim/bert-large-NER'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)


    def classify(self, txts :list[str], multiproc:bool=True, cpu_count:int=0):

        if not multiproc:
            return self._classify(txts)
        
        if not cpu_count:
            cpu_count = multiprocessing.cpu_count()

        chunk_size = len(txts) // cpu_count
        txt_chunks = [txts[i:i + chunk_size] for i in range(0, len(txts), chunk_size)]
        
        with multiprocessing.Pool() as pool:
            results = pool.map(self._classify, txt_chunks)

        return [item for sublist in results for item in sublist]


    def _classify(self, txts: list[str]):
        new_tags = []
        for txt in tqdm(txts):
            txt = txt.replace('’',"'")
            tags = self.nlp_pipeline(txt)
        
            spaces = [-1] + \
                    [i for i,c in enumerate(txt)
                        if c in [' ',',','.'] or
                        i+1<len(txt) and f'{c}{txt[i+1]}' == "'s"
                    ] + \
                    [len(txt)]

            b_tags = [
                i for i,tag in enumerate(tags)
                if tag['entity'].startswith('B-')
            ]

            start = len(txt)
            end = -1

            this_txt = []

            for i,_ in enumerate(b_tags):
                k = tags[b_tags[i]]['start']

                #This means this tag is an errenous sub-token tag and the previous tag already covered it
                if k < end:
                    continue

                #Set start of tag just after the previous space
                start = [j for j in spaces if min(j-k,0)][-1]+1

                #Get end of tag and set it just before the next space
                end = tags[
                    b_tags[i+1]-1 if i+1 < len(b_tags)
                    else -1
                ]['end']-1
                end = [j for j in spaces if max(j-end,0)][0]

                this_txt.append({
                    "entity":tags[b_tags[i]]['entity'][2:],
                    "start":start, "end":end,
                    "text":txt[start:end]
                })
            
            #Hard-coding "Prime Minister" as a Named Entity
            start_idx = 0
            while "Prime Minister" in txt[start_idx:]:
                start_idx = txt.index("Prime Minister", start_idx)
                this_txt.append({
                    "entity": "PER",
                    "start": start_idx,
                    "end": start_idx + len("Prime Minister"),
                    "text": "Prime Minister"
                })
                start_idx += len("Prime Minister")
            
            new_tags.append(this_txt)

        return new_tags


class TSC():
    def __init__(self):
        self.model = TargetSentimentClassifier()

    def classify(self,txt_tags):

        data = [(txt[           :t['start']],
                 txt[ t['start']:t[ 'end' ]],
                 txt[ t[ 'end' ]:          ])
                for txt,t in txt_tags]

        sentiments = self.model.infer(targets=data)

        return sentiments

