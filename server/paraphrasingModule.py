import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModel
from arabert.preprocess import ArabertPreprocessor
#import tkseem as tk
import nltk
from termcolor import colored

import re

# Use NLTK's data downloader to download the required data packages (WordNet and Open Multilingual Wordnet) if not present already
for resource in ["wordnet", "omw-1.4"]:
    try:
        nltk_path = nltk.find("corpora/{0}".format(resource))
    except Exception:
        nltk.download(resource)

from nltk.corpus import wordnet


class araParaphraser:
    def __init__(self, num_beams=10):
        # self.module_dir = os.path.dirname(__file__)

        if(torch.cuda.is_available()):
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu:0")

        # Arabic Tokenizer & Model for Paraphrasing
        print(colored("INFO", "green"),":\t  Loading Arabic Tokenizer & Model for Paraphrasing.")

        paraphraser_model_name="malmarjeh/t5-arabic-text-summarization"
        self.preprocessor = ArabertPreprocessor(model_name="")

        self.tokenizer = AutoTokenizer.from_pretrained(paraphraser_model_name, return_tensors='pt')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_model_name).to(self.device)
        self.pipeline = pipeline("text2text-generation",model=self.model,tokenizer=self.tokenizer)

        self.num_beams = num_beams


    def paraphrase_text(self, text):
        text = self.preprocessor.preprocess(text)
        paraphrased_text = self.pipeline(text,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=self.num_beams,
            repetition_penalty=3.0,
            max_length=200,
            length_penalty=1.0,
            no_repeat_ngram_size = 3)[0]['generated_text']
        return paraphrased_text


class KeywordSynonyms:
    def __init__(self):
        #KeyBERT model for Keyword Extraction
        print(colored("INFO", "green"),":\t  Loading Arabert Model for Keyword Extraction.")
        self.keyword_extraction_model = "aubmindlab/bert-base-arabertv2"
        self.arabert_prep = ArabertPreprocessor(model_name=self.keyword_extraction_model)
        model = AutoModel.from_pretrained(self.keyword_extraction_model)
        model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.keyword_extraction_model)

        # load the model of Takseem
        # print(colored("INFO", "green"),":\t  Loading Takseem tokenizer for Keyword Extraction.")
        # self.tokenizer = tk.WordTokenizer()
        # self.tokenizer.load_model(PATH+'vocab.pl')
        

    def clean_arabic_text(self, text):
        # Remove any non-Arabic characters or symbols
        arabic_pattern = r'[\u0600-\u06FF\s]+'
        return re.findall(arabic_pattern, text)
        
        
    def extractKeywords(self, text):
        text_preprocessed = self.arabert_prep.preprocess(text)
        #inputs is a dictionary containing inputs_ids, attention_masks and token_type_ids as pytorch tensors
        inputs = self.tokenizer.encode_plus(text_preprocessed, return_tensors='pt')
        keywords = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])# some tokens might be split with ## by the tokenizer
        return keywords[1:-1]
        # return self.tokenizer.tokenize(text)
        
    
    def getSynonyms(self, word, max_synonyms=6):
        synonyms = []
        for syn in wordnet.synsets(word,lang='arb'):
            for l in syn.lemmas():
                synonyms.append(l.name().replace("+", " "))
                # Multi-word synonyms contain a '_' between the words, which needs to be replaced with a ' '
        
        return [x for x in list(set(synonyms))][:max_synonyms]
        # Consider those synonyms that are not the same as the original word
    
    def getSynonymsForKeywords(self, text, max_synonyms=6):
        kw_syn = {}
        cleaned_text = self.clean_arabic_text(text)
        keywords = self.extractKeywords(cleaned_text)
        for word in keywords:
            synonyms = self.getSynonyms(word, max_synonyms)
            if len(synonyms) > 0: 
                kw_syn[word] = synonyms
        return kw_syn