import utils
from utils import remove_punctuation, remove_emoji
import consts
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from pyvi import ViTokenizer

class Preprocessor:

    def __init__(
            self,
            path_corpus,
            num_cores=8,
            use_cache=True):
        self.use_cache = use_cache
        self.num_cores = num_cores

        # establish preprocess folder
        self.path_corpus = Path(path_corpus)
        self.dir_corpus = self.path_corpus.parent
        self.dir_preprocess = self.dir_corpus / f'preprocess-{consts.LM_NAME_SUFFIX}'
        self.dir_preprocess.mkdir(exist_ok=True)

        # path_tokenized_corpus: wordpieces tokenized with huggingface LM tokenizer
        # path_tokenized_id_corpus: tokenized wordpiece ids with word boundaries
        self.path_tokenized_corpus = self.dir_preprocess / f'tokenized.{self.path_corpus.name}'
        self.path_tokenized_id_corpus = self.dir_preprocess / f'tokenized.id.{self.path_corpus.name}'

    @staticmethod
    def _par_tokenize_doc(doc, is_phobert: bool = True):
        docid = doc['_id_']
        sents = doc['sents']

        # tokenize
        # NOTE: add space before each raw sentence to tokenize the first token with GPT_TOKEN for phrase matching

        # NOTE: This is OG
        # tokenized_sents = [consts.LM_TOKENIZER.tokenize(' ' + s, add_special_tokens=False) for s in sents]

        # NOTE: This is mine
        if is_phobert:
            sents = [remove_punctuation(s) for s in sents]
            sents = [ViTokenizer.tokenize(s) for s in sents]
            sents = [remove_emoji(s) for s in sents]
            
        tokenized_sents = [consts.LM_TOKENIZER.tokenize(s) for s in sents]
        if is_phobert:
            tokenized_sents = [utils.add_G_TOKEN(s) for s in tokenized_sents]

        cleaned_tokenized_sents = []
        for tokens in tokenized_sents:
            tokens_batch = utils.get_batches(tokens, batch_size=consts.MAX_SENT_LEN)
            cleaned_tokenized_sents += tokens_batch
        tokenized_doc = {'_id_': docid, 'sents': [' '.join(tokens) for tokens in cleaned_tokenized_sents]}

        tokenized_id_doc = {'_id_': doc['_id_'], 'sents': []}
        for tokens in cleaned_tokenized_sents:
            widxs = [i for i, token in enumerate(tokens) if token.startswith(consts.GPT_TOKEN)]  # the indices of start of words
            ids = consts.LM_TOKENIZER.convert_tokens_to_ids(tokens)
            tokenized_id_doc['sents'].append({'ids': ids, 'widxs': widxs})

        return tokenized_doc, tokenized_id_doc

    def tokenize_corpus(self):
        if self.use_cache and utils.IO.is_valid_file(self.path_tokenized_corpus) and utils.IO.is_valid_file(self.path_tokenized_id_corpus):
            print(f'[Preprocessor] Use cache: {self.path_tokenized_corpus}')
            return
        docs = utils.JsonLine.load(self.path_corpus)
        pool = Pool(processes=self.num_cores)
        pool_func = pool.imap(func=Preprocessor._par_tokenize_doc, iterable=docs)
        doc_tuples = list(tqdm(pool_func, total=len(docs), ncols=100, desc=f'[Tokenize] {self.path_corpus}'))
        tokenized_docs = [doc for doc, iddoc in doc_tuples]
        tokenized_id_docs = [iddoc for doc, iddoc in doc_tuples]
        pool.close()
        pool.join()
        utils.JsonLine.dump(tokenized_docs, self.path_tokenized_corpus)
        utils.JsonLine.dump(tokenized_id_docs, self.path_tokenized_id_corpus)
