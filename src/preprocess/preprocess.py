import utils
from utils import remove_emoji, remove_punctuation, add_G_TOKEN, remove_G_TOKEN
from consts import consts
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

c = consts()


class Preprocessor:
    def __init__(self, path_corpus, num_cores=8, use_cache=False):
        self.use_cache = use_cache
        self.num_cores = num_cores

        # establish preprocess folder
        self.path_corpus = Path(path_corpus)
        self.dir_corpus = self.path_corpus.parent
        self.dir_preprocess = self.dir_corpus / f"preprocess-{c.LM_NAME_SUFFIX}"
        self.dir_preprocess.mkdir(exist_ok=True)

        # path_tokenized_corpus: wordpieces tokenized with huggingface LM tokenizer
        # path_tokenized_id_corpus: tokenized wordpiece ids with word boundaries
        self.path_tokenized_corpus = (
            self.dir_preprocess / f"tokenized.{self.path_corpus.name}"
        )
        self.path_tokenized_id_corpus = (
            self.dir_preprocess / f"tokenized.id.{self.path_corpus.name}"
        )

    @staticmethod
    def _par_tokenize_doc(doc, phobert: bool = True):
        docid = doc["_id_"]
        sents = doc["sents"]
        if phobert:
            sents = [remove_emoji(i) for i in sents]  # Mine
            sents = [remove_punctuation(i) for i in sents]  # Mine
            # tokenize
            # NOTE: add space before each raw sentence to tokenize the first token with GPT_TOKEN for phrase matching
            tokenized_sents = [c.LM_TOKENIZER.tokenize(s) for s in sents]
            tokenized_sents = [add_G_TOKEN(i) for i in tokenized_sents]  # Mine
        else:
            tokenized_sents = [c.LM_TOKENIZER.tokenize(" " + s) for s in sents]

        cleaned_tokenized_sents = []
        for tokens in tokenized_sents:
            tokens_batch = utils.get_batches(tokens, batch_size=c.MAX_SENT_LEN)
            cleaned_tokenized_sents += tokens_batch

        tokenized_doc = {
            "_id_": docid,
            "sents": [" ".join(tokens) for tokens in cleaned_tokenized_sents],
        }

        tokenized_id_doc = {"_id_": doc["_id_"], "sents": []}
        for tokens in cleaned_tokenized_sents:
            widxs = [
                i for i, token in enumerate(tokens) if token.startswith(c.GPT_TOKEN)
            ]  # the indices of start of words
            if phobert:
                tokens = remove_G_TOKEN(tokens)
                
            ids = c.LM_TOKENIZER.convert_tokens_to_ids(tokens)
            tokenized_id_doc["sents"].append({"ids": ids, "widxs": widxs})
        return tokenized_doc, tokenized_id_doc

    def tokenize_corpus(self):
        if (
            self.use_cache
            and utils.IO.is_valid_file(self.path_tokenized_corpus)
            and utils.IO.is_valid_file(self.path_tokenized_id_corpus)
        ):
            print(f"[Preprocessor] Use cache: {self.path_tokenized_corpus}")
            return
        docs = utils.JsonLine.load(self.path_corpus)
        pool = Pool(processes=self.num_cores)
        pool_func = pool.imap(func=Preprocessor._par_tokenize_doc, iterable=docs)
        doc_tuples = list(
            tqdm(
                pool_func,
                total=len(docs),
                ncols=100,
                desc=f"[Tokenize] {self.path_corpus}",
            )
        )
        tokenized_docs = [doc for doc, iddoc in doc_tuples]
        tokenized_id_docs = [iddoc for doc, iddoc in doc_tuples]
        pool.close()
        pool.join()
        utils.JsonLine.dump(tokenized_docs, self.path_tokenized_corpus)
        utils.JsonLine.dump(tokenized_id_docs, self.path_tokenized_id_corpus)
