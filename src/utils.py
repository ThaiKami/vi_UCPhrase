import gc
import json
import string
import orjson
import torch
import pickle
import shutil
import time
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from termcolor import colored
from functools import lru_cache
from nltk.stem.snowball import SnowballStemmer
import re

PUNCS = set(string.punctuation) - {"-"}
STEMMER = SnowballStemmer("porter", ignore_stopwords=False)


@lru_cache(maxsize=100000)
def stem_word(w):
    return STEMMER.stem(w)


def stem_cand(c):
    return " ".join([stem_word(w) for w in c.split()]).lower()


def get_device(gpu):
    return torch.device("cpu" if gpu is None else f"cuda:{gpu}")


def mean(nums):
    return sum(nums) / len(nums)


def get_batches(input_list, batch_size):
    return [
        input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)
    ]


def get_possible_spans(word_idxs, num_wordpieces, max_word_gram, max_subword_gram):
    possible_spans = []
    num_words = len(word_idxs)
    max_gram = min(max_word_gram, num_words)
    for len_span in range(max_gram, 1, -1):
        for i in range(num_words - len_span + 1):
            l_idx = word_idxs[i]
            r_idx = (
                word_idxs[i + len_span] - 1
                if i + len_span < num_words
                else num_wordpieces - 1
            )
            if r_idx - l_idx + 1 <= max_subword_gram:
                possible_spans.append((l_idx, r_idx))
    return possible_spans


class Log:
    @staticmethod
    def info(message):
        print(colored(message, "green"))


class String:
    @staticmethod
    def removeprefix(s: str, prefix: str) -> str:
        return s[len(prefix) :] if s.startswith(prefix) else s[:]

    def removesuffix(s: str, suffix: str) -> str:
        return s[: -len(suffix)] if suffix and s.endswith(suffix) else s[:]


class IO:
    @staticmethod
    def is_valid_file(filepath):
        filepath = Path(filepath)
        return filepath.exists() and filepath.stat().st_size > 0

    def load(path):
        raise NotImplementedError

    def dump(data, path):
        raise NotImplementedError


# class Json(IO):
#     @staticmethod
#     def load(path):
#         with open(path) as rf:
#             data = json.load(rf)
#         return data

#     @staticmethod
#     def loads(jsonline):
#         return json.loads(jsonline)

#     @staticmethod
#     def dump(data, path):
#         with open(path, 'w') as wf:
#             json.dump(data, wf, indent=4, ensure_ascii=False)


#     @staticmethod
#     def dumps(data):
#         return json.dumps(data, ensure_ascii=False)
def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


def remove_punctuation(text):
    # Create a translation table to remove punctuation characters
    translator = str.maketrans("", "", string.punctuation.replace("_", ""))

    # Use the translate method to remove punctuation
    text_without_punctuation = text.translate(translator)

    return text_without_punctuation


def add_G_TOKEN(tokens_list: list):
    G_TOKEN = "Ġ"
    PUNCTUATIONS = string.punctuation.replace("_", "")
    SKIP_TOKEN = "<\\SKIP>"
    inloop = tokens_list
    idx = 0
    try:
        while True:
            # logic here
            if "@@" in inloop[idx]:
                next_AtAt = idx + 1
                while True:
                    if "@@" not in inloop[next_AtAt]:
                        break
                    else:
                        next_AtAt += 1
                inloop[idx] = G_TOKEN + inloop[idx]
                idx = next_AtAt
            elif inloop[idx] in PUNCTUATIONS:
                inloop[idx] = SKIP_TOKEN
            elif inloop[idx] == SKIP_TOKEN:
                continue
            else:
                inloop[idx] = G_TOKEN + inloop[idx]
            # next step
            idx += 1
            if idx == len(inloop):
                break
        inloop = [i for i in inloop if i != SKIP_TOKEN]
        return inloop
    except IndexError as e:
        return inloop

def remove_G_TOKEN(tokens_list: list):
    G_TOKEN = "Ġ"
    tokens_list = [i.replace(G_TOKEN, "") for i in tokens_list]
    return tokens_list

class OrJson(IO):
    @staticmethod
    def load(path):
        with open(path) as rf:
            data = orjson.loads(rf.read())
        return data

    @staticmethod
    def loads(jsonline):
        return orjson.loads(jsonline)

    @staticmethod
    def dump(data, path):
        with open(path, "w") as wf:
            wf.write(
                orjson.dumps(
                    data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
                ).decode()
            )

    @staticmethod
    def dumps(data):
        return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS).decode()


Json = OrJson


class JsonLine(IO):
    @staticmethod
    def load(path, use_tqdm=False):
        with open(path) as rf:
            lines = rf.read().splitlines()
        if use_tqdm:
            lines = tqdm(lines, ncols=100, desc="Load JsonLine")
        return [json.loads(l) for l in lines]

    @staticmethod
    def dump(instances, path):
        assert type(instances) == list
        lines = [json.dumps(d, ensure_ascii=False) for d in instances]
        with open(path, "w") as wf:
            wf.write("\n".join(lines))


class OrJsonLine(IO):
    @staticmethod
    def load(path):
        with open(path) as rf:
            lines = rf.read().splitlines()
        return [orjson.loads(l) for l in lines]

    @staticmethod
    def dump(instances, path):
        assert type(instances) == list
        lines = [
            orjson.dumps(d, option=orjson.OPT_NON_STR_KEYS).decode() for d in instances
        ]
        with open(path, "w") as wf:
            wf.write("\n".join(lines))


class TextFile(IO):
    @staticmethod
    def load(path):
        with open(path) as rf:
            text = rf.read()
        return text

    @staticmethod
    def readlines(path, skip_empty_line=False):
        with open(path) as rf:
            lines = rf.read().splitlines()
        if skip_empty_line:
            return [l for l in lines if l]
        return lines

    @staticmethod
    def dump(text, path):
        with open(path, "w") as wf:
            wf.write(text)

    @staticmethod
    def dumplist(target_list, path):
        with open(path, "w") as wf:
            wf.write("\n".join([str(o) for o in target_list]) + "\n")


class Pickle:
    @staticmethod
    def load(path):
        with open(path, "rb") as rf:
            gc.disable()
            data = pickle.load(rf)
            gc.enable()
        return data

    @staticmethod
    def dump(data, path):
        with open(path, "wb") as wf:
            gc.disable()
            pickle.dump(data, wf, protocol=4)
            gc.enable()

    @staticmethod
    def batch_dump(instances, dirpath, num_files=10):
        assert type(instances) == list
        dirpath = Path(dirpath)
        if dirpath.exists():
            shutil.rmtree(dirpath)
        dirpath.mkdir(exist_ok=True)
        num_instances = len(instances)
        batch_size = num_instances // num_files
        threads = []
        print("start batch dumping...", end="")
        time1 = time.perf_counter()
        for i in range(0, num_instances, batch_size):
            filepath = dirpath / str(len(threads))
            thread = multiprocessing.Process(
                target=Pickle.dump, args=(instances[i : i + batch_size], filepath)
            )
            threads.append(thread)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time2 = time.perf_counter()
        print(f"OK in {time2-time1:.1f} secs")


class Process:
    @staticmethod
    def par(func, iterables, num_processes, desc=""):
        pool = multiprocessing.Pool(processes=num_processes)
        pool_func = pool.imap(func=func, iterable=iterables)
        pool_func = tqdm(pool_func, total=len(iterables), ncols=100, desc=desc)
        # results = list(pool_func)
        results = [r for r in pool_func]
        pool.close()
        pool.join()
        return results


if __name__ == "__main__":
    print(OrJson.dumps({1: 2, 3: "sheaf"}))
