import itertools
import re
from typing import Any, Dict, Iterable, List

from pymystem3 import Mystem
from tqdm import tqdm


class Tokenizer:
    def __init__(self):
        self.stemmer = Mystem()
        with open('stop_words.txt') as f:
            self.stop_words = f.read().split('\n')

    def _chunkify(self, iterable: Iterable[Any], chunk_size: int):
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, chunk_size))
            if not chunk:
                return
            yield chunk

    def _make_vocabulary(
        self,
        lemmas: List[str]
    ) -> Dict[str, int]:
        voc = {}
        for lemma in lemmas:
            if lemma in voc.keys():
                voc[lemma] += 1
            else:
                voc[lemma] = 1

        return voc

    def _clean(
        self,
        lemmas: List[str]
    ) -> Dict[str, int]:
        """
        Remove punctuation and stop words from list of lemmas

        """
        cleaned = [
            lemma for lemma in lemmas
            if re.search(r'[\W_]', lemma) is None and lemma not in self.stop_words 
        ]

        return cleaned

    def lemmatize_step(
        self,
        text: str,
        clean: bool = True
    ) -> List[str]:
        """
        Lemmatize the input text.
        
        """
        lemmas = self.stemmer.lemmatize(text)
        if clean:
            lemmas = self._clean(lemmas)

        return lemmas

    def lemmatize_batch(
        self,
        batch: Iterable[str],
        texts_separator: str = "<sep>",
        clean: bool = True
    ) -> List[List[str]]:
        """
        Lemmatize the input batch of texts.
        
        """
        merged_text = texts_separator.join(batch)

        text = []
        batch_of_lemmas = []

        for lemma in self.lemmatize_step(merged_text):
            if lemma != texts_separator:
                if clean:
                    if re.search(r'[\W_]', lemma) is None and lemma not in self.stop_words:
                        text.append(lemma)
                else:
                    text.append(lemma)
            else:
                batch_of_lemmas.append(text)
                text = []

        return batch_of_lemmas

    def create_inverted_index(
        self,
        texts: Dict[Any, str],
        clean: bool = True,
        progress: bool = True
    ) -> Dict[str, Dict[Any, int]]:
        """
        Dummy creation of inverted index using dictionaries
        
        """

        inverted_index = {}
        if progress:
            iterator = tqdm(texts.items())
        else:
            iterator = texts.items()

        for text_id, text in iterator:
            for word in self.lemmatize_step(text, clean):
                if word in inverted_index.keys():
                    if text_id in inverted_index[word].keys():
                        inverted_index[word][text_id] += 1
                    else:
                        inverted_index[word][text_id] = 1
                else:
                    inverted_index[word] = {text_id: 1}
        
        return inverted_index


if __name__ == "__main__":
    story = 'Парни рассказывали, как на демо заказчик нажал на какую-то кнопку, а там алерт выскочил "Куда жмёшь придурок".'\
        'Заказчика оказался с юмором и не предал значения, но было неприятненько'\
        '\n'\
        'Одного уволили за то, что выкатил на прод, а там в форме плейсхолдер <ДАННЫЕ_УДАЛЕНЫ>.'\
        'Поэтому да, аккуратнее надо'
    print(story)
    tokenizer = Tokenizer()
    print(tokenizer.lemmatize_step(story))  # Create lemmas of text without cleaning
    print(tokenizer.lemmatize_step(story))  # Create lemmas of text
