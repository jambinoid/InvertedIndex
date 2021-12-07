from dataclasses import dataclass, field
from functools import partial
import heapq
from itertools import starmap
import math
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from tqdm import tqdm

from inverted_index.tokenizer import Tokenizer
from inverted_index.encoders import GammaEncoder, DeltaEncoder


_ENCODERS_MAP = {
    "gamma": GammaEncoder(),
    "delta": DeltaEncoder()
}

_INT_TYPES = [
    "smallint", "integer", "bigint",
    "smallserial", "serial", "bigserial"
]


@dataclass
class PostgresConfig:
    user: str
    password: str
    host: str
    port: str
    database: str
    iindex_table: Optional[str] = field(
        default="inverted_index", init=False)
    iindex_term_col: Optional[str] = field(
        default="word", init=False)
    iindex_docid_col: Optional[str] = field(
        default="page_id", init=False)
    iindex_count_col: Optional[str] = field(
        default="word_count", init=False)
    dlens_table: Optional[str] = field(
        default="page_lengths", init=False)
    dlens_docid_col: Optional[str] = field(
        default="page_id", init=False)
    dlens_len_col: Optional[str] = field(
        default="length", init=False)
    src_table: Optional[str] = field(
        default="pages", init=False)
    src_docid_col: Optional[str] = field(
        default="id", init=False)
    src_doc_col: Optional[str] = field(
        default="parsed_data", init=False)
    encoding: Optional[str] = field(
        default=None, init=False),
    search_col: Optional[str] = field(
        default="url", init=False)


class InvertedIndex:
    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: str,
        database: str,
        iindex_table: str,
        iindex_term_col: str,
        iindex_docid_col: str,
        iindex_count_col: str,
        dlens_table: str,
        dlens_docid_col: str,
        dlens_len_col: str,
        src_table: str,
        src_docid_col: str,
        src_doc_col: str,
        encoding: str = None,
        clean: bool = True
    ):
        """
        Wrapper for inverted index stored as table in
        PostgreSQL database. Performs creation tables
        of parsed data (not implemented yet) and search
        using chosen metric 
        
        Parameters:
            encoding: str = None
                How to encode the integer numbers in table.
                `None`: no encoding [default]
                `gamma`: Ellias gamma-encoding
                `delta`: Ellias delta-encoding
                To make encoding work, docID must be integer,
                otherwise raise ValueError.
        
        """
        self.connection = psycopg2.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database
        )
        self.iindex_table = iindex_table
        self.iindex_term_col = iindex_term_col
        self.iindex_docid_col = iindex_docid_col
        self.iindex_count_col = iindex_count_col

        self.dlens_table = dlens_table
        self.dlens_docid_col = dlens_docid_col
        self.dlens_len_col = dlens_len_col

        self.src_table = src_table
        self.src_docid_col = src_docid_col
        self.src_doc_col = src_doc_col

        self.tokenizer = Tokenizer()
        self.clean = clean

        if encoding and encoding not in _ENCODERS_MAP.keys():
            raise ValueError(f"There is no such encoding {encoding}")
        self.encoding = encoding
        self.encoder = _ENCODERS_MAP.get(encoding)

    def __del__(self):
        self.connection.close()
        del self.tokenizer

    def _create_dict(
        self,
        texts: Dict[Any, str],
        clean: bool = True,
        progress: bool = True
    ) -> Tuple[Dict[str, Dict[Any, int]], Dict[Any, int]]:
        """
        Dummy creation of inverted index using dictionaries
        
        """

        inverted_index = {}
        words_len = {}
        if progress:
            iterator = tqdm(texts.items())
        else:
            iterator = texts.items()

        for text_id, text in iterator:
            for word in self.tokenizer.lemmatize_step(text, clean):
                if word in inverted_index.keys():
                    if text_id in inverted_index[word].keys():
                        inverted_index[word][text_id] += 1
                    else:
                        inverted_index[word][text_id] = 1
                else:
                    inverted_index[word] = {text_id: 1}
            words_len[text_id] = len(text)
        
        return inverted_index, words_len

    def create(self):
        """
        Create inverted index and put in to the Postgres
        database
        
        """

        with self.connection.cursor() as cursor:
            # Check if tables exists
            cursor.execute(
                 "SELECT EXISTS (\n"
                 "   SELECT FROM information_schema.tables\n" 
                f"   WHERE table_name = '{self.dlens_table}'\n"
                 ");"
            )
            if cursor.fetchone()[0]:
                raise Exception(f"Table {self.dlens_table} already exists")
            # Check if table exists
            cursor.execute(
                 "SELECT EXISTS (\n"
                 "   SELECT FROM information_schema.tables\n" 
                f"   WHERE table_name = '{self.iindex_table}'\n"
                 ");"
            )
            if cursor.fetchone()[0]:
                raise Exception(f"Table {self.iindex_table} already exists")

            # Get parsed data from the database table
            print("Reading data")
            cursor.execute(f"SELECT {self.src_docid_col}, {self.src_doc_col} FROM {self.src_table};")
            docs = cursor.fetchall()
            inverted_index, docs_lens = self._create_dict(
                dict(docs), clean=self.clean)
            # Get DocID datatype
            cursor.execute(
                "SELECT data_type FROM information_schema.columns\n"
                f"WHERE table_name = '{self.src_table}' AND column_name = '{self.src_docid_col}';"
            )
            docid_type = cursor.fetchone()[0]
            if self.encoding:
                if docid_type in _INT_TYPES:
                    docid_type = "bit varying"
                else:
                    raise ValueError(
                        f"DocID type must be one of integer, got {docid_type}")

            # Create new table for docs lengths
            if self.encoding:
                cursor.execute(
                    f"CREATE TABLE {self.dlens_table} (\n"
                     "    id                        bigserial PRIMARY KEY,\n"
                    f"    {self.dlens_docid_col}    {docid_type},\n"
                    f"    {self.dlens_len_col}      {docid_type}\n"
                     ");"
                )
            else:
                # Create new table for docs lengths
                cursor.execute(
                    f"CREATE TABLE {self.dlens_table} (\n"
                     "    id                        bigserial PRIMARY KEY,\n"
                    f"    {self.dlens_docid_col}    {docid_type} REFERENCES {self.src_table}({self.src_docid_col}) ON DELETE CASCADE,\n"
                    f"    {self.dlens_len_col}      smallint\n"
                     ");"
                )
            print("Loading list of text length to PostgreSQL database")
            # Add data to created table
            if self.encoding:
                for doc_id, doc_len in tqdm(docs_lens.items()):
                    cursor.execute(
                        f"INSERT INTO {self.dlens_table}"
                        f"({self.dlens_docid_col}, {self.dlens_len_col})\n"
                         "VALUES(B%s, B%s);",
                        (self.encoder.encode(doc_id), self.encoder.encode(doc_len))
                    )
            else:
                for doc_id, doc_len in tqdm(docs_lens.items()):
                    cursor.execute(
                        f"INSERT INTO {self.dlens_table}"
                        f"({self.dlens_docid_col}, {self.dlens_len_col})\n"
                        "VALUES(%s, %s);",
                        (doc_id, doc_len)
                    )
            # Create new table for inverted index
            if self.encoding:
                cursor.execute(
                    f"CREATE TABLE {self.iindex_table} (\n"
                    "    id                         bigserial PRIMARY KEY,\n"
                    f"    {self.iindex_term_col}     text,\n"
                    f"    {self.iindex_docid_col}    {docid_type},\n"
                    f"    {self.iindex_count_col}    {docid_type}\n"
                    ");"
                )
            else:
                cursor.execute(
                    f"CREATE TABLE {self.iindex_table} (\n"
                    "    id                         bigserial PRIMARY KEY,\n"
                    f"    {self.iindex_term_col}     text,\n"
                    f"    {self.iindex_docid_col}    {docid_type} REFERENCES {self.src_table}({self.src_docid_col}) ON DELETE CASCADE,\n"
                    f"    {self.iindex_count_col}    integer\n"
                    ");"
                )
            cursor.execute(
                f"ALTER TABLE {self.iindex_table}\n"
                f"ADD UNIQUE ({self.iindex_term_col}, {self.iindex_docid_col});"
            )
            # Create index for table for faster search
            cursor.execute(
                f"CREATE INDEX ON {self.iindex_table} ({self.iindex_term_col});"
            )
            print("Loading inverted index to PostgreSQL database")
            # Add data to new table
            if self.encoding:
                for word, posting_dict in tqdm(inverted_index.items()):
                    for docid, count in posting_dict.items():
                        cursor.execute(
                            f"INSERT INTO {self.iindex_table}"
                            f"({self.iindex_term_col}, {self.iindex_docid_col}, {self.iindex_count_col})\n"
                            "VALUES(%s, %s, %s);",
                            (
                                word,
                                self.encoder.encode(docid),
                                self.encoder.encode(count)
                            )
                        )
            else:
                for word, posting_dict in tqdm(inverted_index.items()):
                    for docid, count in posting_dict.items():
                        cursor.execute(
                            f"INSERT INTO {self.iindex_table}"
                            f"({self.iindex_term_col}, {self.iindex_docid_col}, {self.iindex_count_col})\n"
                            "VALUES(%s, %s, %s);",
                            (word, docid, count)
                        )
        # Commit changes
        self.connection.commit()

    def remove(self, with_additional_tables: bool = True):
        """
        Remove inverted index from the Postgres
        database
        
        """
        with self.connection.cursor() as cursor:
            # Check if table exists and remove
            cursor.execute(
                 "SELECT EXISTS ("
                 "   SELECT FROM information_schema.tables" 
                f"   WHERE table_name = '{self.iindex_table}'"
                 ");"
            )
            if cursor.fetchone()[0]:
                cursor.execute(f"DROP TABLE {self.iindex_table}")
            else:
                print(f"Table {self.iindex_table} do not exist, nothing to remove")
            # Check if tables exists
            if with_additional_tables:
                cursor.execute(
                     "SELECT EXISTS (\n"
                     "   SELECT FROM information_schema.tables\n" 
                    f"   WHERE table_name = '{self.dlens_table}'\n"
                     ");"
                )
                if cursor.fetchone()[0]:
                    cursor.execute(f"DROP TABLE {self.dlens_table}")
                else:
                    print(f"Table {self.dlens_table} do not exist, nothing to remove")
        # Commit changes
        self.connection.commit()
        print("Tables are removed")

    @staticmethod
    def _bm25_term(
        tf_td: int,
        df_t: int,
        dl_d: int,
        dl_d_avg: int,
        n_docs: int,
        k: float,
        b: float
    ) -> float:
        """
        BM25 score for one term from query

        Parameters:
            tf_td: int
                Term frequency in document
            df_t: int
                Frequency of documents with term
            dl_d: int
                Length of document in term of terms
            dl_d_avg: int
                Average length of document in term of terms
            n_docs: int
                Size of collection
            k: float
                Saturation coefficient
            b: float
                Length coefficient

        Returns:
            float
            BM25 score for one term

        """
        return (
            (tf_td * (k + 1)) /
            (tf_td + k * (1 - b + b * dl_d / dl_d_avg)) *
            math.log10((n_docs - df_t + .5) / (df_t + .5))
        )

    def search(
        self,
        query: str,
        top: int = 25,
        saturation_coeff: float = 5.,
        len_coeff: float = .5,
        col_to_return: str = None
    ) -> List[Any]:
        """
        Perform search on inverted index

        Parameters:
            - query: str
                Text query
            - top: int
                Number of top results of a query [default = 25]
            - saturation_coeff: float
                Saturation coefficient [default = .5]
            - len_coeff: float
                Text length coefficient [default = .5]
            - col_to_return: str
                Column of DB to return in result.
                If None [default] return list of docIDs

        Returns:
            List of top documents for given query
        
        """

        query_pattern = f"SELECT {self.iindex_docid_col}, {self.iindex_count_col} "\
            f"FROM {self.iindex_table} WHERE {self.iindex_term_col} = %s;"

        terms = self.tokenizer.lemmatize_step(query, self.clean)

        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {self.dlens_docid_col}, {self.dlens_len_col}\n"
                f"FROM {self.dlens_table};")
            if self.encoding:
                docs_lens = dict([
                    (self.encoder.decode(docid), self.encoder.decode(doc_len))
                    for docid, doc_len in cursor.fetchall()
                ])
            else:
                docs_lens = dict(cursor.fetchall())
            
            n_docs = len(docs_lens)
            docs_len_avg = sum(docs_lens.values()) / n_docs
            
            df_ts = dict()
            for term in terms:
                cursor.execute(
                    f"SELECT count(*) FROM {self.iindex_table}\n"
                    f"WHERE {self.iindex_term_col} = "+ "%s;",
                    (term,)
                )
                df_ts[term] = cursor.fetchone()[0]
            
        top_heap = [(-math.inf, "url")] * top
        # Gotta iterate over many queries in parallel so use many cursors 
        cursors_dict = dict([(term, self.connection.cursor()) for term in terms])

        tf_tds = dict()
        cursors_states = dict()
        for term, cursor in cursors_dict.items():
            cursor.execute(query_pattern, (term,))
            row = cursor.fetchone()
            if row:
                if self.encoding:
                    cursors_states[term] = self.encoder.decode(row[0])  # doc_id
                    tf_tds[term] = self.encoder.decode(row[1])  # tf_td
                else:
                    cursors_states[term] = row[0]  # doc_id
                    tf_tds[term] = row[1]  # tf_td

        # Perform top heap query search
        # Stop when no docs left in every cursor
        while cursors_states:
            # Sort terms by current cursor position (doc_id)
            lowest_docid = min(cursors_states.values())
            # Get keys(terms) of cursor which state is the same as
            # the state of first cursor after sorting by doc_id
            doc_terms = [term for term, state in cursors_states.items() if state == lowest_docid]
            # Get BM25 for query on current doc
            bm25_doc = partial(
                self._bm25_term,
                dl_d=docs_lens[lowest_docid],
                dl_d_avg=docs_len_avg,
                n_docs=n_docs,
                k=saturation_coeff,
                b=len_coeff
            )
            bm25_score = sum(
                starmap(bm25_doc, ((tf_tds[term], df_ts[term]) for term in doc_terms)))
            # If doc is in current top25 then push it to heap
            if bm25_score > top_heap[0][0]:
                heapq.heapreplace(top_heap, (bm25_score, lowest_docid))
            # Update state for cursors which had considered doc as their state   
            for term in doc_terms:
                row = cursors_dict[term].fetchone()
                if row:
                    if self.encoding:
                        cursors_states[term] = self.encoder.decode(row[0])  # doc_id
                        tf_tds[term] = self.encoder.decode(row[1])  # tf_td
                    else:
                        cursors_states[term] = row[0]  # doc_id
                        tf_tds[term] = row[1]  # tf_td
                else:
                    cursors_states.pop(term)
            
        # Close all cursors
        for cursor in cursors_dict.values():
            cursor.close()
        
        # Get sorted by scores list of top documents ids
        top_heap = [
            doc_id for _, doc_id in sorted(top_heap, key=lambda x: x[0])
            if doc_id != 'url'
        ]
        if col_to_return:
            with self.connection.cursor() as cursor:
                if len(top_heap) > 1:
                    cursor.execute(
                        f"SELECT {col_to_return} FROM {self.src_table}\n"
                        f"WHERE {self.src_docid_col} IN {tuple(top_heap)};"
                    )
                elif len(top_heap) == 1:
                    cursor.execute(
                        f"SELECT {col_to_return} FROM {self.src_table}\n"
                        f"WHERE {self.src_docid_col} = {top_heap[0]};"
                    )
                else:
                    return []
                return list(*zip(*cursor.fetchall()))
        return top_heap
