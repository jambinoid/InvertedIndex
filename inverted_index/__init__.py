from dataclasses import dataclass, field
from functools import partial
import heapq
from itertools import starmap
import math
from typing import Any, List, Optional

import psycopg2
from tqdm import tqdm

from inverted_index.tokenizer import Tokenizer


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
        default="url", init=False)
    iindex_count_col: Optional[str] = field(
        default="count", init=False)
    dlens_table: Optional[str] = field(
        default="pages_len", init=False)
    dlens_docid_col: Optional[str] = field(
        default="url", init=False)
    dlens_len_col: Optional[str] = field(
        default="length", init=False)
    src_table: Optional[str] = field(
        default="pages", init=False)
    src_docid_col: Optional[str] = field(
        default="url", init=False)
    src_doc_col: Optional[str] = field(
        default="parsed_data", init=False)


class InvertedIndex:
    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: str,
        database: str,
        iindex_table: str = "inverted_index",
        iindex_term_col: str = "word",
        iindex_docid_col: str = "url",
        iindex_count_col: str = "count",
        dlens_table: str = "pages_len",
        dlens_docid_col: str = "url",
        dlens_len_col: str = "length",
        clean: bool = True
    ):
        """
        Wrapper for inverted index stored as table in
        PostgreSQL database. Performs creation tables
        of parsed data (not implemented yet) and search
        using chosen metric 
        
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

        self.tokenizer = Tokenizer()
        self.clean = clean

    def __del__(self):
        self.connection.close()
        del self.tokenizer

    def create(
        self,
        src_table: str,
        src_docid_col: str,
        src_doc_col: str
    ):
        """
        Create inverted index and put in to the Postgres
        database
        
        """
        with self.connection.cursor() as cursor:
            # Get parsed data from the database table
            print("Readig data")
            cursor.execute(f"SELECT {src_docid_col}, {src_doc_col} FROM {src_table};")
            docs = cursor.fetchall()
            inverted_index, docs_lens = self.tokenizer.create_inverted_index(
                dict(docs), clean=self.clean)
            # Create new table for docs lengths
            cursor.execute(
                f"CREATE TABLE {self.dlens_table} (\n"
                 "    id                        bigserial PRIMARY KEY,\n"
                f"    {self.dlens_docid_col}    text REFERENCES pages(url) ON DELETE CASCADE,\n"
                f"    {self.dlens_len_col}      int\n"
                 ");"
            )
            print("Loading list of text length to PostgreSQL database")
            # Add data to created table
            for doc_id, doc_len in tqdm(docs_lens.items()):
                cursor.execute(
                    f"INSERT INTO {self.dlens_table}"
                    f"({self.dlens_docid_col}, {self.dlens_len_col})\n"
                    "VALUES(%s, %s);",
                    (doc_id, doc_len)
                )
            # Create new table for inverted index
            cursor.execute(
                f"CREATE TABLE {self.iindex_table} (\n"
                 "    id                         bigserial PRIMARY KEY,\n"
                f"    {self.iindex_term_col}     text,\n"
                f"    {self.iindex_docid_col}    text REFERENCES pages(url) ON DELETE CASCADE,\n"
                f"    {self.iindex_count_col}    smallint\n"
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
            for word, posting_dict in tqdm(inverted_index.items()):
                for url, count in posting_dict.items():
                    cursor.execute(
                        f"INSERT INTO {self.iindex_table}"
                        f"({self.iindex_term_col}, {self.iindex_docid_col}, {self.iindex_count_col})\n"
                        "VALUES(%s, %s, %s);",
                        (word, url, count)
                    )
        # Commit changes
        self.connection.commit()

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
        len_coeff: float = .5
    ) -> List[Any]:
        """
        Perform search
        
        """

        query_pattern = f"SELECT {self.iindex_docid_col}, {self.iindex_count_col} "\
            f"FROM {self.iindex_table} WHERE {self.iindex_term_col} = %s;"

        terms = self.tokenizer.lemmatize_step(query, self.clean)

        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT {self.dlens_docid_col}, {self.dlens_len_col} FROM {self.dlens_table};")
            docs_lens = dict(cursor.fetchall()) 
            n_docs = len(docs_lens)
            docs_len_avg = sum(docs_lens.values()) / n_docs
            
            df_ts = dict()
            for term in terms:
                cursor.execute(
                    f"SELECT count(*) FROM {self.iindex_table} "
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
            # print(bm25_score)
            # If doc is in current top25 then push it to heap
            if bm25_score > top_heap[0][0]:
                heapq.heapreplace(top_heap, (bm25_score, lowest_docid))
            # Update state for cursors which had considered doc as their state   
            for term in doc_terms:
                row = cursors_states[term].fetchone()
                if row:
                    cursors_states[term] = row[0]  # doc_id
                    tf_tds[term] = row[1]  # tf_td
                else:
                    cursors_states.pop(term)
            
        # Close all cursors
        for cursor in cursors_dict.values():
            cursor.close()
        
        # Return sorted by scores list of top documents ids
        return [
            doc_id for _, doc_id in sorted(top_heap, key=lambda x: x[0])
            if doc_id != 'url'
        ]