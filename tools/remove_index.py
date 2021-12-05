import argparse
import json

import dacite

from inverted_index import InvertedIndex, PostgresConfig


def main(config: PostgresConfig):
    """
    Create inverted index database using InvertedIndex
    wrapper for postgres database

    """
    iindex = InvertedIndex(
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        database=config.database,
        iindex_table=config.iindex_table,
        iindex_term_col=config.iindex_term_col,
        iindex_docid_col=config.iindex_docid_col,
        iindex_count_col=config.iindex_count_col,
        dlens_table=config.dlens_table,
        dlens_docid_col=config.dlens_docid_col,
        dlens_len_col=config.dlens_len_col,
        encoding=config.encoding,
        clean=True
    )
    iindex.remove(with_additional_tables=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Inverted index removal"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to PostgreSQL DB .json config"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config_json = json.load(f)

    config = dacite.from_dict(PostgresConfig, config_json)

    main(config)