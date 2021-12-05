import argparse
import json
import dacite
from flask import Flask, request

from inverted_index import InvertedIndex, PostgresConfig

app = Flask(__name__)

@app.route("/search")
def search():
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
        src_table=config.src_table,
        src_docid_col=config.src_docid_col,
        src_doc_col=config.src_doc_col,
        encoding=config.encoding,
        clean=True
    )

    query = {"query": request.args.get("q")}

    top = request.args.get("top")

    if top:
        query = {**query, "top": int(top)}

    return {"data": iindex.search(**query, col_to_return=config.search_col)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Inverted index creation"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to PostgreSQL DB .json config"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config_json = json.load(f)

    config = dacite.from_dict(PostgresConfig, config_json)

    app.run(host="0.0.0.0", port=5005, debug=True)
