import argparse
from dataclasses import dataclass
import json

import dacite
import psycopg2
from tqdm import tqdm

from tokenizer import Tokenizer


@dataclass
class Config:
    user: str
    password: str
    host: str
    port: str
    database: str


def main(config: Config):
    """
    1. Connect to PostgreSQL database using given config
    2. Create inverted index from straight
    3. Save it as table in PostgreSQL database
    
    """
    # Create connection with database from config
    connection = psycopg2.connect(
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        database=config.database
    )

    # Create cursor to communicate with 
    with connection.cursor() as cursor:
        # Get parsed data from the database table
        print("Readig data")
        cursor.execute("SELECT url, parsed_data FROM pages;")
        texts = cursor.fetchall()
        # Create `Tokenizer` instance
        tokenizer = Tokenizer()
        # Create inverted index
        print("Creating inverted index") 
        inverted_index = tokenizer.create_inverted_index(
            dict(texts))
        # Create new table for inverted index
        cursor.execute(
            "CREATE TABLE inverted_index (\n"
            "    id     bigserial PRIMARY KEY,\n"
            "    word   text,\n"
            "    url    text REFERENCES pages(url) ON DELETE CASCADE,\n"
            "    count  smallint\n"
            ");"
        )
        cursor.execute(
            "ALTER TABLE inverted_index ADD UNIQUE (word, url);"
        )
        # Create index for table for faster search
        cursor.execute(
            "CREATE INDEX ON inverted_index (word);"
        )
        print("Loading inverted index to PostgreSQL database")
        # Add data to new table
        for word, posting_dict in tqdm(inverted_index.items()):
            for url, count in posting_dict.items():
                cursor.execute(
                    "INSERT INTO inverted_index(word, url, count) VALUES(%s, %s, %s);",
                    (word, url, count)
                )

    connection.commit()
    connection.close()
    # cursor.close()


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

    config = dacite.from_dict(Config, config_json)

    main(config)
