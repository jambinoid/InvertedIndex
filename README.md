# InvertedIndex

Script to create inverted index for collection of documents.

Required packages are described in [requirements.txt](requirements.txt).

## Installation

```
git clone -b index_as_object git@github.com:jambinoid/InvertedIndex.git
python3 setup.py install
```

## Usage

To create inverted index specify PostgreSQL database config (e.g, [config.json](config.json)) and run following script:

```
python3 tools/create_index.py --config config.json
```

Example of search you can find in [examples](examples) and run in with following script:

```
python3 examples/search.py --config config.json
```

Or run API

```
python3 api.py --config config.json
```

And send request like this

http://localhost:5005/search?q=github&top=10

You can omit `top` param. By default it's 25.

You have to run [Parser Manager](https://github.com/mechnicov/parser-manager) to make this example work.
