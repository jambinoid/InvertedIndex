# InvertedIndex

Script to create inverted index for collection of documents.

Required packages are described in [requirements.txt](requirements.txt).

To create inverted index specify PostgreSQL database config (e.g, [config.json](config.json)) and run following script:

```
python3 create_index.py --config config.json
```

You have to run [Parser Manager](https://github.com/mechnicov/parser-manager) to make this example work.