# Tests for InvertedIndex

Tests use Python3 along with `pytest` library. 

To test system temporary PostgreSQL database created using Docker-Compose, filled with text files from [test_files](test).

Tests include testing time, ranking of search results in all encodings and finding documents with searched word and not finding if there's no such word.

To run tests do the following:

1. Go to test directory

```bash
$ cd test
```

2. Run all the tests from console

```bash
$ pytest run_tests.py
```