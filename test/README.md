# Tests for InvertedIndex

Tests use Python3 along with `unittest` library. 

To test system temporary PostgreSQL database created using Docker, filled with text files from [test_files](test). Api also lauched in separate process.

To run tests do the following:

1. Go to test directory

```bash
$ cd test
```

2. Run all the tests from console

```bash
python3 -m unittest run_tests.py
```