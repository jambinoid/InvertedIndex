import subprocess
import psycopg2

from inverted_index import InvertedIndex, PostgresConfig

from os import listdir
from os.path import isfile, join
from time import sleep

import pytest
import time


from dotenv import dotenv_values

CONFIG = {
    **dotenv_values('.env'),
    **dotenv_values('.env.local'),
}

_ENCODINGS = [None,'gamma','delta']
_INDEX = None

#----------------- CONTAINER METHODS-----------------

def initialize_psql_container():
    '''
    Initializes PostgreSQL docker container and fills it with files from test_files directory.
    '''
    # Starting psql container
    subprocess.run(['sudo','docker-compose','up','-d','--force-recreate'])

    sleep(10)
    
    # Connecting to container
    try:
        conn = psycopg2.connect(
            database = CONFIG['POSTGRES_DB'], 
            user = CONFIG['POSTGRES_USER'], 
            password = CONFIG['POSTGRES_PASSWORD'], 
            host = "localhost", 
            port = "5600")
    except:
        print("I am unable to connect to the database") 
    

    # Creating Table
    cur = conn.cursor()
    
    try:
        cur.execute("CREATE TABLE pages (id serial PRIMARY KEY,url varchar NOT NULL UNIQUE, parsed_data varchar NOT NULL);")
    except:
        print("I can't drop our test database!")

    
    # Filling db with test files
    test_file_dir = 'test_files' # to run from root
    test_files = [join(test_file_dir, f) for f in listdir(test_file_dir) if isfile(join(test_file_dir, f))]


    for filename in test_files:
        with open(filename,'r',encoding = 'utf-8') as f:
            cur.execute("INSERT INTO pages (url,parsed_data) VALUES(%s,%s);",(filename,f.read()))


    conn.commit() # <--- makes sure the change is shown in the database
    conn.close()
    cur.close()

def stop_psql_container():
    '''
    Stops and deletes docker compose container. 
    '''
     #print('stopping container')
    subprocess.run(['sudo','docker-compose','rm','-fsv'])

#----------------- SEARCH METHODS-----------------

def create_inv_idx(encoding = None):
    '''
    Creates instanse of Inverted Index. Won't work without calculating index.
    '''
    search_config = PostgresConfig(
        database = CONFIG['POSTGRES_DB'], 
        user = CONFIG['POSTGRES_USER'], 
        password = CONFIG['POSTGRES_PASSWORD'], 
        host = "localhost", 
        port = "5600")

    pytest._IINDEX = InvertedIndex(
        user=search_config.user,
        password=search_config.password,
        host=search_config.host,
        port=search_config.port,
        database=search_config.database,
        iindex_table="inverted_index_gamma",
        iindex_term_col=search_config.iindex_term_col,
        iindex_docid_col=search_config.iindex_docid_col,
        iindex_count_col=search_config.iindex_count_col,
        dlens_table="page_lengths_gamma",
        dlens_docid_col=search_config.dlens_docid_col,
        dlens_len_col=search_config.dlens_len_col,
        src_table=search_config.src_table,
        src_docid_col=search_config.src_docid_col,
        src_doc_col=search_config.src_doc_col,
        encoding=encoding,
        clean=True
    )

def calculate_index():
    pytest._IINDEX.create()

def clean_indexes():
    pytest._IINDEX.remove(with_additional_tables=True)

def prepare_search(encoding = None):
    '''
    Prepares database and test system for search.
    If index already exists recreates it if encoding is different.
    '''
    if pytest._IINDEX is not None:
        if pytest._IINDEX.encoding != encoding:
            clean_indexes()
            del pytest._IINDEX
        else:
            return

    create_inv_idx(encoding)
    calculate_index()

def search(term):
    return pytest._IINDEX.search(
        query=term,
        col_to_return='url'
    )

#----------------- TEST PREPARATIONS -----------------

@pytest.fixture(scope="session", autouse=True)
def preparation(request):
    # start psql container using docker-compose
    initialize_psql_container()
    # create global variable to remember search reference and create inverted index
    pytest._IINDEX = None
    prepare_search()
    # after all tests, close anb delete container
    request.addfinalizer(stop_psql_container)

#----------------- TESTS -----------------

def test_finding_right():
        
    test_word = 'шаман'
    document = "test_files/shaman_king.txt"

    req = search(test_word)
    #Testing finding right document
    assert document in req

def test_finding_wrong():
        
    test_word = 'шаман'
    wrong_doc = "test_files/borsch.txt"

    req = search(test_word)
    #Testing finding right document
    assert wrong_doc not in req

def test_order():

    test_word = 'земля'
    ordered_docs = ["test_files/earth.txt","test_files/walking_to_river.txt","test_files/mars.txt","test_files/heart_stopped.txt"]

    search_res = search(test_word)
    
    print(search_res)
    assert search_res == ordered_docs

def test_encodings_same_ranking():
    rankings= []
    test_word = 'земля'

    # getting default encoding ranks and killing api
    for code in _ENCODINGS:
        prepare_search(code)
        rankings.append(search(test_word))
    
    assert all(ranks == rankings[0] for ranks in rankings)

def test_search_time_default():
    prepare_search()

    test_words = ['шаман','борщ','свет',"земля","планета","гармония"]
    times = []
    for word in test_words:
        t1 = time.time()
        search(word)
        t2 = time.time()
        times.append(t2-t1)

    mean_time = sum(times)/len(times)
    #print(mean_time)
    
    assert mean_time< 3

def test_search_time_default():
    prepare_search()

    test_words = ['шаман','борщ','свет',"земля","планета","гармония"]
    times = []
    for word in test_words:
        t1 = time.time()
        search(word)
        t2 = time.time()
        times.append(t2-t1)

    mean_time = sum(times)/len(times)
    #print(mean_time)
    
    assert mean_time< 3

def test_search_time_gamma():
    prepare_search(_ENCODINGS[1])

    test_words = ['шаман','борщ','свет',"земля","планета","гармония"]
    times = []
    for word in test_words:
        t1 = time.time()
        search(word)
        t2 = time.time()
        times.append(t2-t1)

    mean_time = sum(times)/len(times)
    #print(mean_time)
    
    assert mean_time< 3

def test_search_time_delta():
    prepare_search(_ENCODINGS[2])

    test_words = ['шаман','борщ','свет',"земля","планета","гармония"]
    times = []
    for word in test_words:
        t1 = time.time()
        search(word)
        t2 = time.time()
        times.append(t2-t1)

    mean_time = sum(times)/len(times)
    #print(mean_time)
    
    assert mean_time< 3