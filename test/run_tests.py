import subprocess
import psycopg2

#from ..inverted_index import InvertedIndex, PostgresConfig
import requests
import logging
logging.getLogger("requests").setLevel(logging.CRITICAL)

import os
import signal
from os import listdir
from os.path import isfile, join, basename
from time import sleep

import unittest
import ast

from dotenv import dotenv_values

CONFIG = {
    **dotenv_values('.env'),
    **dotenv_values('.env.local'),
}

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
        cur.execute("CREATE TABLE pages (url varchar NOT NULL UNIQUE, parsed_data varchar NOT NULL);")
    except:
        print("I can't drop our test database!")

    
    # Filling db with test files
    test_file_dir = 'test_files' # to run from root
    test_files = [join(test_file_dir, f) for f in listdir(test_file_dir) if isfile(join(test_file_dir, f))]

    for filename in test_files:
        with open(filename,'r',encoding = 'utf-8') as f:
            cur.execute("INSERT INTO pages VALUES(%s,%s);",(filename,f.read()))


    conn.commit() # <--- makes sure the change is shown in the database
    conn.close()
    cur.close()

def start_api():
    #Starting api
    subprocess.run('python3 ../tools/create_index.py --config test_config.json'.split())
    api_pid = subprocess.Popen('python3 ../api.py --config test_config.json'.split(), preexec_fn=os.setsid)
    sleep(5)
    return(os.getpgid(api_pid.pid))
    
def stop_api(api_pid):
    os.killpg(api_pid, signal.SIGTERM)

def stop_psql_container():
    '''
    Stops and deletes docker compose container. 
    '''
    subprocess.run(['sudo','docker-compose','rm','-fsv'])

def send_api_request(word):
    r = requests.get(f'http://localhost:5005/search?q={word}&top=10')
    #print(f'Query: {word}\nStatus: {r.status_code}\nElapsed: {r.elapsed.total_seconds()}\nResult: {r.text}')
    return r




class RankingTests(unittest.TestCase):
    api_pid = 0
    @classmethod
    def setUpClass(cls):
        initialize_psql_container()
        cls.api_pid = start_api()
        print(f'Api is running under PID - {cls.api_pid}')
        
    @classmethod
    def tearDownClass(cls):
        stop_api(cls.api_pid)
        stop_psql_container()

    def test_word_search(self):
        test_word = 'шаман'
        document = "test_files/shaman_king.txt"
        wrong_doc = "test_files/borsch.txt"

        req = send_api_request(test_word)
        #Testing finding right document
        self.assertIn(document,req.text)
        # Testing not finding wrong document
        self.assertNotIn(wrong_doc,req.text)

    def test_order(self):
        test_word = 'земля'
        ordered_docs = ["test_files/additional_text.txt","test_files/walking_to_river.txt","test_files/heart_stopped.txt"]

        req = send_api_request(test_word)

        d2 = ast.literal_eval(req.text)

        self.assertEqual(d2['data'],ordered_docs)

    def test_search_time(self):
        test_words = ['шаман','борщ','свет',"земля","планета","гармония"]
        
        # List of request times
        search_times =[]
        for word in test_words:
            search_times.append(send_api_request(word).elapsed.total_seconds())

        #Testing finding right document
        mean_time = sum(search_times) / len(search_times)
        print(f'Mean search time: {mean_time}')
        self.assertLess(mean_time,3)




if __name__ == "__main__":

    unittest.main()
    