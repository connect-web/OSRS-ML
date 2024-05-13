import os

import pandas as pd
import psycopg2
from psycopg2.extras import Json
import time
from time import sleep
import random
random.seed(time.time())

class Connection:
    def local(self):
        self._DBNAME = os.environ.get('runescape_database_name')

        self._USERNAME = os.environ.get('rsLocalUsername', 'postgres')
        self._PASSWORD = os.environ.get('rsPassword')

        self._HOST = os.environ.get('rsLocalHost')
        self._PORT = os.environ.get('rsLocalPort')

    def online(self):
        self._DBNAME = os.environ.get('runescape_database_name')

        self._USERNAME = os.environ.get('rsLocalUsername', 'postgres')
        self._PASSWORD = os.environ.get('rsPassword')

        self._HOST = os.environ.get('rsHost')
        self._PORT = os.environ.get('rsPort')

    def __init__(self, localhost):
        self.local() if localhost else self.online()

        self.conn = psycopg2.connect(
            dbname = self._DBNAME ,
            user = self._USERNAME ,
            password = self._PASSWORD ,
            host = self._HOST ,
            port = self._PORT
            )


    def get (self , query , params = None):
        with self.conn.cursor() as cursor:
            cursor.execute(query , params)
            return cursor.fetchall()

    def post (self , query , params = None):
        with self.conn.cursor() as cursor:
            cursor.execute(query , params)
            return cursor.fetchall()

    def postMany (self , query , params = None,commit=True):
        with self.conn.cursor() as cursor:
            cursor.executemany(query , params)
            if commit:
                self.conn.commit()
            return

    def get_df(self, query, params = None, columns = None):
        rows = self.get(query, params)
        if columns:
            return pd.DataFrame(rows , columns = columns)

        return pd.DataFrame(rows)

    def close(self):
        self.conn.close()
