import json
from pprint import pprint
import os
import time
import logging


from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("elasticsearch")
logger.setLevel(logging.DEBUG)

load_dotenv()

class Search:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es = Elasticsearch(os.environ['ELASTICSEARCH_CONNSTRING'],
        http_auth=(os.environ['ELASTICSEARCH_USERNAME'],os.environ['ELASTICSEARCH_PASSWORD']))
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)
    
    def search(self, **query_args):
        return self.es.search(index='my_documents',**query_args)

    def retrieve_document(self,id):
        return self.es.get(index='my_documents',id=id)

    def create_index(self):
        self.es.indices.delete(index='my_documents',ignore_unavailable=True)
        self.es.indices.create(
            index='my_documents',
            mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector'
                },
                'elser_embedding':{
                    'type': 'sparse_vector'
                }
            }
        },
        settings={
            'index': {
                'default_pipeline': 'elser-ingest-pipeline'
            }
        }
        )
    def deploy_elser(self):
        self.es.ml.put_trained_model(model_id='.elser_model_2',
                                    input={'field_names':['text_field']})
        while True:
            status = self.es.ml.get_trained_models(model_id='.elser_model_2',
                                       include='definition_status')
            if status['trained_model_configs'][0]['fully_defined']:
                # model is ready
                break
            time.sleep(1)

        self.es.ml.start_trained_model_deployment(model_id='.elser_model_2')

        self.es.ingest.put_pipeline(
            id='elser-ingest-pipeline',
            processors=[
                {
                    'inference': {
                        'model_id': '.elser_model_2',
                        'input_output': [
                            {
                                'input_field': 'summary',
                                'output_field': 'elser_embedding',
                            }
                        ]
                    }
                }
            ]
        )
        
    def get_embedding(self,text):
        return self.model.encode(text)

    def insert_document(self, document):
        return self.es.index(index='my_documents',body={
            **document,
            'embedding': self.get_embedding(document['summary'])
        })
    
    def insert_documents(self, documents):
        operations = []
        for document in documents:
            operations.append({'index': {'_index': 'my_documents'}})
            operations.append({
                **document,
                'embedding': self.get_embedding(document['summary'])
            })
        return self.es.bulk(operations=operations)

    def reindex(self):
        self.create_index()
        with open('data.json','rt') as f:
            documents = json.loads(f.read())
        return self.insert_documents(documents)

    