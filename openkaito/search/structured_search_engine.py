import os

import bittensor as bt
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from datetime import datetime
import logging
import re
import concurrent.futures
import requests


from ..utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM


class StructuredSearchEngine:
    def __init__(
        self,
        search_client,
        relevance_ranking_model,
        twitter_crawler=None,
        recall_size=50,
    ):
        self.structured_search_engine = None
        load_dotenv()

        self.search_client = search_client
        self.init_indices()

        # for relevance ranking recalled results
        self.relevance_ranking_model = relevance_ranking_model

        self.recall_size = recall_size

        # optional, for crawling data
        self.twitter_crawler = twitter_crawler
        # self.get_output1()
        # self.get_output2()
        # self.get_client_ai()
        # self.get_ranked_docs()

    def twitter_doc_mapper(cls, doc):
        return {
            "id": doc["id"],
            "text": doc["text"],
            "created_at": doc["created_at"],
            "username": doc["username"],
            "url": doc["url"],
            "quote_count": doc["quote_count"],
            "reply_count": doc["reply_count"],
            "retweet_count": doc["retweet_count"],
            "favorite_count": doc["favorite_count"],
        }

    def init_indices(self):
        """
        Initializes the indices in the elasticsearch database.
        """
        index_name = "twitter"
        if not self.search_client.indices.exists(index=index_name):
            bt.logging.info("creating index...", index_name)
            self.search_client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "id": {"type": "long"},
                            "text": {"type": "text"},
                            "created_at": {"type": "date"},
                            "username": {"type": "keyword"},
                            "url": {"type": "text"},
                            "quote_count": {"type": "long"},
                            "reply_count": {"type": "long"},
                            "retweet_count": {"type": "long"},
                            "favorite_count": {"type": "long"},
                        }
                    }
                },
            )

    def search(self, search_query):
        """
        Structured search interface for this search engine

        Args:
        - search_query: A `StructuredSearchSynapse` or `SearchSynapse` object representing the search request sent by the validator.
        """

        result_size = search_query.size

        recalled_items = self.recall(
            search_query=search_query, recall_size=self.recall_size
        )

        ranking_model = self.relevance_ranking_model

        results = ranking_model.rank(search_query.query_string, recalled_items)

        return results[:result_size]

    def recall(self, search_query, recall_size):
        """
        Structured recall interface for this search engine
        """
        query_string = search_query.query_string

        es_query = {
            "query": {
                "bool": {
                    "must": [],
                }
            },
            "size": recall_size,
        }

        if search_query.query_string:
            es_query["query"]["bool"]["must"].append(
                {
                    "query_string": {
                        "query": query_string,
                        "default_field": "text",
                        "default_operator": "AND",
                    }
                }
            )

        if search_query.name == "StructuredSearchSynapse":
            if search_query.author_usernames:
                es_query["query"]["bool"]["must"].append(
                    {
                        "terms": {
                            "username": search_query.author_usernames,
                        }
                    }
                )

            time_filter = {}
            if search_query.earlier_than_timestamp:
                time_filter["lte"] = search_query.earlier_than_timestamp
            if search_query.later_than_timestamp:
                time_filter["gte"] = search_query.later_than_timestamp
            if time_filter:
                es_query["query"]["bool"]["must"].append(
                    {"range": {"created_at": time_filter}}
                )

        bt.logging.trace(f"es_query: {es_query}")

        try:
            response = self.search_client.search(
                index="twitter",
                body=es_query,
            )
            documents = response["hits"]["hits"]
            results = []
            for document in documents if documents else []:
                doc = document["_source"]
                results.append(self.twitter_doc_mapper(doc))
            bt.logging.info(f"retrieved {len(results)} results")
            bt.logging.trace(f"results: ")
            return results
        except Exception as e:
            bt.logging.error("recall error...", e)
            return []



    def vector_search(self, query):
        # client_ai = self.get_client_ai()
        topk = query.size
        query_string = query.query_string
        index_name = query.index_name if query.index_name else "eth_denver"

        embedding = text_embedding(query_string)[0]
        embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": embedding.tolist(),
                "k": topk,
                "num_candidates": 5 * topk,
            },
            "_source": {
                "excludes": ["embedding"],
            },
        }

        prompts = []
        for doc in body:
            prompt = (
                    "You are a crypto researcher, and you will be given speaker transcript as your source of knowledge in ETH Denver 2024. Your primary source of information is a transcript of a speaker's presentation. Your task is to return 5 responds to a specific question related to the speaker's topic."
                    "Please provide your answers based solely on the content provided in the transcript. The question is as follows:"
                    + query_string +
                    "Please review the transcript carefully:\n\n"
                    + doc['text'] +
                    "Provide your answers in a concise and insightful manner, focusing directly on the information relevant to the question. Each answer should be comprehensive and suitable for informed decision-making, incorporating key words from the question to ensure relevance."
                    "Provide the question in less than 30 words. "
                    """Format your responses as JSON format of {'text': ["answear 1", "answear2", ... ]} """
                    "Remember to: Ensure that each answer is directly related to the question, Provide detailed and insightful information that reflects a deep understanding of the transcript content, Avoid including any off-topic information or unnecessary context in your answers."
            )
            prompts.append(prompt)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            text_lists = list(executor.map(self.send_first_query, prompts))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.send_second_query, text_lists))

        chosen_text = []
        preferences = ["relevant", "somewhat relevant", "off topic"]

        for content in results:
            content_json = json.loads(content)
            found = False
            for preference in preferences:
                for metric in content_json['evaluation']:
                    try:
                        if metric['metric'] == preference:
                            chosen_text.append(metric['text'])
                            found = True
                            break
                    except:
                        continue
                if found:
                    break


        # bt.logging.info(f"ANSWEARS: {answears}")
        # bt.logging.info(f"QUERY_INDEX: {query.index_name}")
        # bt.logging.info(f"BODY: {body}")
        # ranked_docs = self.structured_search_engine.vector_search(query, body)
        response = self.structured_search_engine.search_client.search(index=query.index_name, body=body)
        bt.logging.info(f"RESPONSE {response}")
        ranked_docs = [doc["_source"] for doc in response["hits"]["hits"]]
        for i in enumerate(ranked_docs):
            ranked_docs[i]['text'] = chosen_text[i]
        return ranked_docs



    def crawl_and_index_data(self, query_string, author_usernames, max_size):
        """
        Crawls the data from the twitter crawler and indexes it in the elasticsearch database.
        """
        if self.twitter_crawler is None:
            bt.logging.warning(
                "Twitter crawler is not initialized. skipped crawling and indexing"
            )
        try:
            processed_docs = self.twitter_crawler.search(
                query_string, author_usernames, max_size
            )
            bt.logging.debug(f"crawled {len(processed_docs)} docs")
            bt.logging.trace(processed_docs)
        except Exception as e:
            bt.logging.error("crawling error...", e)
            processed_docs = []

        if len(processed_docs) > 0:
            try:
                bt.logging.info(f"bulk indexing {len(processed_docs)} docs")
                bulk_body = []
                for doc in processed_docs:
                    bulk_body.append(
                        {
                            "update": {
                                "_index": "twitter",
                                "_id": doc["id"],
                            }
                        }
                    )
                    bulk_body.append(
                        {
                            "doc": doc,
                            "doc_as_upsert": True,
                        }
                    )

                r = self.search_client.bulk(
                    body=bulk_body,
                    refresh=True,
                )
                bt.logging.trace("bulk update response...", r)
                if not r.get("errors"):
                    bt.logging.info("bulk update succeeded")
                else:
                    bt.logging.error("bulk update failed: ", r)
            except Exception as e:
                bt.logging.error("bulk update error...", e)

    def send_first_query(self, prompt):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        client_ai = OpenAI(api_key=api_key)
        output = client_ai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=60
        )

        text_list = output.choices[0].message.content
        return text_list

    def send_second_query(self, text_list):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        client_ai = OpenAI(api_key=api_key)
        prompt = (
                "Below are the metrics and definitions:\n"
                "off topic: Superficial or unrelevant content that can not answer the given question.\n"
                "somewhat relevant: Offers partial insight to partially answer the given question.\n"
                "relevant: Comprehensive, insightful content suitable for answering the given question.\n"
                "\nCurrent Time: {}\n".format(datetime.now().isoformat().split('T')[0]) +
                "You will be given a list with 5 answers. Use the metric choices [off topic, somewhat relevant, relevant] to evaluate answers. Return answer with metric. The answers are as follows:\n" +
                str(text_list)
        )

        output = client_ai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            timeout=60
        )

        ranked_docs = output.choices[0].message.content
        return ranked_docs