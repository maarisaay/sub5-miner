import os

import bittensor as bt
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from datetime import datetime
import logging


from ..utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM


class StructuredSearchEngine:
    def __init__(
        self,
        search_client,
        relevance_ranking_model,
        twitter_crawler=None,
        recall_size=50,
    ):
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

        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        client_ai = OpenAI(api_key=api_key)
        answears = []
        for i, doc in enumerate(body):
            prompt = (
                "You are a crypto researcher, and you will be given speaker transcript as your source of knowledge in ETH Denver 2024. "
                "Your job is to look for a question about the speaker and text 5 answers that can be answered"
                "Transcript:\n\n"
            )
            prompt += str(doc)
            prompt += (
                # "Provide the question in less than 15 words. "
                "Please give the text answears only (no questions), without any additional context or explanation."
                "Answear in JSON format of {'text': [list of 5 answears]}"
            )
            output = client_ai.chat.completions.create(
                model="gpt-4-turbo",
                # response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                timeout=60,
            )
            # logging.info("Response 1: %s", output)
            # print(f"OUTPUT: {output}")
            output_json = output.json()
            output_dict = json.loads(output_json)
            text = output_dict['choices'][0]['message']['content']

            bt.logging.info(f"TEXT: {text}")

            output2 = client_ai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definitions:
                         outdated: Time-sensitive information that is no longer current or relevant.
                         insightless: Superficial content lacking depth and comprehensive insights.
                         somewhat insightful: Offers partial insight but lacks depth and comprehensive coverage.
                         Insightful: Comprehensive, insightful content suitable for informed decision-making.""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "user",
                        "content": "You will be given a list with 5 answears. Use the metric choices [off topic, somewhat relevant, relevant] to evaluate answears. Return answear with metric relevant if it exists, if not choose the best one from answears with somewhat relevant metric. Please give choosen answear only, without any additional context or explanation. The answears are as follows:\n"
                                   + text,
                    },
                ],
                temperature=0,
            )
            output2_json = output2.json()
            output2_dict = json.loads(output2_json)
            answear = {}
            answear['item_id'] = i
            answear['text'] = output2_dict['choices'][0]['message']['content']
            answear['text'] = answear['text'].replace('"', '')
            # answear = search_engine.get_output2(i, text)
            answears.append(answear['text'])

        bt.logging.info(f"ANSWEARS: {answears}")
        bt.logging.info(f"QUERY_INDEX: {query.index_name}")
        bt.logging.info(f"BODY: {body}")
        # ranked_docs = self.structured_search_engine.vector_search(query, body)
        response = self.structured_search_engine.search_client.search(index=query.index_name, body=body)
        bt.logging.info(f"RESPONSE {response}")
        ranked_docs = [doc["_source"] for doc in response["hits"]["hits"]]
        for i, item in enumerate(ranked_docs):
            item['text'] = answears[i]['text']
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
