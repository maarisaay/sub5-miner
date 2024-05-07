import os

import bittensor as bt
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from datetime import datetime
import logging
import re


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
        recalled_items[0]['text'] += 'ABC'
        results = ranking_model.rank(search_query.query_string, recalled_items)
        # results[0]['text'] += 'ABC'
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
            prompt += doc['text']
            prompt += (
                "Provide the question in less than 30 words. "
                "Please give the answear text only (no questions), without any additional context or explanation. Your answear must be Insightful: Comprehensive, insightful content suitable for informed decision-making. Don't write anything off topic."
                """Answear in JSON format of {'text': ["answear 1", "answear2", ... ]}"""
                # "Answear in JSOM format"
            )
            output = client_ai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "system",
                        "content": "Please provide the response in JSON format.",
                    }
                ],
                response_format={"type": "json_object"},
                temperature=1.5,
                timeout=60,
            )
            output_string = output.choices[0].message.content
            start_index = output_string.find('"text": [') + len('"text": ')
            end_index = output_string.find(']', start_index)
            list_string = output_string[start_index - 1:end_index + 1]

            try:
                text_list = json.loads(list_string)
            except:
                json_str = re.sub(r'\s+', ' ', list_string).strip()
                json_str = re.sub(r'(?<=")(?!\s*,)(?!\s*\])', ',', json_str)
                json_str = re.sub(r'(?<=")(?!\s*,)(?!\s*\])', ',', json_str)
                json_str = re.sub(r'(?<=")(?!\s*,)(?!\s*\])', ',', json_str)
                json_str = re.sub(r'(?<=\")([^\"]*?)\s*$', r'\1"', json_str)

                text_list = json.loads(json_str)

            output2 = client_ai.chat.completions.create(
                model="gpt-4-turbo",
                # response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definitions:
                                    off topic: Superficial or unrelevant content that can not answer the given question.
                                    somewhat relevant: Offers partial insight to partially answer the given question.
                                    relevant: Comprehensive, insightful content suitable for answering the given question.""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list with 5 answears. Use the metric choices [off topic, somewhat relevant, relevant] to evaluate answears. Return answear with metric. The answears are as follows:\n"
                                   + str(text_list),
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                                   + "{'results': [{'item_id': the item id of choice, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ]}",
                    },
                ],
                temperature=0,
            )

            output2_string = output2.choices[0].message.content
            try:
                output2_json = json.loads(output2_string)
            except:
                start_index = output2_string.find('{"results":') + len('{"results": ')
                end_index = output2_string.find('} ] }', start_index)
                list_string = "{" + output2_string[start_index:end_index - 5] + "}"
                output2_json = json.loads(list_string)
            print(output2_json)
            chosen_text = ""
            for i, content in enumerate(output2_json['results']):
                if content['choice'] == "relevant":
                    chosen_text = text_list[i]
                    break
            if chosen_text == "":
                for i, content in enumerate(output2_json['results']):
                    if content['choice'] == "somewhat relevant":
                        chosen_text = text_list[i]
                        break
            if chosen_text == "":
                for i, content in enumerate(output2_json['results']):
                    if content['choice'] == "off topic":
                        chosen_text = text_list[i]
                        break

            answears.append(chosen_text)

        bt.logging.info(f"ANSWEARS: {answears}")
        bt.logging.info(f"QUERY_INDEX: {query.index_name}")
        bt.logging.info(f"BODY: {body}")
        # ranked_docs = self.structured_search_engine.vector_search(query, body)
        response = self.structured_search_engine.search_client.search(index=query.index_name, body=body)
        bt.logging.info(f"RESPONSE {response}")
        ranked_docs = [doc["_source"] for doc in response["hits"]["hits"]]
        for i in enumerate(ranked_docs):
            ranked_docs[i]['text'] = answears[i]
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
