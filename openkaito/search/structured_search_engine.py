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
                "You are a crypto researcher, and you will be given a speaker's transcript from ETH Denver 2024 as your primary source of knowledge. "
                "Here is a question related to the speaker's topic:"
                + query_string +
                "\n\nPlease read the following transcript segment carefully:\n\n"
                + doc['text'] +
                "\nBased on the transcript above, answer the question by citing specific parts of the transcript that are relevant to the question. "
                "Your answers should explicitly connect the content of the transcript with the question, using quotes from both the transcript and the question itself to substantiate your response. "
                "Ensure each answer is comprehensive, accurate, and directly addresses the question based on the transcript content provided. "
                # """Format your responses as JSON format of {'text': ["answear 1", "answear2", ... ]} """
            )
            prompts.append(prompt)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            text_lists = list(executor.map(self.send_first_query_tuning, prompts))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.send_second_query_tuning, text_lists))

        chosen_text = []
        preferences = ["relevant", "somewhat relevant", "off topic"]

        for content in results:
            print(content)
            try:
                content_json = json.loads(content[0])
            except:
                content_json = json.loads(content)
            found = False
            for preference in preferences:
                try:
                    if content_json[0]['metric'] == preference:
                        chosen_text.append(content_json[0]['text'])
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
        thread = client_ai.beta.threads.create()
        output = client_ai.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id='asst_LLUKdYUpGZDd25niw6HNq9aX',
            instructions=prompt,
        )
        messages = client_ai.beta.threads.messages.list(
            thread_id=thread.id
        )
        messages_dicts = [self.message_to_dict(msg) for msg in messages]
        text_list = []
        for mes in messages_dicts:
            text_list.append(mes['content'][0])
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
                "You will be given a list with 5 answers. Use the metric choices [off topic, somewhat relevant, relevant] to evaluate answers. Return answer with metric. The answers are as follows:\n" +
                str(text_list) +
                """Format your responses as JSON format of [{'text': text, 'metric': metric}, {...}, ... ]"""
        )

        thread = client_ai.beta.threads.create()
        output = client_ai.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id='asst_LLUKdYUpGZDd25niw6HNq9aX',
            instructions=prompt,
        )
        messages = client_ai.beta.threads.messages.list(
            thread_id=thread.id
        )
        messages_dicts = [self.message_to_dict(msg) for msg in messages]
        ranked_docs = []
        for mes in messages_dicts:
            ranked_docs.append(mes['content'][0])
        return ranked_docs

    def send_first_query_tuning(self, prompt):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        client_ai = OpenAI(api_key=api_key)
        output = client_ai.completions.create(
            model="ft:davinci-002:dawomeq01::9PTscfX4",
            prompt=[{
                "role": "user",
                "content": prompt,
            }]
        )
        return output.choices[0].text

    def send_second_query_tuning(self, text_list):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        client_ai = OpenAI(api_key=api_key)
        prompt = (
                "Below are the metrics and definitions for evaluating the answers:\n"
                "- Off topic: The content does not answer the given question or is unrelated.\n"
                "- Somewhat relevant: The content offers partial insight but does not fully address the question.\n"
                "- Relevant: The content directly answers the question with comprehensive and insightful information.\n\n"
                "Please evaluate the following answers based on the criteria provided above. Use the metric choices "
                "[off topic, somewhat relevant, relevant] to classify each answer. Your response should be formatted as a "
                "list of dictionaries, each containing the 'text' of the answer and your 'metric' classification. "
                "Ensure your responses directly correspond to the provided answers and include specific reasons for your classification.\n\n"
                "Answers for Evaluation:\n"
                + ''.join(f"- {answer}\n" for answer in text_list) +
                "\nExample of expected format response:\n"
                "[{'text': 'Example text of the answer', 'metric': 'relevant'}, {'text': 'Another example text', 'metric': 'off topic'}]\n\n"
                "Format your responses correctly and ensure each metric is clearly justified based on the content of the answer."
        )

        output = client_ai.completions.create(
            model="ft:davinci-002:dawomeq01::9PTscfX4",
            prompt=[{
                "role": "user",
                "content": prompt,
            }],
            max_tokens=150,
            stop=None,
            temperature=0.3
        )
        return output.choices[0].text

    def message_to_dict(self, message):
        return {
            "id": message.id,
            "assistant_id": message.assistant_id,
            "content": [content_block.text.value for content_block in message.content],
            "created_at": message.created_at,
            "role": message.role,
            "thread_id": message.thread_id
        }