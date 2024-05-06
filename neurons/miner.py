# The MIT License (MIT)
# Copyright © 2024 OpenKaito

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import typing
from datetime import datetime
import json
from openai import OpenAI

import bittensor as bt
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import logging

import openkaito
from openkaito.base.miner import BaseMinerNeuron
from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.protocol import (
    SearchSynapse,
    StructuredSearchSynapse,
    SemanticSearchSynapse,
)
from openkaito.search.ranking import HeuristicRankingModel
from openkaito.search.structured_search_engine import StructuredSearchEngine
from openkaito.utils.version import compare_version, get_version
from openkaito.utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self):
        super(Miner, self).__init__()

        load_dotenv()

        search_client = Elasticsearch(
            os.environ["ELASTICSEARCH_HOST"],
            basic_auth=(
                os.environ["ELASTICSEARCH_USERNAME"],
                os.environ["ELASTICSEARCH_PASSWORD"],
            ),
            verify_certs=False,
            ssl_show_warn=False,
        )

        # for ranking recalled results
        ranking_model = HeuristicRankingModel(length_weight=0.8, age_weight=0.2)

        # optional, for crawling data
        twitter_crawler = (
            # MicroworldsTwitterCrawler(os.environ["APIFY_API_KEY"])
            ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])
            if os.environ.get("APIFY_API_KEY")
            else None
        )

        self.structured_search_engine = StructuredSearchEngine(
            search_client=search_client,
            relevance_ranking_model=ranking_model,
            twitter_crawler=twitter_crawler,
            recall_size=self.config.neuron.search_recall_size,
        )

    async def forward_search(self, query: SearchSynapse) -> SearchSynapse:
        """
        Processes the incoming Search synapse by performing a search operation on the crawled data.

        Args:
            query (SearchSynapse): The synapse object containing the query information.

        Returns:
            SearchSynapse: The synapse object with the 'results' field set to list of the 'Document'.
        """
        start_time = datetime.now()
        bt.logging.info(f"received SearchSynapse: ", query)
        self.check_version(query)

        if not self.config.neuron.disable_crawling:
            crawl_size = max(self.config.neuron.crawl_size, query.size)
            self.structured_search_engine.crawl_and_index_data(
                query_string=query.query_string,
                author_usernames=None,
                # crawl and index more data than needed to ensure we have enough to rank
                max_size=crawl_size,
            )

        ranked_docs = self.structured_search_engine.search(query)
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed SearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def forward_structured_search(
        self, query: StructuredSearchSynapse
    ) -> StructuredSearchSynapse:

        start_time = datetime.now()
        bt.logging.info(
            f"received StructuredSearchSynapse... timeout:{query.timeout}s ", query
        )
        self.check_version(query)

        # miners may adjust this timeout config by themselves according to their own crawler speed and latency
        if query.timeout > 12:
            # do crawling and indexing, otherwise search from the existing index directly
            crawl_size = max(self.config.neuron.crawl_size, query.size)
            self.structured_search_engine.crawl_and_index_data(
                query_string=query.query_string,
                author_usernames=query.author_usernames,
                # crawl and index more data than needed to ensure we have enough to rank
                max_size=crawl_size,
            )

        # disable crawling for structured search by default

        ranked_docs = self.structured_search_engine.search(query)
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        filtered_docs = self.filter_docs(ranked_docs)
        # bt.logging.info(f"GPT response: {filtered_docs[1]}")
        # bt.logging.debug(f"{len(filtered_docs[0])} filtered_docs", filtered_docs[0])
        query.results = filtered_docs[0]
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed StructuredSearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def forward_semantic_search(
        self, query: SemanticSearchSynapse
    ) -> SemanticSearchSynapse:

        start_time = datetime.now()
        bt.logging.info(
            f"received SemanticSearchSynapse... timeout:{query.timeout}s ", query
        )
        self.check_version(query)
        # body = self.structured_search_engine.vector_search(query)

        # ranked_docs = search_engine.get_ranked_docs(answears, query.index_name, body)

        ranked_docs = self.structured_search_engine.vector_search(query)
        ranked_docs[0]['text'] += "ABC"
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed SemanticSearchSynapse in {elapsed_time} seconds",
        )
        return query

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        log = (
            "Miner | "
            f"Epoch:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{self.block} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"Rank:{metagraph.R[self.uid]} | "
            f"Trust:{metagraph.T[self.uid]} | "
            f"Consensus:{metagraph.C[self.uid] } | "
            f"Incentive:{metagraph.I[self.uid]} | "
            f"Emission:{metagraph.E[self.uid]}"
        )
        bt.logging.info(log)

    def check_version(self, query):
        """
        Check the version of the incoming request and log a warning if it is newer than the miner's running version.
        """
        if (
            query.version is not None
            and compare_version(query.version, get_version()) > 0
        ):
            bt.logging.warning(
                f"Received request with version {query.version}, is newer than miner running version {get_version()}. You may updating the repo and restart the miner."
            )

    def filter_docs(self, ranked_docs):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        client_ai = OpenAI(api_key=api_key)
        newline = "\n"
        prompt_docs = "\n\n".join(
            [
                f"ItemId: {i}\nTime: {doc['created_at'].split('T')[0]}\nText: {doc['text'][:1000].replace(newline, '  ')}"
                for i, doc in enumerate(ranked_docs)
            ])
        chat_completion = client_ai.chat.completions.create(
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
                    "role": "system",
                    "content": """
        Example 1:
        ItemId: 0
        Time: "2023-11-25" 
        Text: Also driving the charm is Blast's unique design: Depositors start earning yields on the transferred ether alongside BLAST points. "Blast natively participates in ETH staking, and the staking yield is passed back to the L2's users and dapps," the team said in a post Tuesday. 'We've redesigned the L2 from the ground up so that if you have 1 ETH in your wallet on Blast, over time, it grows to 1.04, 1.08, 1.12 ETH automatically."
        As such, Blast is invite-only as of Tuesday, requiring a code from invited users to gain access. Besides, the BLAST points can be redeemed starting in May.Blast raised over $20 million in a round led by Paradigm and Standard Crypto and is headed by pseudonymous figurehead @PacmanBlur, one of the co-founders of NFT marketplace Blur.
        @PacmanBlur said in a separate post that Blast was an extension of the Blur ecosystem, letting Blur users earn yields on idle assets while improving the technical aspects required to offer sophisticated NFT products to users.
        BLUR prices rose 12%% in the past 24 hours following the release of Blast


        Output:
        item_id: 0
        choice: insightful
        reason: It is contains insightful information about the Blast project.

        Example 2:
        ItemId: 1
        Time: "2024-03-19"
        Text: $SLERF to the moon!
        $BOME $SOL $MUMU $BONK $BOPE $WIF $NAP 🥳

        Output:
        item_id: 1
        choice: insightless
        reason: It does not contain much meaningful information, just sentiment about some tickers.
        """,
                },
                {
                    "role": "user",
                    "content": f"You will be given a list of documents with id and you have to rate them based on its information and insightfulness. The documents are as follows:\n"
                               + prompt_docs,
                },
                {
                    "role": "user",
                    "content": f"Use the metric choices [outdated, insightless, somewhat insightful, insightful] to evaluate the text.",
                },
                {
                    "role": "user",
                    "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                               + "{'results': [{'item_id': the item id of choice, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ] } ",
                },
            ],
            model="gpt-3.5-turbo"
        )

        content_string = chat_completion.choices[0].message.content
        data = json.loads(content_string)
        selected_tweets = []
        for item in data['results']:
            if item['choice'] == 'insightful':
                selected_tweets.append(item)
            if len(selected_tweets) == 10:
                break
        if len(selected_tweets) < 10:
            for item in data['results']:
                if item['choice'] == 'somewhat insightful' and item not in selected_tweets:
                    selected_tweets.append(item)
                if len(selected_tweets) == 10:
                    break
        if len(selected_tweets) < 10:
            for item in data['results']:
                if item not in selected_tweets:
                    selected_tweets.append(item)
                if len(selected_tweets) == 10:
                    break

        result = json.dumps({"results": selected_tweets}, indent=4)
        data_result = json.loads(result)
        item_ids = [item["item_id"] for item in data["results"]]
        filtered_docs = []
        for i in range(len(ranked_docs)):
            if i in item_ids:
                filtered_docs.append(ranked_docs[i])
        filtered_and_gpt_response = [filtered_docs, data_result]
        return filtered_and_gpt_response



    def add_tweet_with_new_id(self, tweet, new_id, selected_tweets):
        tweet = tweet.copy()
        tweet['item_id'] = new_id
        selected_tweets.append(tweet)


    # def vector_search(self, query):
    #     load_dotenv()
    #     api_key = os.environ.get("OPENAI_API_KEY")
    #     base_url = "https://chat.openai.com/g/g-xEh6jyzw1-subnet-5"
    #     client_ai = OpenAI(api_key=api_key)
    #     logging.basicConfig(filename='openai.log', level=logging.INFO)
    #     topk = query.size
    #     query_string = query.query_string
    #     index_name = query.index_name if query.index_name else "eth_denver"
    #
    #     embedding = text_embedding(query_string)[0]
    #     embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
    #     body = {
    #         "knn": {
    #             "field": "embedding",
    #             "query_vector": embedding.tolist(),
    #             "k": topk,
    #             "num_candidates": 5 * topk,
    #         },
    #         "_source": {
    #             "excludes": ["embedding"],
    #         },
    #     }
    #
    #     answears = []
    #     for i, doc in enumerate(body["knn"]["query_vector"]):
    #         prompt = (
    #             "You are a crypto researcher, and you will be given speaker transcript as your source of knowledge in ETH Denver 2024. "
    #             "Your job is to look for a question about the speaker and text 5 answers that can be answered"
    #             "Transcript:\n\n"
    #         )
    #         prompt += doc
    #         prompt += (
    #             # "Provide the question in less than 15 words. "
    #             "Please give the question text only, without any additional context or explanation."
    #             "Answear in JSON format of {'text': [list of 5 answears]}"
    #         )
    #         output = client_ai.chat.completions.create(
    #             model="gpt-4-turbo",
    #             # response_format={"type": "json_object"},
    #             messages=[
    #                 {
    #                     "role": "user",
    #                     "content": prompt,
    #                 }
    #             ],
    #             temperature=0,
    #             timeout=60,
    #         )
    #         logging.info("Response 1: %s", output)
    #         print(f"OUTPUT: {output}")
    #         output_json = output.json()
    #         output_dict = json.loads(output_json)
    #         text = output_dict['choices'][0]['message']['content']
    #
    #         load_dotenv()
    #         api_key = os.environ.get("OPENAI_API_KEY")
    #         base_url = "https://chat.openai.com/g/g-xEh6jyzw1-subnet-5"
    #         client_ai = OpenAI(api_key=api_key)
    #
    #         output2 = client_ai.chat.completions.create(
    #             model="gpt-4-turbo",
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": """Below are the metrics and definitions:
    #                                                     outdated: Time-sensitive information that is no longer current or relevant.
    #                                                     insightless: Superficial content lacking depth and comprehensive insights.
    #                                                     somewhat insightful: Offers partial insight but lacks depth and comprehensive coverage.
    #                                                     Insightful: Comprehensive, insightful content suitable for informed decision-making.""",
    #                 },
    #                 {
    #                     "role": "system",
    #                     "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": "You will be given a list with 5 answears. Use the metric choices [off topic, somewhat relevant, relevant] to evaluate answears. Return answear with metric relevant if it exists, if not choose the best one from answears with somewhat relevant metric. Please give choosen answear only, without any additional context or explanation. The answears are as follows:\n"
    #                                + text,
    #                 },
    #             ],
    #             temperature=0,
    #         )
    #         return output2
    #
    #         logging.info("Response 2: %s", output2)
    #         print(f"OUTPUT2: {output2}")
    #         output2_json = output2.json()
    #         output2_dict = json.loads(output2_json)
    #         answear = {}
    #         answear['item_id'] = i
    #         answear['text'] = output2_dict['choices'][0]['message']['content']
    #         answear['text'] = answear['text'].replace('"', '')
    #         answears.append(answear)
    #
    #     response = self.search_client.search(index=index_name, body=body)
    #     ranked_docs = [doc["_source"] for doc in response["hits"]["hits"]]
    #     for i, item in enumerate(ranked_docs):
    #         item['text'] = answears[i]['text']
    #     # optional: you may implement yourselves additional post-processing filtering/ranking here
    #     return ranked_docs

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            miner.print_info()
            time.sleep(30)
