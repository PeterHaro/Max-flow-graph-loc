import json
import os
import re
import sys
from collections import defaultdict
from enum import Enum
from functools import partial
from operator import itemgetter
from pprint import pprint

import networkx as nx
import requests


def calculate_averate_likert_values_from_transaction(transaction):
    likert_fields = ["qualityLikert", "deliveryTimeLikert", "packagingLikert", "responseRateLikert",
                     "overallSatesfactionLikert"]
    length = len(likert_fields)
    avg = sum(transaction[likert] for likert in likert_fields) / length
    return avg


def create_graph_representation(graph_data):
    """ Generate graph with suppliers/buyers as nodes\
    transactions as edges, #transactions as degree"""
    graph = nx.Graph()

    # Add suppliers to graph
    for supplier in graph_data['suppliers']:
        graph.add_node(int(supplier['id']) * -1)

    for buyer in graph_data['buyers']:
        current_buyer_id = buyer['id']
        graph.add_node(current_buyer_id)
        try:
            for transaction in buyer['historicalData']:
                current_supplier_id = int(transaction['supplierId']) * -1
                if current_supplier_id not in graph[current_buyer_id]:
                    graph.add_edge(current_buyer_id, current_supplier_id, degree=0)
                    graph[current_buyer_id][current_supplier_id]['weight'] = 0
                graph[current_buyer_id][current_supplier_id]['degree'] += 1
                graph[current_buyer_id][current_supplier_id][
                    'weight'] += calculate_averate_likert_values_from_transaction(transaction)
        except TypeError:
            print("Invalid type in graph generation. Hopefully an empty transaction list.")
    return graph


class MatchmakingMethodology(Enum):
    CONNECTED_COMPONENT = 1,
    MAX_FLOW = 2


class Matchmaking(object):
    WEIGHTS = {'flow': 0.1,
               'price': -0.1,
               'distance': 0.1,
               }
    is_prod = True

    def __init__(self):
        self.endpoint = os.getenv("SEMANTIC_MATCHMAKING_ENDPOINT", "http://localhost:1337")
        graph_data = {}
        if not self.is_prod:
            for name in ['buyers', 'suppliers', 'transactions']:
                with open(f'static/{name}.json') as fp:
                    graph_data[name] = json.load(fp)
        else:
            graph_data["buyers"] = requests.get(self.endpoint + "/matchmaking/getAllBuyers").json()
            graph_data["suppliers"] = requests.get(self.endpoint + "/matchmaking/getAllSuppliers").json()
            graph_data["transactions"] = requests.get(
                self.endpoint + "/matchmaking/getAllTransactionalData").json()
        self.graph = create_graph_representation(graph_data)

    def reload_graph(self):
        graph_data = {}
        if not self.is_prod:
            for name in ['buyers', 'suppliers', 'transactions']:
                with open(f'static/{name}.json') as fp:
                    graph_data[name] = json.load(fp)
        else:
            graph_data["buyers"] = requests.get(self.endpoint + "/matchmaking/getAllBuyers").json()
            graph_data["suppliers"] = requests.get(self.endpoint + "/matchmaking/getAllSuppliers").json()
            graph_data["transactions"] = requests.get(
                self.endpoint + "/matchmaking/getAllTransactionalData").json()
        self.graph = create_graph_representation(graph_data)

    def perform_matchmaking(self, offers, buyer, method):
        res = None
        if method == MatchmakingMethodology.CONNECTED_COMPONENT:
            res = self.connected_component(offers, buyer)
        elif method == MatchmakingMethodology.MAX_FLOW:
            res = self.max_flow(buyer, offers)
        else:
            pprint(f"{method} not implemented")
        return sorted(res.items(), key=itemgetter(1), reverse=True)

    def connected_component(self, offers, buyer):
        """ naive: match is -1 if in different graph components """
        adj_offers = dict()
        for offer in offers:
            if offer['supplierId'] in nx.algorithms.descendants(self.graph, buyer):
                adj_offers[offer['id']] = offer['semanticSimilarity']
            else:
                adj_offers[offer['id']] = offer['semanticSimilarity'] - 1
        return adj_offers

    def max_flow(self, buyer, offers):
        mf = partial(nx.maximum_flow_value, self.graph, buyer, capacity='weight')
        adj_offers = dict()
        _temp_rel = defaultdict(dict)  # save price + dist, to extract min/max, normalisation
        for offer in offers:
            o_id = offer['id']
            adj_offers[o_id] = offer['semanticSimilarity']
            if offer['supplierId'] in nx.algorithms.descendants(self.graph, buyer):
                _temp_rel['flow'][o_id] = mf(offer['supplierId'])
            else:
                _temp_rel['flow'][o_id] = 0
            _temp_rel['distance'][o_id] = offer['distanceInKm']
            _temp_rel['price'][o_id] = offer['price']

        for measure, values in _temp_rel.items():
            _min = min(values.items(), key=itemgetter(1))[1]
            _max = max(values.items(), key=itemgetter(1))[1]
            if _min == _max:
                continue
            for o_id, value in values.items():
                adj_offers[o_id] += self.WEIGHTS[measure] * ((value - _min) / (_max - _min) - 0.5)
        return adj_offers


if __name__ == '__main__':
    matchmaker = Matchmaking()
    while True:
        input_data = sys.stdin.readline()
        if input_data is "\n":
            pass
        else:
            re_sq_long = r"""
                # Match double quoted string with escaped stuff. 
                "            # Opening literal quote
                (            # $1: Capture string contents
                  [^"\\]*    # Zero or more non-', non-backslash
                  (?:        # "unroll-the-loop"!
                    \\.      # Allow escaped anything.
                    [^"\\]*  # Zero or more non-', non-backslash
                  )*         # Finish {(special normal*)*} construct.
                )            # End $1: String contents.
                "            # Closing literal quote
                """
            matches = re.findall(re_sq_long, input_data, re.DOTALL | re.VERBOSE)
            test = matches[0].strip().replace('\\', "")
            result = matchmaker.perform_matchmaking(json.loads(test), int(input_data.rsplit(None, 1)[-1]),
                                                    MatchmakingMethodology.MAX_FLOW)
            print(result)
