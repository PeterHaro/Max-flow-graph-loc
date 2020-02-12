""" Matchmaking"""
import json
from collections import defaultdict
from functools import partial
from operator import itemgetter
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx

magic = {'out': 'transactions'}
names = ['buyers', 'suppliers', 'transactions']
o_names = ['offers_1', 'offers_2', 'offers_3']
weights = {'flow': 0.1,
           'price': -0.1,
           'distance': 0.1,
           }


def load_data(_names, _o_names):
    """ dict of lists of dicts """
    _data = dict()

    for name in _names:
        with open(f'static/{name}.json') as fp:
            _data[name] = json.load(fp)
    # temp - unique node ids
    # for s in _data['suppliers']:
    #     s['id'] *= -1
    # for b in _data['buyers']:
    #     try:
    #         for trans in b['historicalData']:
    #             trans['supplierId'] *= -1
    #     except TypeError:
    #         pass
    # for name in _names:
    #     with open(f'input/{name}.json', 'w') as fp:
    #         json.dump(_data[name], fp)
    # end temp
    _orders = dict()
    for name in _o_names:
        with open(f'static/{name}.json') as fp:
            _orders[name] = json.load(fp)

    return _data, _orders


def generate_graph(_data):
    """ Generate graph with suppliers/buyers as nodes\
    transactions as edges, #transactions as degree"""
    G = nx.Graph()

    def avg_likert(_d: dict):
        num_rt = ["qualityLikert", "deliveryTimeLikert", "packagingLikert", "responseRateLikert",
                  "overallSatesfactionLikert"]
        l = 5  # len(num_rt)
        avg = sum(_d[k] for k in num_rt) / l
        return avg

    # nodes
    for supplier in _data['suppliers']:
        G.add_node(supplier['id'])

    for buyer in _data['buyers']:
        n = buyer['id']
        G.add_node(n)
        # edges
        try:
            for trans in buyer['historicalData']:
                n1 = trans['supplierId']
                if not n1 in G[n]:
                    G.add_edge(n, n1, degree=0)
                    G[n][n1]['weight'] = 0
                G[n][n1]['degree'] += 1
                G[n][n1]['weight'] += avg_likert(trans)
        except TypeError:
            pass  # empty transaction list
    return G


def draw_graph(_graph):
    """ Visualise graph """
    plt.subplot()
    draw_args = {'with_labels': True,
                 'font_weight': 'bold',
                 'edge_color': 'royalblue',
                 'alpha': 0.05}
    nx.draw(_graph, **draw_args)
    plt.show()


def dump_graph(_graph, out='transactions'):
    """ Dump networkx graph to graphml lxml """
    nx.write_graphml_lxml(_graph, f'output/{out}.graphml')


def matchmaking(offers, buyer, _data, method='connected_component'):
    """ matchmaking functions """
    G = generate_graph(_data)

    def connected_component():
        """ naive: match is -1 if in different graph components """
        adj_offers = dict()
        for offer in offers:
            if offer['supplierId'] in nx.algorithms.descendants(G, buyer):
                adj_offers[offer['id']] = offer['semanticSimilarity']
            else:
                adj_offers[offer['id']] = offer['semanticSimilarity'] - 1
        return adj_offers

    def max_flow(G, buyer):
        mf = partial(nx.maximum_flow_value, G, buyer, capacity='weight')
        adj_offers = dict()
        _temp_rel = defaultdict(dict)  # save price + dist, to extract min/max, normalisation
        for offer in offers:
            o_id = offer['id']
            adj_offers[o_id] = offer['semanticSimilarity']
            if offer['supplierId'] in nx.algorithms.descendants(G, buyer):
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
                adj_offers[o_id] += weights[measure] * ((value - _min) / (_max - _min) - 0.5)
        return adj_offers

    def svd():
        # import numpy as np
        # import pandas as pd
        # from scipy.linalg import sqrtm

        # format data
        "OverallSatesfactionLikert"
        return 0

    if method == 'connected_component':
        res = connected_component()
    elif method == 'max_flow':
        res = max_flow(G, buyer)
    else:
        pprint(f"{method} not implemented")
    return sorted(res.items(), key=itemgetter(1), reverse=True)


if __name__ == '__main__':
    data, offerses = load_data(names, o_names)
    H = generate_graph(data)
    # draw_graph(H.subgraph(list(nx.connected_components(H))[0]))

    # dump_graph(H)

    pprint(matchmaking(offerses[o_names[0]], -5, data, method='max_flow'))
