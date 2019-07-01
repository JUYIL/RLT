import copy
import numpy as np
from network import Network


class GRC:
    def __init__(self, damping_factor, sigma):
        self.damping_factor = damping_factor
        self.sigma = sigma

    def run(self, sub, req):

        # 对底层节点排名
        sub_grc_vector = self.calculate_grc(sub)
        # 对虚拟节点排名
        req_grc_vector = self.calculate_grc(req, category='req')

        node_map = {}
        sub_copy = copy.deepcopy(sub)
        for v_node in req_grc_vector:
            v_id = v_node[0]
            for s_node in sub_grc_vector:
                s_id = s_node[0]
                if s_id not in node_map.values() and sub_copy.nodes[s_id]['cpu_remain'] > req.nodes[v_id]['cpu']:
                    node_map.update({v_id: s_id})
                    tmp = sub_copy.nodes[s_id]['cpu_remain'] - req.nodes[v_id]['cpu']
                    sub_copy.nodes[s_id]['cpu_remain'] = round(tmp, 6)
                    break
        return node_map

    def calculate_grc(self, graph, category='substrate'):
        """calculate grc vector of a substrate network or a virtual network"""

        if category == 'req':
            cpu_type = 'cpu'
            bw_type = 'bw'
        else:
            cpu_type = 'cpu_remain'
            bw_type = 'bw_remain'

        cpu_vector, m_matrix = [], []
        n = graph.number_of_nodes()
        for u in range(n):
            cpu_vector.append(graph.nodes[u][cpu_type])
            sum_bw = Network.calculate_adjacent_bw(graph, u, bw_type)
            for v in range(n):
                if v in graph.neighbors(u):
                    m_matrix.append(graph[u][v][bw_type] / sum_bw)
                else:
                    m_matrix.append(0)
        cpu_vector = np.array(cpu_vector) / np.sum(cpu_vector)
        m_matrix = np.array(m_matrix).reshape(n, n)
        cpu_vector *= (1 - self.damping_factor)
        current_r = cpu_vector
        delta = float('inf')
        while delta >= self.sigma:
            mr_vector = np.dot(m_matrix, current_r)
            mr_vector *= self.damping_factor
            next_r = cpu_vector + mr_vector
            delta = np.linalg.norm(next_r - current_r, ord=2)
            current_r = next_r

        output = []
        for i in range(n):
            output.append((i, current_r[i]))
        output.sort(key=lambda element: element[1], reverse=True)
        return output
