from comparison1.grc import GRC
from comparison2.mcts import MCTS
from comparison3.reinforce import RL
from Mine.RLN import RLN
from Mine.RLNL import *
from MIne_CE.agent2 import Agent2
from Mine_CEB.agent3 import Agent3
from Mine_C.agent1 import Agent1
from network import Network
import tensorflow as tf


class Algorithm:

    def __init__(self, name, param=10):
        self.name = name
        self.agent = None
        self.param = param
        self.evaluation = Evaluation()
        self.link_env = None

    def execute(self, network_path, sub_filename, req_num=1000):
        networks = Network(network_path)
        sub, requests = networks.get_networks(sub_filename, req_num)

        tf.reset_default_graph()
        with tf.Session() as sess:
            self.configure(sub, sess)
            start = time.time()
            self.handle(sub, requests)
            runtime = time.time() - start

        tf.get_default_graph().finalize()

        return runtime

    def configure(self, sub, sess=None):

        if self.name == 'GRC':
            agent = GRC(damping_factor=0.9, sigma=1e-6)

        elif self.name == 'MCTS':
            agent = MCTS(computation_budget=5, exploration_constant=0.5)

        elif self.name == 'RL':
            training_set_path = 'comparison3/nodetrset/'
            networks = Network(training_set_path)
            training_set = networks.get_reqs_for_train(1000)
            agent = RL(sub=sub,
                       n_actions=sub.number_of_nodes(),
                       n_features=4,
                       learning_rate=0.05,
                       epoch_num=self.param,
                       batch_size=100)
            agent.train(training_set)

        elif self.name == 'RLN1':
            training_set_path = 'Mine/nodetrset/'
            networks = Network(training_set_path)
            training_set = networks.get_reqs_for_train(1000)
            agent = Agent1(sub=sub,
                       n_actions=sub.number_of_nodes(),
                       n_features=5,
                       learning_rate=0.05,
                       num_epoch=self.param,
                       batch_size=100)
            agent.train(training_set)

        elif self.name == 'RLN2':
            training_set_path = 'Mine/nodetrset/'
            networks = Network(training_set_path)
            training_set = networks.get_reqs_for_train(1000)
            agent = Agent2(sub=sub,
                       n_actions=sub.number_of_nodes(),
                       n_features=6,
                       learning_rate=0.05,
                       num_epoch=self.param,
                       batch_size=100)
            agent.train(training_set)

        elif self.name == 'RLN3':
            training_set_path = 'Mine/nodetrset/'
            networks = Network(training_set_path)
            training_set = networks.get_reqs_for_train(1000)
            agent = Agent3(sub=sub,
                       n_actions=sub.number_of_nodes(),
                       n_features=7,
                       learning_rate=0.05,
                       num_epoch=self.param,
                       batch_size=100)
            agent.train(training_set)
        elif self.name == 'RLN':
            training_set_path = 'Mine/nodetrset/'
            networks = Network(training_set_path)
            training_set = networks.get_reqs_for_train(1000)
            agent = RLN(sub=sub,
                       n_actions=sub.number_of_nodes(),
                       n_features=5,
                       learning_rate=0.05,
                       num_epoch=self.param,
                       batch_size=100)
            agent.train(training_set)
            nodesaver = tf.train.Saver()
            nodesaver.save(agent.sess, "./Mine/nodemodel/nodemodel.ckpt")
        else:
            self.link_env=LinkEnv(sub)
            training_set_path = 'Mine/linktrset/'
            networks = Network(training_set_path)
            training_set = networks.get_reqs_for_train(1000)
            if sub.number_of_nodes()>60:
                linknum=59614
            else:
                linknum=60278
            agent = RLNL(sub=sub,
                           n_actions=linknum,
                           n_features=2,
                           learning_rate=0.05,
                           num_epoch=self.param,
                           batch_size=100)
            agent.train(training_set)
            linksaver = tf.train.Saver()
            linksaver.save(agent.sess, "./Mine/linkmodel/linkmodel.ckpt")
        self.agent = agent

    def handle(self, sub, requests):

        for req in requests:
            req_id = req.graph['id']
            if req.graph['type'] == 0:
                print("\nTry to map request%s: " % req_id)
                self.mapping(sub, req)

            if req.graph['type'] == 1:
                Network.recover(sub, req)

    def mapping(self, sub, req):
        """两步映射：先节点映射阶段再链路映射阶段"""

        self.evaluation.total_arrived += 1

        # mapping virtual nodes
        node_map = self.node_mapping(sub, req)

        if len(node_map) == req.number_of_nodes():
            # mapping virtual links
            print("link mapping...")
            link_map = self.link_mapping(sub, req, node_map)
            if len(link_map) == req.number_of_edges():
                Network.allocate(sub, req, node_map, link_map)
                # 更新实验结果
                self.evaluation.collect(sub, req, link_map)
                print("Success!")
                return True
            else:
                print("Failed to map all links!")
                return False
        else:
            print("Failed to map all nodes!")
            return False

    def node_mapping(self, sub, req):
        """求解节点映射问题"""

        print("node mapping...")
        node_map = {}

        if self.name != 'RLNL':
            node_map = self.agent.run(sub, req)

        else:
            nodeenv = NodeEnv(sub)
            nodeenv.set_vnr(req)
            nodep = nodepolicy(nodeenv.action_space.n, nodeenv.observation_space.shape)
            nodeobservation = nodeenv.reset()
            for vn_id in range(req.number_of_nodes()):
                sn_id = nodep.choose_max_action(nodeobservation, nodeenv.sub,
                                                req.nodes[vn_id]['cpu'],
                                                req.number_of_nodes())
                if sn_id == -1:
                    break
                else:
                    # 执行一次action，获取返回的四个数据
                    nodeobservation, _, done, info = nodeenv.step(sn_id)
                    node_map.update({vn_id: sn_id})

        # 返回节点映射集合
        return node_map

    def link_mapping(self, sub, req, node_map):

        if self.name=="GRC":
            # 剪枝后再寻最短路径
            link_map = Network.cut_then_find_path(sub, req, node_map)
        elif self.name == "RLNL":
            link_map = self.agent.run(sub, req, node_map, self.link_env)
        else:
            # K最短路径
            link_map = Network.find_path(sub, req, node_map)

        return link_map
