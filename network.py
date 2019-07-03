import copy
import networkx as nx
from itertools import islice


class Network:

    def __init__(self, path):
        self.files_dir = path

    def get_networks(self, sub_filename, req_num):
        """读取 req_num 个虚拟网络及 req_num*child_num 个子虚拟网络请求，构成底层虚拟网络请求事件队列和子虚拟网络请求事件队列"""
        # 底层物理网络
        sub = self.read_network_file(sub_filename)
        # 第1层虚拟网络请求
        queue1 = self.get_reqs(req_num)
        return sub, queue1

    def get_reqs(self, req_num):
        """读取req_num个虚拟网络请求文件，构建虚拟网络请求事件队列"""
        queue = []
        if req_num == 1000:
            offset = 2000 - req_num
        else:
            offset = 0
        for i in range(req_num):
            index = i + offset
            filename = 'req%d.txt' % index
            req_arrive = self.read_network_file(filename)
            req_arrive.graph['parent'] = -1
            req_arrive.graph['id'] = index
            req_leave = copy.deepcopy(req_arrive)
            req_leave.graph['type'] = 1
            req_leave.graph['time'] = req_arrive.graph['time'] + req_arrive.graph['duration']
            queue.append(req_arrive)
            queue.append(req_leave)
            # 按照时间（到达时间或离开时间）对这些虚拟网络请求从小到大进行排序
        queue.sort(key=lambda r:r.graph['time'])
        return queue

    def get_reqs_for_train(self, req_num):
        """读取req_num个虚拟网络请求文件，构建虚拟网络请求事件队列"""
        queue = []
        for i in range(req_num):
            filename = 'req%d.txt' % i
            req_arrive = self.read_network_file(filename)
            req_arrive.graph['id'] = i
            req_leave = copy.deepcopy(req_arrive)
            req_leave.graph['type'] = 1
            req_leave.graph['time'] = req_arrive.graph['time'] + req_arrive.graph['duration']
            queue.append(req_arrive)
            queue.append(req_leave)
        # 按照时间（到达时间或离开时间）对这些虚拟网络请求从小到大进行排序
        queue.sort(key=lambda r:r.graph['time'])
        return queue

    def read_network_file(self, filename):
        """读取网络文件并生成networkx.Graph实例"""

        mapped_info = {}
        node_id, link_id = 0, 0

        with open(self.files_dir + filename) as f:
            lines = f.readlines()

        # Step 1: 获取网络节点数量和链路数量，并根据网络类型进行初始化
        if len(lines[0].split()) == 2:
            """物理网络"""
            node_num, link_num = [int(x) for x in lines[0].split()]
            graph = nx.Graph(mapped_info=mapped_info)
        else:
            """虚拟网络"""
            node_num, link_num, time, duration, max_dis = [int(x) for x in lines[0].split()]
            graph = nx.Graph(type=0, time=time, duration=duration, mapped_info=mapped_info)

        # Step 2: 依次读取节点信息
        for line in lines[1: node_num + 1]:

            x, y, c = [float(x) for x in line.split()]
            graph.add_node(node_id,
                               x_coordinate=x, y_coordinate=y,
                               cpu=c, cpu_remain=c)
            node_id = node_id + 1

        # Step 3: 依次读取链路信息
        for line in lines[-link_num:]:
            """依次读取链路信息"""
            src, dst, bw, dis = [float(x) for x in line.split()]
            graph.add_edge(int(src), int(dst), link_id=link_id, bw=bw, bw_remain=bw, distance=dis)
            link_id = link_id + 1

        # Step 4: 返回网络实例
        return graph

    @staticmethod
    def get_path_capacity(sub, path):
        """找到一条路径中带宽资源最小的链路并返回其带宽资源值"""

        bandwidth = 1000
        head = path[0]
        for tail in path[1:]:
            if sub[head][tail]['bw_remain'] <= bandwidth:
                bandwidth = sub[head][tail]['bw_remain']
            head = tail
        return bandwidth

    @staticmethod
    def calculate_adjacent_bw(graph, u, kind='bw'):
        """计算一个节点的相邻链路带宽和，默认为总带宽和，若计算剩余带宽资源和，需指定kind属性为bw-remain"""

        bw_sum = 0
        for v in graph.neighbors(u):
            bw_sum += graph[u][v][kind]
        return bw_sum

    @staticmethod
    def getallpath(sub):
        i = 0
        k = 0
        linkaction = {}
        while i < sub.number_of_nodes():
            j = 0
            while j < sub.number_of_nodes():
                path = nx.shortest_simple_paths(sub, i, j)
                for p in path:
                    if 1 < len(p) < 6:
                        linkaction.update({k:{(i, j):p}})
                        k += 1
                    else:
                        break
                j += 1
            i += 1

        return linkaction

    @staticmethod
    def getbtns(sub):
        if sub.number_of_nodes() > 60:
            filename = "Mine/btns/"
        else:
            filename = "Mine/btnsj/"
        btns = []
        for i in range(1, 7):
            path = filename + 'btn%s.txt' % i
            with open(path) as file_object:
                lines = file_object.readlines()
            for line in lines:
                btns.append(float(line))
        return btns

    @staticmethod
    def k_shortest_path(graph, source, target, k=5):
        """K最短路径算法"""
        return list(islice(nx.shortest_simple_paths(graph, source, target), k))

    @staticmethod
    def cut_then_find_path(sub, req, node_map):
        """求解链路映射问题"""

        link_map = {}
        sub_copy = copy.deepcopy(sub)

        for vLink in req.edges:
            vn_from, vn_to = vLink[0], vLink[1]
            resource = req[vn_from][vn_to]['bw']
            # 剪枝操作，先暂时将那些不满足当前待映射虚拟链路资源需求的底层链路删除
            sub_tmp = copy.deepcopy(sub_copy)
            sub_edges = []
            for sLink in sub_tmp.edges:
                sub_edges.append(sLink)
            for edge in sub_edges:
                sn_from, sn_to = edge[0], edge[1]
                if sub_tmp[sn_from][sn_to]['bw_remain'] <= resource:
                    sub_tmp.remove_edge(sn_from, sn_to)

            # 在剪枝后的底层网络上寻找一条可映射的最短路径
            sn_from, sn_to = node_map[vn_from], node_map[vn_to]
            if nx.has_path(sub_tmp, source=sn_from, target=sn_to):
                path = Network.k_shortest_path(sub_tmp, sn_from, sn_to, 1)[0]
                link_map.update({vLink: path})

                # 这里的资源分配是暂时的
                start = path[0]
                for end in path[1:]:
                    bw_tmp = sub_copy[start][end]['bw_remain'] - resource
                    sub_copy[start][end]['bw_remain'] = round(bw_tmp, 6)
                    start = end
            else:
                break

        # 返回链路映射集合
        return link_map

    @staticmethod
    def find_path(sub, req, node_map):
        """求解链路映射问题"""

        link_map = {}
        for vLink in req.edges:
            vn_from = vLink[0]
            vn_to = vLink[1]
            sn_from = node_map[vn_from]
            sn_to = node_map[vn_to]
            if nx.has_path(sub, source=sn_from, target=sn_to):
                for path in Network.k_shortest_path(sub, sn_from, sn_to):
                    if Network.get_path_capacity(sub,path) >= req[vn_from][vn_to]['bw']:
                        link_map.update({vLink:path})
                        # 这里的资源分配是暂时的
                        start = path[0]
                        for end in path[1:]:
                            bw_tmp = sub[start][end]['bw_remain'] - req[vn_from][vn_to]['bw']
                            sub[start][end]['bw_remain'] = round(bw_tmp, 6)
                            start = end
                        break
                    else:
                        continue
        return link_map

    @staticmethod
    def allocate(sub, req, node_map, link_map):
        """分配节点和链路资源"""

        # 分配节点资源
        for v_id, s_id in node_map.items():
            cpu_tmp = sub.nodes[s_id]['cpu_remain'] - req.nodes[v_id]['cpu']
            sub.nodes[s_id]['cpu_remain'] = round(cpu_tmp, 6)

        # 分配链路资源
        for vl, path in link_map.items():
            link_resource = req[vl[0]][vl[1]]['bw']
            start = path[0]
            for end in path[1:]:
                bw_tmp = sub[start][end]['bw_remain'] - link_resource
                sub[start][end]['bw_remain'] = round(bw_tmp, 6)
                start = end

        # 更新映射信息
        mapped_info = sub.graph['mapped_info']
        mapped_info.update({req.graph['id']: (node_map, link_map)})
        sub.graph['mapped_info'] = mapped_info

    @staticmethod
    def recover(sub, req):
        """收回节点和链路资源"""

        req_id = req.graph['id']
        mapped_info = sub.graph['mapped_info']
        if req_id in mapped_info.keys():
            print("\nRelease the resources which are occupied by request%s" % req_id)

            # 读取该虚拟网络请求的映射信息
            node_map = mapped_info[req_id][0]
            link_map = mapped_info[req_id][1]

            # 释放节点资源
            for v_id, s_id in node_map.items():
                cpu_tmp = sub.nodes[s_id]['cpu_remain'] + req.nodes[v_id]['cpu']
                sub.nodes[s_id]['cpu_remain'] = round(cpu_tmp, 6)

            # 释放链路资源
            for vl, path in link_map.items():
                link_resource = req[vl[0]][vl[1]]['bw']
                start = path[0]
                for end in path[1:]:
                    bw_tmp = sub[start][end]['bw_remain'] + link_resource
                    sub[start][end]['bw_remain'] = round(bw_tmp, 6)
                    start = end

            # 移除相应的映射信息
            mapped_info.pop(req_id)
            sub.graph['mapped_info'] = mapped_info
