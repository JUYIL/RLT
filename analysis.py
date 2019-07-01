import os
import networkx as nx
import matplotlib.pyplot as plt


class Analysis:

    def __init__(self, result_dir):

        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.metric_names = {'acceptance ratio': 'Acceptance Ratio',
                             'average revenue': 'Long Term Average Revenue',
                             'average cost': 'Long Term Average Cost',
                             'R_C': 'Long Term Revenue/Cost Ratio',
                             'node utilization': 'Average Node Utilization',
                             'link utilization': 'Average Link Utilization'}

        self.algorithm_lines = ['b:', 'g--', 'y-.', 'r-']
        self.algorithm_names = ['GRC', 'MCTS', 'RL', 'ML']

        self.epoch_lines = ['b-', 'r-', 'y-', 'g-', 'c-', 'm-']
        self.epoch_types = ['50', '60', '70', '80', '90', '100']

        self.granularity_lines = ['g-', 'r--', 'b:']
        self.granularity_types = ['cpu', 'cpu, flow', 'cpu, flow, queue']

        self.multi_lines = ['g-', 'b--', 'r-.']
        self.multi_types = ['situation 1', 'situation 2', 'situation 3']

    def save_evaluations(self, evaluation, filename):
        """将一段时间内底层网络的性能指标输出到指定文件内"""

        filename = self.result_dir + filename
        with open(filename, 'w') as f:
            for time, evaluation in evaluation.metrics.items():
                f.write("%-10s\t" % time)
                f.write("%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\n" % evaluation)

    def save_epoch(self, epoch, acc, runtime):
        """保存不同采样次数的实验结果"""

        filename = self.result_dir + 'epoch.txt'
        with open(filename, 'a') as f:
            f.write("%-10s\t%-20s\t%-20s\n" % (epoch, acc, runtime))

    def save_loss(self, runtime, epoch_num, loss_average, name):
        filename = self.result_dir + '%s-%s.txt' % (name,epoch_num)
        with open(filename, 'w') as f:
            f.write("Training time: %s hours\n" % runtime)
            for value in loss_average:
                f.write(str(value))
                f.write('\n')

    def read_result(self, filename):
        """读取结果文件"""

        with open(self.result_dir + filename) as f:
            lines = f.readlines()

        t, acceptance, revenue, cost, r_to_c, node_stress, link_stress = [], [], [], [], [], [], []
        for line in lines:
            a, b, c, d, e, f, g = [float(x) for x in line.split()]
            t.append(a)
            acceptance.append(b)
            revenue.append(c / a)
            cost.append(d / a)
            r_to_c.append(e)
            node_stress.append(f)
            link_stress.append(g)

        return t, acceptance, revenue, cost, r_to_c, node_stress, link_stress

    def draw_result_algorithms(self):
        """绘制实验结果图"""

        results = []

        for alg in self.algorithm_names:
            results.append(self.read_result(alg + '.txt'))

        index = 0
        for metric, title in self.metric_names.items():
            index += 1
            if index == 2 or index == 3:
                continue
            plt.figure()
            for alg_id in range(len(self.algorithm_names)):
                x = results[alg_id][0]
                y = results[alg_id][index]
                plt.plot(x, y, self.algorithm_lines[alg_id], label=self.algorithm_names[alg_id])
            plt.xlim([25000, 50000])
            plt.legend(loc='lower right', fontsize=12)
            if metric == 'acceptance ratio':
                plt.ylim([0.7, 1])
            if metric == 'R_C':
                plt.ylim([0.5, 0.8])
                plt.legend(loc='upper right', fontsize=12)
            if metric == 'node utilization':
                plt.ylim([0, 0.7])
            if metric == 'link utilization':
                plt.ylim([0, 0.3])
            plt.xlabel("time", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(title, fontsize=15)
            # plt.savefig(self.result_dir + metric + '.png')
        plt.show()

    def draw_result_granularity(self):
        """绘制实验结果图"""

        results = []
        for i in range(3):
            results.append(self.read_result('ML-VNE-%d.txt' % (i+1)))

        index = 0
        for metric, title in self.metric_names.items():
            index += 1
            plt.figure()
            for i in range(3):
                x = results[i][0]
                y = results[i][index]
                plt.plot(x, y, self.granularity_lines[i], label=self.granularity_types[i])
            plt.xlim([25000, 50000])
            if metric == 'acceptance ratio' or metric == 'node utilization':
                plt.ylim([0, 1])
            if metric == 'link utilization':
                plt.ylim([0, 0.5])
            plt.xlabel("time", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(title, fontsize=15)
            plt.legend(loc='lower right', fontsize=12)
            # plt.savefig(self.result_dir + metric + '.png')
        plt.show()

    def draw_result_multi(self):
        """绘制实验结果图"""

        results = []
        for i in range(3):
            index = (i+1) * 1000
            results.append(self.read_result('ML-VNE-%s.txt' % index))

        plt.figure()
        for i in range(3):
            x = results[i][0]
            y = results[i][1]
            plt.plot(x, y, self.multi_lines[i], label=self.multi_types[i])
        plt.xlim([0, 50000])
        plt.ylim([0, 1])
        plt.xlabel("time", fontsize=12)
        plt.ylabel("acceptance ratio", fontsize=12)
        plt.title("Acceptance Ratio", fontsize=15)
        plt.legend(loc='lower right', fontsize=12)
        # plt.savefig(self.result_dir + metric + '.png')
        plt.show()

    def draw_loss(self, loss_filename):
        """绘制loss变化趋势图"""

        with open(self.result_dir + loss_filename) as f:
            lines = f.readlines()
        loss = []
        for line in lines[1:]:
            loss.append(float(line))
        plt.plot(loss)
        plt.show()

    def draw_epoch(self):
        """绘制时间变化趋势图"""

        with open(self.result_dir + 'epoch.txt') as f:
            lines = f.readlines()
        epoch, acc, runtime = [], [], []
        for line in lines:
            a, b, c = [float(x) for x in line.split()]
            epoch.append(a)
            acc.append(b)
            runtime.append(c)
        acc.sort()
        plt.figure()
        plt.plot(epoch, acc, 'b-o')
        plt.xlabel("epoch", fontsize=12)
        plt.ylabel("acceptance ratio", fontsize=12)
        plt.xticks([10, 30, 50, 70, 90, 110, 130, 150])

        plt.figure()
        plt.plot(epoch, runtime, 'b-o')
        plt.xticks([10, 30, 50, 70, 90, 110, 130, 150])
        plt.yticks([1000, 2000, 3000, 4000, 5000, 6000])
        plt.xlabel("epoch", fontsize=12)
        plt.ylabel("runtime", fontsize=12)
        plt.show()

    @staticmethod
    def draw_topology(graph):
        """绘制网络拓扑图"""

        nx.draw(graph, with_labels=False, node_color='black', edge_color='gray', node_size=50)
        # plt.savefig(self.result_dir + filename + '.png')
        # plt.close()
        plt.show()


if __name__ == '__main__':
    analysis = Analysis('results_new/')
    analysis.draw_result_algorithms()
