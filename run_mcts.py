from algorithm import Algorithm
from analysis import Analysis

if __name__ == '__main__':

    tool = Analysis('results_algorithm/')
    name = 'MCTS'
    algorithm = Algorithm(name)
    runtime = algorithm.execute(network_path='networks/',
                                sub_filename='subsj.txt',
                                req_num=2000)
    tool.save_evaluations(algorithm.evaluation, '%s.txt' % name)
    print(runtime)