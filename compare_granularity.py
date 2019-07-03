from algorithm import Algorithm
from analysis import Analysis


if __name__ == '__main__':

    tool = Analysis('results_granularity/')
    for i in range(1,3):
        granularity = i + 1
        algorithm = Algorithm('RLN%d' % granularity,
                              param=10,)
        algorithm.execute(network_path='networks/',
                          sub_filename='subts.txt')
        tool.save_evaluations(algorithm.evaluation, '%s.txt' % granularity)
