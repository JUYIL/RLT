from algorithm import Algorithm
from analysis import Analysis


if __name__ == '__main__':

    tool = Analysis('results_algorithm/')
    name = 'RLNL'
    algorithm = Algorithm(name, param=5)
    runtime = algorithm.execute(network_path='networks/',
                                sub_filename='subts.txt',
                                req_num=2000)
    tool.save_evaluations(algorithm.evaluation, '%s.txt' % name)
    print(runtime)
