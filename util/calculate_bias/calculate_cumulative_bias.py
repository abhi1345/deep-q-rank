import numpy as np
import pickle


def calculate_cumulative_bias():
    experiments = ['biased', 'unbiased']

    metrics = ['ARaB']
    methods = ['tf']

    qry_bias_paths = {}
    for metric in metrics:
        qry_bias_paths[metric] = {}
        for exp_name in experiments:
            qry_bias_paths[metric][exp_name] = {}
            for _method in methods:
                qry_bias_paths[metric][exp_name][_method] = f'run_bias_{exp_name}_{_method}_{metric}.pkl'

    queries_gender_annotated_path = "/home/shiva_soleimany/RL/deep-q-rerank/util/calculate_bias/queries_gender_annotated.csv"

    at_rank_list = [5, 10, 20]

    query_bias_per_query = {}

    for metric in metrics:
        query_bias_per_query[metric] = {}
        for exp_name in experiments:
            query_bias_per_query[metric][exp_name] = {}
            for _method in methods:
                _path = qry_bias_paths[metric][exp_name][_method]
                print(f"Loading {metric} bias values from {_path} ...")
                with open(_path, 'rb') as fr:
                    query_bias_per_query[metric][exp_name][_method] = pickle.load(fr)

    queries_effective = {}
    with open(queries_gender_annotated_path, 'r') as fr:
        for line in fr:
            vals = line.strip().split(',')
            qid = int(vals[0])
            qtext = ' '.join(vals[1:-1])
            qgender = vals[-1]
            if qgender == 'n':
                queries_effective[qid] = qtext
    len(queries_effective)

    eval_results_bias = {}
    eval_results_feml = {}
    eval_results_male = {}

    for metric in metrics:
        eval_results_bias[metric] = {}
        eval_results_feml[metric] = {}
        eval_results_male[metric] = {}
        for exp_name in experiments:
            eval_results_bias[metric][exp_name] = {}
            eval_results_feml[metric][exp_name] = {}
            eval_results_male[metric][exp_name] = {}
            for _method in methods:
                eval_results_bias[metric][exp_name][_method] = {}
                eval_results_feml[metric][exp_name][_method] = {}
                eval_results_male[metric][exp_name][_method] = {}
                for at_rank in at_rank_list:
                    _bias_list = []
                    _feml_list = []
                    _male_list = []

                    for qid in queries_effective.keys():
                        if qid in query_bias_per_query[metric][exp_name][_method][at_rank]:
                            _bias_list.append(query_bias_per_query[metric][exp_name][_method][at_rank][qid][0])
                            _feml_list.append(query_bias_per_query[metric][exp_name][_method][at_rank][qid][1])
                            _male_list.append(query_bias_per_query[metric][exp_name][_method][at_rank][qid][2])
                        else:
                            pass
                            # print ('missing', metric, exp_name, _method, at_rank, qid)

                    eval_results_bias[metric][exp_name][_method][at_rank] = np.mean(
                        [(_male_x - _feml_x) for _male_x, _feml_x in zip(_male_list, _feml_list)])
                    eval_results_feml[metric][exp_name][_method][at_rank] = np.mean(_feml_list)
                    eval_results_male[metric][exp_name][_method][at_rank] = np.mean(_male_list)

    print()
    results = []
    for metric in metrics:
        for at_rank in at_rank_list:
            for _method in methods:
                for exp_name in experiments:
                    res = f"{exp_name:15}\tcutoff@{at_rank}\t{_method}\t{eval_results_bias[metric][exp_name][_method][at_rank]:.6f}"
                    print(res)
                    results.append(res)
            print("------------------------------------------------")

    return results


if __name__ == "__main__":
    calculate_cumulative_bias()
