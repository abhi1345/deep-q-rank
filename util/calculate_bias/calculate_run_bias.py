import numpy as np
import pickle
import argparse


def calculate_run_bias(biased_run_file, unbiased_run_file):
    experiments = {
        'unbiased': unbiased_run_file,
        'biased': biased_run_file
    }

    docs_bias_paths = {'tf': "documents_bias_tf.pkl"}

    at_rank_list = [5, 10, 20]
    queries_gender_annotated_path = "/home/shiva_soleimany/RL/deep-q-rerank/util/calculate_bias/queries_gender_annotated.csv"

    # Loading saved document bias values
    docs_bias = {}
    for _method in docs_bias_paths:
        print(f"Loading {_method} bias values ...")
        with open(docs_bias_paths[_method], 'rb') as fr:
            docs_bias[_method] = pickle.load(fr)
            docs_bias[_method] = {int(k): v for k, v in docs_bias[_method].items()}

    # Loading gendered queries
    qids_filter = []
    with open(queries_gender_annotated_path, 'r') as fr:
        for line in fr:
            vals = line.strip().split(',')
            qid = int(vals[0])
            qids_filter.append(qid)

    print(f"Total number of gendered queries: {len(qids_filter)}")

    # Loading run files
    runs_docs_bias = {}
    for exp_name in experiments:
        run_path = experiments[exp_name]
        runs_docs_bias[exp_name] = {}

        for _method in docs_bias_paths:
            runs_docs_bias[exp_name][_method] = {}

        with open(run_path) as fr:
            qid_cur = 0
            for i, line in enumerate(fr):
                data = line.strip().split()
                qid = int(data[0])
                doc_id = int(data[2])

                if qid not in qids_filter:
                    continue

                if qid != qid_cur:
                    for _method in docs_bias_paths:
                        runs_docs_bias[exp_name][_method][qid] = []
                    qid_cur = qid

                for _method in docs_bias_paths:
                    runs_docs_bias[exp_name][_method][qid].append(docs_bias[_method][doc_id])

        for _method in docs_bias_paths:
            print(f"Number of effective queries in {exp_name} using {_method} : {len(runs_docs_bias[exp_name][_method].keys())}")

    def calc_RaB_q(bias_list, at_rank):
        bias_val = np.mean([x[0] for x in bias_list[:at_rank]])
        bias_feml_val = np.mean([x[1] for x in bias_list[:at_rank]])
        bias_male_val = np.mean([x[2] for x in bias_list[:at_rank]])

        return bias_val, bias_feml_val, bias_male_val

    def calc_ARaB_q(bias_list, at_rank):
        _vals = []
        _feml_vals = []
        _male_vals = []
        for t in range(at_rank):
            if len(bias_list) >= t + 1:
                _val_RaB, _feml_val_RaB, _male_val_RaB = calc_RaB_q(bias_list, t + 1)
                _vals.append(_val_RaB)
                _feml_vals.append(_feml_val_RaB)
                _male_vals.append(_male_val_RaB)

        bias_val = np.mean(_vals)
        bias_feml_val = np.mean(_feml_vals)
        bias_male_val = np.mean(_male_vals)

        return bias_val, bias_feml_val, bias_male_val

    print('Calculating ranking bias ...')
    query_bias_ARaB = {}
    for exp_name in experiments:
        query_bias_ARaB[exp_name] = {}

        for _method in docs_bias_paths:
            print(f"Calculating ranking bias for {exp_name} based on {_method} ...")

            query_bias_ARaB[exp_name][_method] = {}

            for at_rank in at_rank_list:
                query_bias_ARaB[exp_name][_method][at_rank] = {}

                for qid in runs_docs_bias[exp_name][_method]:
                    query_bias_ARaB[exp_name][_method][at_rank][qid] = calc_ARaB_q(runs_docs_bias[exp_name][_method][qid], at_rank)

    print('Saving results ...')
    for exp_name in experiments:
        for _method in docs_bias_paths:
            save_path = f"run_bias_{exp_name}_{_method}"

            with open(save_path + '_ARaB.pkl', 'wb') as fw:
                pickle.dump(query_bias_ARaB[exp_name][_method], fw)

            print(f"Results saved to {save_path}")


if __name__ == "__main__":
    biased = '/home/sajadeb/LLaMA_Debiasing/BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased/Run_neutral.txt',
    unbiased = '/home/sajadeb/LLaMA_Debiasing/deep-q-rank/output/dqn_model/Run_nn.txt'
    calculate_run_bias(biased, unbiased)
