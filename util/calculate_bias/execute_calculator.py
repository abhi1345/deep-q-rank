import os
from .calculate_docs_bias import calculate_docs_bias
from .calculate_run_bias import calculate_run_bias
from .calculate_cumulative_bias import calculate_cumulative_bias

def calculate_bias(unbiased_run_file):

    biased_run_file = "/home/shiva_soleimany/RL/deep-q-rerank/util/calculate_bias/Run_neutral.txt"
    docs_bias_file = 'documents_bias_tf.pkl'

    if not os.path.exists(biased_run_file) or not os.path.exists(unbiased_run_file):
        print("Error: One or both run files do not exist.")
        return

    if not os.path.exists(docs_bias_file):
        calculate_docs_bias()

    calculate_run_bias(biased_run_file, unbiased_run_file)
    results = calculate_cumulative_bias()

    return results

if __name__ == "__main__":
    calculate_bias()