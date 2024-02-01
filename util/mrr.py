import argparse
import pandas as pd


def calculate_MRR(qrel_file, run_file, k):

    qrel, run = {}, {}

    with open(qrel_file, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split("\t")
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    if isinstance(run_file, pd.DataFrame):
        for index, row in run_file.iterrows():
            qid = str(row['qid'])
            did = str(row['doc_id'])
            if qid not in run:
                run[qid] = []
            run[qid].append(did)
    else:
        with open(run_file, 'r') as f_run:
            for line in f_run:
                qid, _, did, *_ = line.strip().split(" ")
                if qid not in run:
                    run[qid] = []
                run[qid].append(did)
        
    mrr = 0.0
    qids = []
    rrs = []

    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i+1)
                break
        qids.append(qid)
        rrs.append(rr)
        mrr += rr
    mrr /= len(run)

    print("MRR@10: ", mrr)
    return mrr
    



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-qrels', type=str, default='')
    parser.add_argument('-run', type=str, default='')
    parser.add_argument('-metric', type=str, default='mrr_cut_10')
#    parser.add_argument('-rrs_path',default='../mrrs/tire_akhar/rrs_only_male_female.csv', type=str)
    args = parser.parse_args()

    metric = args.metric
    k = int(metric.split('_')[-1])
    
    mrr = calculate_MRR(args.qrels, args.run, k)


if __name__ == "__main__":
    main()