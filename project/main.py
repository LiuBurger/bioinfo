import torch as pt
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from data import names
from model import FakeModel


def main():

    # parse the arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('query', type=str, help='query file')
    argparser.add_argument('--output', type=str, help='output file')
    args = argparser.parse_args()

    query = args.query

    # handle the query input
    with open(query) as f:
        queries = f.readlines()
    queries = [x.strip() for x in queries] 

    dataloader = DataLoader(queries, batch_size=256, shuffle=False)

    model = FakeModel()
    # model.load_state_dict(pt.load('model.pt'))  


    idx = 0

    outputs = []

    for data in dataloader:
        # get the output of the model
        preds = model(data)

        for p in preds:
            # get top 12 predictions
            top12 = pt.topk(p, 12)
            # get the names of the top 12 predictions
            top12_names = [names[i] for i in top12.indices.tolist()]
            for name in top12_names:
                outputs.append(f"QUERY{idx}\t{name}")

            idx +=1 

    # write the output to a file if --output is specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(outputs))

    else:
        print('\n'.join(outputs))


if __name__ == '__main__':
    main()