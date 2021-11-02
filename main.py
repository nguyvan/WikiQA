from package.data.dataset import get_data, QADataset
from package.utils import Tokenizer, Padding
import argparse
import torch
from package.model.simpleQAmodel.models import get_model
from package.model.simpleQAmodel.optimize import CosineWithRestarts
from package.model.simpleQAmodel.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=16)
    opt = parser.parse_args()
    
    opt.device = 0 if opt.no_cuda is False else -1
    
    root_dir = "WikiQACorpus"
    train_txt = "WikiQA-train.txt"
    test_txt = "WikiQA-test.txt"
    val_txt = "WikiQA-dev.txt"
    
    transform = {
        "tokenize": Tokenizer(num_words=1000),
        "padding": Padding()
    }

    train_sample, test_sample, val_sample, tokenize = get_data(root_dir, train_txt, test_txt, val_txt, opt.batch_size, **transform)

    # for indx, train in enumerate(train_sample):
    #     print(type(train.get("question")[0]))
    #     break
    opt.train = train_sample
    opt.val = val_sample
    opt.test = test_sample
    model = get_model(opt, len(tokenize.word_index), len(tokenize.word_index))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    if opt.SGDR:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/" %
              opt.checkpoint)
    
    train_model(model, opt)


if __name__ == "__main__":
    main()