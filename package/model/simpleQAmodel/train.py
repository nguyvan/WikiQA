import time
import torch
from torch._C import dtype
from .models import get_model
import torch.nn.functional as F
from .optimize import CosineWithRestarts
from .batch import create_masks
import dill as pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, opt):
    
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
                 
    for epoch in range(opt.epochs):

        total_loss_seq = 0
        total_loss_label = 0
        print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
        ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
                    
        for i, batch in enumerate(opt.train): 
            len_question = len(batch["question"])
            question = torch.cat(batch["question"]).reshape((len_question, -1)).transpose(0,1).to(device=device)
            len_answer = len(batch["answer"])
            answer = torch.cat(batch["answer"]).reshape((len_answer, -1)).transpose(0,1).to(device=device)
            label = batch["label"].reshape((-1, 1)).to(device=device, dtype=torch.float)
            answer_input = answer[:, :-1]
            question_mask, answer_mask = question.unsqueeze(-2), answer_input.unsqueeze(-2)
            preds_seq, preds_label = model(question, answer_input, question_mask, answer_mask)
            ys = answer[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss_seq = F.cross_entropy(preds_seq.view(-1, preds_seq.size(-1)), ys)
            loss_seq.backward(retain_graph=True)
            loss_label = F.binary_cross_entropy(preds_label, label)
            
            loss_label.backward(retain_graph=True)

            opt.optimizer.step()
            if opt.SGDR == True: 
                opt.sched.step()
            
            total_loss_seq += loss_seq.item()
            total_loss_label += loss_label.item()
            if (i + 1) % opt.printevery == 0:
                p = int(100 * (i + 1) / len(opt.train))
                avg_loss_seq = total_loss_seq/opt.printevery
                avg_loss_label = total_loss_label/opt.printevery
                print("   %dm: epoch %d [%s%s]  %d%%  loss_seq = %.3f  loss_label = %.3f " %\
                ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss_seq, avg_loss_label))
                total_loss_seq = 0
                total_loss_label = 0
            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
   
   
        print("%dm: epoch %d [%s%s]  %d%%  loss_seq = %.3f   loss_label = %.3f\nepoch %d complete, loss = %.03f   loss_label = %.3f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss_seq, total_loss_label, epoch + 1, avg_loss_seq, avg_loss_label))