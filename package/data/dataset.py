from typing import Iterable
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QASample(object):
    def __init__(self, question, answer, label):
        self.question = question
        self.answer = answer
        self.label = int(label)


class QADataset:
    def __init__(self, root_dir, txt_file):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = list()
        print("[SETUP] Reading Data ...")
        try:
            with open(os.path.join(root_dir, txt_file), encoding="utf8") as fin:
                for line in fin.readlines():
                    question, answer, label = line.split("\t")
                    self.data.append(QASample(question=question, answer=answer, label=label))
        except InterruptedError as error:
            print("[ERROR] {} ...".format(error))
            sys.exit()

        print("[FINISH] setup dones")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer, label = None, None, None
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, slice):
            question = [x.question for x in self.data[idx]]
            answer = [x.answer for x in self.data[idx]]
            label = [x.label for x in self.data[idx]]
        elif isinstance(idx, list):
            question = [x.question for id, x in enumerate(self.data) if id in idx]
            answer = [x.answer for id, x in enumerate(self.data) if id in idx]
            label = [x.label for id, x in enumerate(self.data) if id in idx]
        sample = {"question": question, "answer": answer, "label": label}
        return sample

    def get(self, text):
        for data in self.data:
            yield data.__getattribute__(text)


class QAData(Dataset):
    def __init__(self, question, answer, label):
        assert len(question) == len(answer) == len(label)
        self.question = question
        self.answer = answer
        self.label = label

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        sample = {"question": self.question[idx], "answer": self.answer[idx], "label": self.label[idx]}
        return sample

class DataManager:
    def __init__(self, train_data,
                 test_data,
                 val_data):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def _get_question_train(self):
        question_train = list(self.train_data.get("question"))
        question_val = list(self.val_data.get("question"))
        return question_train + question_val

    def _get_answer_train(self):
        answer_train = list(self.train_data.get("answer"))
        answer_val = list(self.val_data.get("answer"))
        answer_final = []
        for answer in answer_train + answer_val:
            answer_final.append("<SOS> " + answer + " <EOS>")
        return answer_final

    def _get_label_train(self):
        label_train = list(self.train_data.get("label"))
        label_val = list(self.val_data.get("label"))
        return label_train + label_val

    def get_data(self):
        return self._get_question_train(), self._get_answer_train(), self._get_label_train()


def get_data(root_dir, train_txt, test_txt, val_txt, batch_size=16, **transform):
    tokenize = transform["tokenize"]
    padding = transform["padding"]

    train_data = QADataset(root_dir=root_dir, txt_file=train_txt)
    test_data = QADataset(root_dir=root_dir, txt_file=test_txt)
    val_data = QADataset(root_dir=root_dir, txt_file=val_txt)

    manager = DataManager(train_data=train_data, test_data=test_data, val_data=val_data)
    question, answer, _ = manager.get_data()

    tokenize.fit_on_texts(question + answer)

    question_train_tokenize = tokenize.texts_to_sequences(list(train_data.get("question")))
    question_train_tokenize = padding(question_train_tokenize)
    answer_train_tokenize = tokenize.texts_to_sequences(list(train_data.get("answer")))
    answer_train_tokenize = padding(answer_train_tokenize)
    label_train = list(train_data.get("label"))
    train_sample = QAData(question=question_train_tokenize, answer=answer_train_tokenize, label=label_train)
    train_sample = DataLoader(train_sample, batch_size=16, shuffle=True)

    question_test_tokenize = tokenize.texts_to_sequences(list(test_data.get("question")))
    question_test_tokenize = padding(question_test_tokenize)
    answer_test_tokenize = tokenize.texts_to_sequences(list(test_data.get("answer")))
    answer_test_tokenize = padding(answer_test_tokenize)
    label_test = list(test_data.get("label"))
    test_sample = QAData(question=question_test_tokenize, answer=answer_test_tokenize, label=label_test)
    test_sample = DataLoader(test_sample, batch_size=16, shuffle=True)

    question_val_tokenize = tokenize.texts_to_sequences(list(val_data.get("question")))
    question_val_tokenize = padding(question_val_tokenize)
    answer_val_tokenize = tokenize.texts_to_sequences(list(val_data.get("answer")))
    answer_val_tokenize = padding(answer_val_tokenize)
    label_val = list(val_data.get("label"))
    val_sample = QAData(question=question_val_tokenize, answer=answer_val_tokenize, label=label_val)
    val_sample = DataLoader(val_sample, batch_size=batch_size, shuffle=True)

    return train_sample, test_sample, val_sample, tokenize
