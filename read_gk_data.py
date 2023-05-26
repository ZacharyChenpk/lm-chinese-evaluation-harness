from datasets import *
import os
import re

path = "data/Multiple-choice_Questions"
files = os.listdir(path)
# reject_subject = ['Math', 'Physics', 'Chemistry']
# for f in files:
#     ds = load_dataset("json", data_files=path+"/2010-2013_English_MCQs.json", field="example")
ds = load_dataset("json", data_files=[os.path.join(path, a) for a in files 
    if 'MCQ' in a 
    and ('Chinese' in a or 'Geography' in a or 'History' in a or 'Political' in a)], field="example")
ds = ds.filter(lambda x: len(x['answer'])==1)
# drug_dataset_clean.save_to_disk("drug-reviews")

c2int = {'A':0, 'B':1, 'C':2, 'D':3}

def mapping(doc):
    s = doc['question'].replace('．', '.').replace('\n', '').replace('\u3000', ' ')
    ques = s.split('A.')[0].split('分）')[-1].strip()
    ans_A = s.split('A.')[1].split('B.')[0].strip()
    ans_B = s.split('B.')[1].split('C.')[0].strip()
    ans_C = s.split('C.')[1].split('D.')[0].strip()
    ans_D = s.split('D.')[1].strip()
    # ques = re.findall('(.+?)A[．.]+', s)
    # print(ques)
    return {
        'question': ques,
        'answer': c2int[doc['answer'][0]],
        'choices': [ans_A, ans_B, ans_C, ans_D]
    }

ds = ds.map(mapping, remove_columns=["year", 'category', 'score', 'analysis'])
ds = ds["train"]
ds_split = ds.train_test_split(test_size=0.1)
# print(ds_split)
ds_val = ds_split.pop("test")
ds_split = ds_split["train"].train_test_split(test_size=1/9)
ds_split["validation"] = ds_val

print(ds_split)
print(ds_split['train'][2])
ds_split.to_json("gaokaoMCQs")