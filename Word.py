import pickle
from nltk import word_tokenize,sent_tokenize
import pandas as pd
import pickle
from itertools import permutations
with open('all_words','rb') as input_file:
    all_words=pickle.load(input_file)

class Node():
    def __init__(self):
        self.count=0
        self.nodes=dict()
    def add_node(self,letter):
        if letter not in self.nodes:
            self.nodes[letter]=Node()
        return self.nodes[letter]
    def next(self,letter):
        return self.nodes[letter]

base_node=Node()
for i,word in enumerate(all_words):
        print(i/len(all_words))
        node=base_node
        for letter in word:
            node=node.add_node(letter)
        node.word=word
        node.count+=1

with open('node','wb') as output_file:
    pickle.dump(base_node,output_file)



test=['v','i','t','c','e']

test=permutations(test)



for case in test:
    node = base_node
    broke=0
    for letter in case:
        try:
            node=node.next(letter)
        except:
            #print(case,letter)
            broke=1
            break
    if not broke:
        try:
            print(case,node.word , node.count)
        except:
           pass



