#include<iostream>
#include "idx2word.h"
#include "word2idx.h"
#include "firmware/seq2seq.h"
#include <vector>

#define MAX_LEN_INPUT 63
#define MAX_LEN_TARGET

int main()
{
    // word to index
    vector<string> input_sentence = {"das", "kann", "doch", "nicht", "wahr", "sein"};
    int input_idx[MAX_LEN_INPUT];
    for (int i=0; i<input_sentence.size(); i++)
        input_idx[i] = word2idx[input_sentence[i]];

    // encoder 

    // decoder
}