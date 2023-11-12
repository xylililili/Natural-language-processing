from viterbi import viterbi
import numpy as np
import nltk
import random
# nltk.download('universal_tagset')
# nltk.download('brown')
# load data

# to calculate the accuracy


def accuracy(predicted_tag, actual_tag):
    """
    compute the accuracy of predicted tags

    """
    correct = 0
    total = len(actual_tag)
    total_tag = 0
    for i in range(total):
        sent_corr = 0
        for j in range(len(predicted_tag[i])):
            total_tag += 1
            if predicted_tag[i][j] == actual_tag[i][j]:
                correct += 1
                sent_corr += 1
        sent_Acc = sent_corr/len(predicted_tag[i])
        print(
            f"For the sentence {i}, {sent_corr} out of {len(predicted_tag[i])} is predicted correctly, the accuracy is {sent_Acc}.")
    print(f"Out of {total_tag} tags, {correct} were predicted correctly.")
    return correct/total_tag


def get_actual_tag(test_setences):
    actual_tag = []
    for sent in test_setences:
        tags = []
        for _, tag in sent:
            tags.append(tag)
        actual_tag.append(tags)
    return actual_tag


def predict():
    random.seed(42)
    tagset = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    test_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')[
        10150:10152]

    actual_tag = get_actual_tag(test_sentences)

    # remove the duplicated tags and words
    tags = set([tag for sent in tagset for _, tag in sent])
    words = set([word for sent in tagset for word, _ in sent])

    words.add("UNK")
    # Index the words and tags
    tag_index = {}
    i = 0
    for tag in tags:
        tag_index[tag] = i
        i += 1

    word_index = {}
    i = 0
    for word in words:
        word_index[word] = i
        i += 1

    record = np.zeros(len(tags))

    A = np.zeros((len(tags), len(tags)))
    B = np.zeros((len(tags), len(words)))

    for sentence in tagset:
        record[tag_index[sentence[0][1]]] += 1

        for i in range(len(sentence)):
            word, tag = sentence[i]
            B[tag_index[tag], word_index[word]] += 1
            if i + 1 < len(sentence):
                next_tag = sentence[i+1][1]
                A[tag_index[tag], tag_index[next_tag]] += 1

    record += 1
    record /= len(tags)
    # to smooth and avoid the occurence of zero
    A = A+1
    B = B+1
    A /= A.sum(axis=1, keepdims=True)
    B /= B.sum(axis=1, keepdims=True)

    obs = []
    for sentence in test_sentences:
        obs_sequence_sentence = []

        for word, _ in sentence:
            if word in word_index:
                obs_sequence_sentence.append(word_index[word])
            else:
                obs_sequence_sentence.append(word_index["UNK"])
        obs.append(obs_sequence_sentence)

    state_sequences = []
    for item in obs:
        states, _ = viterbi(item, record, A, B)
        state_sequences.append(states)

    tag_inferred = []
    for state_sequence in state_sequences:
        curr_tag = []
        for state in state_sequence:
            tag = list(tag_index.keys())[state]
            curr_tag.append(tag)

        tag_inferred.append(curr_tag)

    print("The inffered tag is:", tag_inferred)
    print("")
    print("The actual_tag is: ", actual_tag)
    print("")
    acc = accuracy(tag_inferred, actual_tag)
    print("")
    print(f"The overall accuracy is {acc}")


if __name__ == "__main__":
    predict()
