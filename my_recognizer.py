import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]


    for word, model in models.items():
>            calculate the scores for each model(word) and update the 'probabilities' list.
>            determine the maximum score for each model(word).
>            Append the corresponding word (the tested word is deemed to be
    the word for which with the model was trained) to the list 'guesses'.
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    test_sequences = list(test_set.get_all_Xlengths().values())

    for X_test, lengths_test in test_sequences:
        word_probabilities = {}
        best_score = float("-inf")
        guess = None
        for word, model in models.items():
            try:
                score = model.score(X_test, lengths_test)
                word_probabilities[word] = score
                if score > best_score:
                    best_score = score
                    guess = word
            except:
                pass
        probabilities.append(word_probabilities)
        guesses.append(guess)

    return probabilities, guesses
