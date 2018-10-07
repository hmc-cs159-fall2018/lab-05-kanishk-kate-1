1. What is Laplace smoothing--You add some constant k to all the n-gram counts. This ensures that n-grams that never appear in the training text are not now impossible to achieve when you run the model.

In the code, it is implemented in train_costs, line 68, where we loop through every combination of letters in the alphabet and add 0.1.

It is necessary because the training data only has so many errors--when spellchecking, the user inputed string can have any mistake--any deletion, insertion, or substitution. Thus, if the training data only has a subset of possible errors, without Laplace smoothing the model will never be able to properly handle mistakes that appear outside this subset. In other words, there should never be no probability.

In prob, this creates a total failure--our probs dictionary starts with a default dict, which initializes all probabilities to zero. Thus, when we call self.probs[intd][obsv], we will get zero if they were never in the training data. We then take the log of this value--thus we will be taking the log(0), which is negative infinite, and will break the program.

2. python3 EditDistance.py -s ed.pkl --source /data/spelling/wikipedia_misspellings.txt is the command you would want to run. 

EditDistance takes in two commandline arguments--one required file that will be the output of the program (the trained model), and one non-required UTF-8 file that will be the training data. 

3. It supports unigram and bigrams.

4. The language model class has an alpha data member. When determining the numerator (which should not be zero, so we don't take the log of zero), you add in this alpha. Thus, you never take the log of zero.

5. It returns if w is in the vocabulary. Specifically, it makes it so that x in LanguageModel checks if x is in the vocabulary.

6. It returns a generator that can be used to make an iterator through all the chunks in the text.

7. python3 LanguageModel.py -s lm.pkl -a 0.1 -v 40000 /data/gutenberg/*.txt is the command you would run. 

LanguageModel takes in one or more UTF-8 encoded files containing the training set SOURCE as a positional argument (and combines them into a list). It also takes in an argument that is a file store, which is required and is where the trained model will be written out to. It takes in some float alpha, which is the amount to add to each entry, and an integer that determines how large the vocabulary is. 

8. (Evaluation 6) The ispell spellchecker performed better in most cases that we analyzed. For every difference between the two, we saw it perform better about 7 to 8 out of 10 times. In the remaining times, ours performed better. There were many cases were they corrected the text to the same thing. In a few interesting cases, they corrected it to the same wrong word (one I thought was particularly interesting was recongise was left alone in both.)

9. (Evaluation 7) Our spell check appears to have a much more limited vocabulary than the ispell one. For example, the word edit is changed almost universally to exit, while banned is changed to banded and phone is changed to shoe. However, the ispell checker leaves these correct words alone. However, I would argue our spellchecker handled incorrect contractions better (Im --> I m while ispell did Im --> Aim). There are some instances of our spellchecker performing better with some names/non-words, such as Lisak. Ispell turned this into Lisa, while ours preserved the true word Lisak, likely because it was too many changes away from a true word.

10. (Evaluation 8) I was surprised by how both spellcheckers handled names--while there were many instances of the spellcheckers inaccurately "correcting" names, there were also a surprisingly large number of cases where they did not change them, even though the name or acronym is not in either spellchecker's dictionary. Upon further reflection, this makes sense with our spellchecker--many names are simply more than two edits/inserts/substitutions away from words in the dictionary.

I was also surprised with how limited the vocabulary was in our spellchecker. As I mentioned in the previous problems, the spellchecker didn't recognize some words like edits and banned.