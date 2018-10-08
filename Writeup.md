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

11. (Extension 8) We attempted to create a model whereby our spellchecker would learn from the vocabulary if it found that there were too many instances of the same misspelling. Our approach was very simple: We would create a large database of unknown words/common misspellings, and if it seemed like we were finding an unknown word extremely often, with an edit distance score that was very high from it's closest words, we would consider that word to be a proper word. At first, I used a random threshold of 15 (just for testing) and this caught many words like album, vandalism, wiki, etc, all of which are real words. At a first pass, therefore, this is a sensible metric.
This is of course not how adaptive spell checkers are created in the real world. 

12. (Extension 9) The way we created our vocabulary model certainly helped with recognizing words that are only part of the internet. For example, our corrected version of a sentence was:

"please note that sorenaaryamanesh is involved in an editing war by removing unbiased primary source material intended to remove pov of article"

Whereas spell check got:

"Please note that Sorenaaryamanesh is involved in an editing war by removing unbiased primary source material intended to remove POD of article."

Our version is obviously the correct version, since it makes sense given the context of the sentence. Another example is the fragment: "btw you did an astonishingly good job breaking down" which was not a valid fragment in our original spellchecker, but it eventually learned that "btw" is a valid word (ispell seems to have had BTW in it's vocabulary already, however).

Thus, making it so that the spellchecker learned new words from the training data helped its overall performance.

13. (Extension 10) We did not run into major challenges in implementing this change other than the fact that our method of making the spellchecker learn new words is not the best one. To explore this problem, we did some research into what real spellcheckers do to handle this.

The interesting thing is that the most common methods for doing adaptive spell check are all deep learning based methods. For example, there are articles by various amateurs, such as [this](https://datascience.stackexchange.com/questions/16128/algorithms-and-techniques-for-spell-checking) one on stackexchange, and [this](https://towardsdatascience.com/creating-a-spell-checker-with-tensorflow-d35b23939f60) one on Medium discussing heavy duty tools like TensorFlow, Pandas and scikit-learn for data science based approaches to adaptive spell checkers. There are also papers like [this](http://aclweb.org/anthology/P18-3021) one, published in 2018, that outline deep learning methodology for identifying new words in resource scarce languages like Hindi or Malayalam. Another interesting tool is "seq2spell". It takes a sequence of words and then suggests a corrected spelling version of that sequence. While we read through many of these papers and tools, we couldn't find the time to fully implement any of these, especially because installing TensorFlow is nightmarishly difficult.

Many of these models use neural networks, such as RNN's or LSTM's to try and establish patterns over long corpora of texts. A few of these exploit the work done by the gloVe vector team as well, trying to learn about the vector word distances in order to establish patterns in vector space. While the research done is extremely interesting, it was unfortunately extremely difficult to actually implement these ideas.

14. (We added this question ourselves!): Why is adaptive spellchecking good?

The question of adaptive spellchecking is very interesting. Not only is it useful when dealing with words that are names or places, and so are likely not in a dictionary, it also has other applications beyond this. In particular, it can aid in the bias that is inherit in spellchecking. Because spellcheckers are usually implemented by those who use the English vernacular taught in public high schools, that is the dialect that is most represented. Other dialects, which may contain different words, conjugations, etc will thus be overtargeted by the spellchecker even if the written words are correct for that dialect. One particular example that comes to mind is when writing using African American Vernacular English. This vernacular has many different spellings, conjugations, and words, which will be considered "incorrect" by a traditional, non-adaptive, and non-comprehensive spellchecker, even though the written words are correct. An adaptive spellchecker would wrongly mark words as incorrect at first, but would learn not to over time, and thus would hopefully end up more properly representing this dialect. 







