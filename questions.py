import nltk
import sys
import os
import string
import math

FILE_MATCHES = 3
SENTENCE_MATCHES = 5


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)
    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    documents = {}
    for document in os.listdir(directory):
        with open(os.path.join(directory,document), encoding='utf8') as f:
            content = f.read()
            documents[document] = str(content)
    return documents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stopWords = set(nltk.corpus.stopwords.words("english"))
    stringPunctuaction = list(string.punctuation)
    wordTokens = nltk.word_tokenize(document)
    filteredSentence = [w.lower() for w in wordTokens if not w in stringPunctuaction and w not in stopWords]
    return filteredSentence


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    for filename in documents:
        words.update(documents[filename])
    idfs = {}
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfsFiles = []
    for filename in files:
        suma = 0
        for word in query:
            frequency = files[filename].count(word)
            if frequency != 0: 
                tfidfs = frequency * idfs[word]
                suma += tfidfs
        tfidfsFiles.append((filename, suma))
    tfidfsFiles.sort(key=lambda tfidf: tfidf[1], reverse=True)
    topFiles = [tfidfFile[0] for tfidfFile in tfidfsFiles]
    return topFiles[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idfsSentences = []
    for sentence in sentences:
        suma = 0
        for word in query:
            if word in sentences[sentence]:
                suma += idfs[word]
        idfsSentences.append((sentence, suma))
    idfsSentences.sort(key=lambda tfidf: tfidf[1], reverse=True)
    topSentences = []
    for sentence in idfsSentences[:n]:
        equalsIdfs = []
        for sentence1 in idfsSentences[:n]:
            if sentence != sentence1 and sentence[1] == sentence1[1]:
                if sentence[0] not in equalsIdfs and sentence[0] not in topSentences:
                    equalsIdfs.append(sentence[0])
                if sentence1[0] not in topSentences:
                    equalsIdfs.append(sentence1[0])
        if len(equalsIdfs) == 0 and sentence[0] not in topSentences:
            topSentences.append(sentence[0])
        else:
            sentencesDensity = sortByDensity(query, equalsIdfs, sentences)
            topSentences = topSentences + sentencesDensity
    return topSentences

def sortByDensity(query, equalsIdfs, sentences):
    densities = []
    for sentence in equalsIdfs:
        countWord = 0
        for word in query:
            if word in sentences[sentence]:
                countWord += 1
        density =  countWord / len(sentences[sentence])
        densities.append((sentence, density))
    densities.sort(key=lambda tfidf: tfidf[1], reverse=True)
    densities = [sentence[0] for sentence in densities]
    return densities

if __name__ == "__main__":
    main()
