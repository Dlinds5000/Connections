'''
    Starter code for A4: Connections
    Prof. Eric Alexander, Fall 2024
'''

import gensim.downloader
import numpy as np
import random
import time
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
# Run the lines below to download wordnet for the lemmatizer(one-time run) and words for the dictionary check
# nltk.download('wordnet')
# nltk.downlo
#ad('words')
random.seed(15)
#TODO: implement good _puzzle, figure out how to not return a lemmatized version of a seed(e.g. 'shouting' to 'shout')
def give_lemma(word):
    '''Receives a word and returns the lemma after trying it as each word type'''
    lemmatizer = WordNetLemmatizer()
    for type_speech in ['a', 'r', 'n', 'v', 's']:
        lemma = lemmatizer.lemmatize(word, type_speech) 
        if lemma != word:
            return lemma
    return lemma



def read_in_word_list(file_name):
    '''Reads in a file called file_name with words and returns the words as a list'''
    word_list = []
    with open(file_name) as file:
        for line in file:
            word_list.append(line.strip("\n"))
    return word_list

def tame_seeded_puzzle(seeds, model, valid_words):
    ''' Create a Connections puzzle that includes the four given seeds
        and grabs the three most similar words in the model for each seed. '''

    lemmatized_words = {} # entries:  {lemma: (seed, similarity)}

    if len(seeds) != len(set(seeds)):
        print("You can't have duplicate seeds! Here's a fun one for your trouble!")
        return #Maybe put in a fun one we've come accross here eventually
    
    if "THISISNOTAWORD!" in seeds:
        seeds.remove("THISISNOTAWORD!")
    
    for seed in seeds:

        if seed not in valid_words:
            print("Sorry we don't have "+seed+" in our vocab! Here's a fun one for your trouble!")
            return #Maybe put in a fun one we've come accross here eventually
    
    # we know that the seed will be included so they cannot be replaced!
    for seed in seeds:
        lemmatized_words[seed] = tuple([seed, 2])

    chosen_groups = {}
    for seed in seeds:
        chosen_groups[seed] = []
        
        similar_word_tuples = model.similar_by_word(seed, topn=30)
        similar_words = [seq[0] for seq in similar_word_tuples]
        
        #If the word is not in our smaller vocabulary, don't use it
        for word in similar_words:
            if word not in valid_words:
                similar_words.remove(word)
        

        # For each word and similarity value pair. Tuple is (word, similarity)
        for word_tuple in similar_word_tuples:
            if word_tuple[0] not in similar_words or word_tuple[0] in seeds: # ignore words that aren't valid
                continue
                
            #Check if the word is the same root as the seed and make sure the word is a real english word
            curr_word = word_tuple[0]
            curr_similarity = word_tuple[1]
            if give_lemma(curr_word) != give_lemma(seed):
                if give_lemma(curr_word) not in lemmatized_words.keys():
                    lemmatized_words[give_lemma(curr_word)] = (seed, curr_similarity)
                    chosen_groups[seed].append(curr_word)
                else: # Check if a duplicate word is more similar to its new seed or previously assigned seed
                    original_seed = lemmatized_words[give_lemma(curr_word)][0]
                    original_similarity = lemmatized_words[give_lemma(curr_word)][1]
                    # If duplicate is more similar to its new word
                    if original_similarity < curr_similarity:
                        lemmatized_words[give_lemma(curr_word)] = (seed, curr_similarity) # update existing similarity
                        chosen_groups[seed].append(curr_word) 
                        # get next most similar word for original seed
                       
                        if curr_word in chosen_groups[original_seed]:
                            chosen_groups[original_seed].remove(curr_word) # remove this entry bc it's more similar to another seed
                        else:
                            lemmatizing_list = [give_lemma(seq) for seq in chosen_groups[original_seed]]
                            remove_index = lemmatizing_list.index(give_lemma(curr_word))
                            chosen_groups[original_seed].pop(remove_index)  
                    
                    # If duplicate is more similar to its original word, don't add it in the first place
                           
                # Keep 8 spare words just in case of overlap
                if len(chosen_groups[seed]) == 11:
                    break
    # choose the top three words for each seed and append to a nested list included with the seed
    puzzle = []
    for seed in seeds:
        group = chosen_groups[seed][:3]
        group.append(seed)
        puzzle.append(group)


    return puzzle





def tame_random_puzzle(model, valid_words):
    ''' Grab a random 4 words from the model and use to create a seeded puzzle. '''
    # Check they are not the same lemma, or in each others' top 15 words, so that we will not have a seed that could be similar to another

    random_words = []    
    while len(random_words) != 4: # go until 4 sufficiently distinct words are found
        current_seed = random.choice(valid_words) # generate random word            
        
        if current_seed not in model.key_to_index or len(current_seed) < 3:
            print(current_seed + " not in vocab or too short!")
            continue
        
        accepted = True # assume this potential seed is sufficiently distinct
        for word in random_words:

            # generate lemmas of confirmed random words as well as potential seed
            word_lemma = give_lemma(word)
            current_lemma = give_lemma(current_seed)

            similar_word_tuples = model.similar_by_word(word)[:15] # get the 15 most similar word-probability tuples to the confirmed seed
            similar_words = [seq[0] for seq in similar_word_tuples] # only grab the word from the word similarity tuple
            if current_seed in similar_words or word_lemma == current_lemma or current_seed not in valid_words: # don't allow the potential seed to be similar to a confirmed seed or share a lemma with it
                accepted = False

        if accepted: # add a new confirmed seed if it is sufficiently distinct from all the others
            random_words.append(current_seed)
        
    #print(random_words)
    # now that we have generated the seeds, use tame_seeded_puzzle() to generate the puzzle
    return tame_seeded_puzzle(random_words, model, valid_words)





def generate_similar_seeds(model, valid_words, topn=30):
    '''This function generates seeds that are similar to an initial choice made within the function
    it then outputs topn words as similar words. It uses a gensim model passed in, as well as a lost of valid_words'''
    original_seed = ""
    while original_seed not in model.key_to_index or len(original_seed) < 3: # make sure the random word we generate is valid
        original_seed = random.choice(valid_words)

    similar_seed_tuples = model.similar_by_word(original_seed, topn=topn)
    similar_seeds = [seq[0] for seq in similar_seed_tuples]

    return similar_seeds


def red_herring_puzzle(model, valid_words, herring_num=False):
    ''' Make a puzzle with a specified number of 'red-herring' words! can be 2, 3, or 5 words that seem to be 
    similar but are actually unrelated or there is an odd one out in the case of 5'''
    if herring_num not in [2,3,5]:
        print("Randomizing red herring number")
        herring_num = random.choice([2,3,5])
    
    #Generate intial random word for the herring group
    #word_list = read_in_word_list("mit_wordlist.txt")
    similar_seeds = generate_similar_seeds(model, valid_words)
    
    
    #For 2,3 we add the number of red herring words to the seeds for a future random puzzle
    if herring_num in [2,3]:
        seeds = set()
        lemmatized_seeds = set()
        # Building the seed list with our red herring words
        while len(seeds) < herring_num:
            # As long as our list has words in it
            if len(similar_seeds) != 0:
                chosen = similar_seeds[-1]
                similar_seeds = similar_seeds[:-1]
                if give_lemma(chosen) not in lemmatized_seeds and chosen in valid_words and len(chosen) >= 3:
                    seeds.add(chosen)
                    lemmatized_seeds.add(give_lemma(chosen))
            else: # start over because we ran out of words to consider as seeds (due to vocab weirdness)
                similar_seeds = generate_similar_seeds(model, valid_words)
                seeds = set()
                lemmatized_seeds = set()
                
        # Building the rest of the seed list as we did in tame_random_puzzle, making sure the other seed(s) aren't too simi
        seeds = list(seeds)
        while len(seeds) != 4: # go until 4 sufficiently distinct words are found
            current_seed = random.choice(valid_words) # generate random word            
            
            if current_seed not in model.key_to_index or len(current_seed) < 3:
                continue
            
            accepted = True # assume this potential seed is sufficiently distinct
            for word in seeds:
                #print(word)
                # generate lemmas of confirmed random words as well as potential seed
                word_lemma = give_lemma(word)
                current_lemma = give_lemma(current_seed)

                similar_word_tuples = model.similar_by_word(word)[:15] # get the 15 most similar word-probability tuples to the confirmed seed
                similar_words = [seq[0] for seq in similar_word_tuples] # only grab the word from the word similarity tuple
                if current_seed in similar_seeds or word_lemma == current_lemma or current_seed not in valid_words: # don't allow the potential seed to be similar to a confirmed seed or share a lemma with it
                    accepted = False

            if accepted: # add a new confirmed seed if it is sufficiently distinct from all the others
                seeds.append(current_seed)


        return tame_seeded_puzzle(seeds, model, valid_words)
    
    if herring_num in [5]:
        group = set()
        lemmatized_group = set()

        while similar_seeds[-1] not in valid_words:
            similar_seeds = generate_similar_seeds(model, valid_words)

        herring = similar_seeds[-1]
        print(herring)

        for word in similar_seeds:
            if word in valid_words:
                if give_lemma(word) not in lemmatized_group:
                    group.add(word)
                    lemmatized_group.add(give_lemma(word))

                if len(group) == 4:
                    break
        
        seeds = ["THISISNOTAWORD!", herring]    
        while len(seeds) != 4: # go until 4 sufficiently distinct words are found
            current_seed = random.choice(valid_words) # generate random word            
            
            if current_seed not in model.key_to_index or len(current_seed) < 3:
                #print(current_seed + " not in vocab!")
                continue
            
            accepted = True # assume this potential seed is sufficiently distinct
            for word in seeds:
                # Do not compare the newly generated words to our already made group
                if word == "THISISNOTAWORD!":
                    continue
                
                word_lemma = give_lemma(word)
                current_lemma = give_lemma(current_seed)

                similar_word_tuples = model.similar_by_word(word)[:15] # get the 15 most similar word-probability tuples to the confirmed seed
                similar_words = [seq[0] for seq in similar_word_tuples] # only grab the word from the word similarity tuple
                #print("similar words: ",similar_word_tuples)
                if current_seed in similar_words or word_lemma == current_lemma or current_seed not in valid_words: # don't allow the potential seed to be similar to a confirmed seed or share a lemma with it
                    accepted = False

            if accepted: # add a new confirmed seed if it is sufficiently distinct from all the others
                seeds.append(current_seed)
                
        generated_puzzle = tame_seeded_puzzle(seeds, model, valid_words)
        generated_puzzle.append(list(group))

        return generated_puzzle
        
def generate_short_words(model, valid_words, word, topn=30):
    '''This function generates a list of smaller words(or parts of words) of lengths 2 or 3 to be used by subwords puzzle.
    It returns a list of smaller words that are similar to an  small original input word'''
    while word not in model.key_to_index and len(word) != 3 and len(word) != 2: # make sure the random word we generate is valid
        word = random.choice(valid_words)

    similar_word_tuples = model.similar_by_word(word, topn=topn)
    similar_words = [seq[0] for seq in similar_word_tuples]
    for sim_word in similar_words:
        if len(sim_word) not in [2,3]:
            similar_words.remove(similar_words)
    return similar_words

def make_subwords_group(model, valid_words):
    short_seed = ""
    while len(short_seed) not in [2,3]:
        short_seed = random.choice(valid_words)
    subwords_group = generate_short_words(model, valid_words, short_seed, topn=30)
 
    # take the 4 most similar short words that we just generated
    print("subwords group")
    print(subwords_group)

    return subwords_group

# Subwords within words 
def subwords_puzzle(model, valid_words):

    subwords_group = make_subwords_group(model, valid_words)

    long_words = []
    # find a word that contains each of the 4 subwords above
    for subword in subwords_group:
        if len(long_words) == 4:
            break
        #Find a word that has it as a subword
        for word in valid_words:
            if subword in word and len(word) > 3:
                long_words.append(word)
                break
            
    if len(long_words) != 4:
        # try again!
        pass
        

        


    
        



        
    

    




def print_puzzle(words, shuffle=True):
    ''' Print the given words formatted as a Connections puzzle. 
        Can take a list of 16 words or four lists of 4. '''
    # Make sure we have what we need
    print(words)
    words = np.array(words)
    if words.shape == (4,4):
        words = words.flatten()
    elif words.shape != (16,):
        raise Exception("words must be 4x4 or 16x1.")

    # Randomize and to upper-case
    words = words.copy()
    words = [word.upper() for word in words]
    if shuffle:
        random.shuffle(words)

    # Print the puzzle with padding
    max_length = max(len(word) for word in words)
    width = 5 + 2*max_length
    border_line = ('+' + '-'*(max_length+2))*4 + '+'
    for i in range(4):
        print(border_line)
        print('|' + '|'.join([word.center(max_length+2) for word in words[i*4 : i*4 + 4]]) + '|')
    print(border_line)

def main():
    # Note: the first time you download this model will take a little while
    print('Loading model...')
    start = time.time()
    #model = gensim.downloader.load('word2vec-google-news-300') # This one is slower to load
    model = gensim.downloader.load('glove-wiki-gigaword-50') # This one is faster to load
    print('Done. ({} seconds)'.format(time.time() - start))
    valid_words = read_in_word_list("mit_wordlist.txt")
    # Just to show we can print puzzles
    test_puzzle = ['extra','ball','won','mug','pin','copy','too','tee','ate','spare','pen','lane','alley','tote','for','backup']
    print_puzzle(test_puzzle)

    # Can also provide them as a 4x4 list
    test_puzzle2 = [['ball','pot','javelin','tantrum'],       # things you throw
                    ['basket','blanket','net','web'],         # things you weave
                    ['angry','irate','steaming','fuming'],    # synonyms for "mad"
                    ['hall','center','library','observatory'] # second words in names of campus buildings
                    ]
    #print_puzzle(test_puzzle2, shuffle=False)

    # SOME UNIT TESTS

    #print_puzzle(tame_seeded_puzzle(["tree", "happy", "conscious", "sandwich"], model))

    # should return nothing because tree is a duplicate seed
    # print_puzzle(tame_seeded_puzzle(["tree", "tree", "conscious", "sandwich"], model))

    # cat is more similar to dog than pet is
    # also pet should not get cat because cat is a seed
    # [pet, cat, conscious, sandwich]

    #print_puzzle(tame_seeded_puzzle(['crypt', 'christ', 'altar', 'isbn'], model))

    #print_puzzle(tame_seeded_puzzle(['uses', 'requires', 'enable', 'cow'], model))

    #ideas from office hours: don't make words less than 3 letters.
    # (Thought is that we need only generate seeds like this but we can just load in all words like this if we need), 
    # for red_herrings, as long as 2/3 and 5 are different enough and implemented well enough, no problems,
    # Need 1 more extension still.
    print_puzzle(red_herring_puzzle(model, valid_words, herring_num=2))

if __name__=='__main__':
    main()