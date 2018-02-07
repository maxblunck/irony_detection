import config
import utils
import corpus
import machine_learning


if __name__ == '__main__':
    """
    Entry point for app
    """

    # create log-files and write headers
    utils.write_resultlog_headers()

    # read and preprocess corpus
    corpus = corpus.read_corpus(config.corpus_path)

    # run main programm
    if config.use_all_variants == False:
        machine_learning.run(corpus)

    else:   
        # gather all possible feature combinations
        f_combinations = utils.get_feature_combos()
        count = 1

        # run main programm for all combinations
        for combo in f_combinations:
            config.feature_selection = combo

            print("\nRunning configuration {} of {}".format(count, len(f_combinations)))
            count += 1
    
            machine_learning.run(corpus)