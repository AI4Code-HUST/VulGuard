import os

class Options:
    # Sets the global home of the project (useful for running external tools)
    PYSZZ_HOME = os.path.dirname(os.path.realpath(__file__))

    TEMP_WORKING_DIR = 'vulguard/crawler/szz/_szztemp'
    SZZ_LOG_DIR = '_szzlog'
    SZZ_OUTPUT = '_szzout'