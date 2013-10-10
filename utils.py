import os
from os import listdir, makedirs
from os.path import isdir, join
from multiprocessing import Pool, Value
from sys import stdout

def save_cache(stars, options):
    from pickle import Pickler
    cache_file = open(options.save_cache, 'bw')
    pickler = Pickler(cache_file)
    cache = {
        'stars': stars,
        'interpolant': options.interpolant}
    pickler.dump(cache)

def load_cache(options):
    from pickle import Unpickler
    cache_file = open(options.load_cache, 'br')
    unpickler = Unpickler(cache_file)
    cache = unpickler.load()
    return cache

def get_files(directory, format):
    return [join(directory, filename)
            for filename in sorted(listdir(directory))
            if filename[-4:] == format]

total, progress = 0, 0

def map_reduce(func, args, options):
    print("it's starting map reduce")
    if options.verbose:
        initialize_status_bar(len(args))
    results = []
    append = results.append
    processors = options.processors
    print("reducing")
    if processors is None or processors > 1:
        p = Pool() if processors is None else Pool(processors)
        for arg in args:
            append(p.apply_async(func, (arg,), options.__dict__, task_finished))
        p.close()
        p.join()
        #print results
        print("getting results")
        results = (result.get() for result in results)
        print(results)
    else:
        for arg in args:
            append(func(arg, **options.__dict__))
            task_finished()
    print("reduced")
    return [result for result in results if result is not None]

def initialize_status_bar(new_total=1):
    global total, progress
    total = new_total
    progress = Value('I', 0)
    update_status_bar()

def task_finished(a=None):
    """Call this after finishing a task to update the status bar."""
    global progress, total
    progress.acquire()
    progress.value += 1
    update_status_bar(progress.value/float(total))
    progress.release()

def update_status_bar(frac=0, size=65):
    """Prints a status bar to the terminal with the given fraction filled."""
    perc = int(size*frac)
    out = '\r[' + '='*perc + ' '*(size-perc) + '] {0:%}'.format(frac)
    stdout.write(out)# + '] {0:%}'.format(frac))
    #print out + '] {0:%}'.format(frac)

def raw_string(s):
 #   if isinstance(s, str):
 #       s = s.encode('string_escape')
 #   elif isinstance(s, unicode):
 #       s = s.encode('unicode_escape')
    return s

def make_sure_path_exists(path):
    """Creates the supplied path. Raises OS error if the path cannot be
    created."""
    try:
      makedirs(path)
    except OSError:
      if not isdir(path):
        raise

def get_unmasked(data):
    """Returns all of the values that are not outliers."""
    return data[~data.mask].reshape(-1, data.shape[1])

def get_masked(data):
    """Returns all identified outliers"""
    return data[data.mask].data.reshape(-1, data.shape[1])


def splitOn(split_type, string):
    same = type(string[0]) == split_type
    for index, character in zip(range(len(string)), string):
        if (    type(c) == split_type and not same or
                type(c) != split_type and same):
            return string[:index], string[index:]
def splitAtFirst(split_type, string):
    for index, character in zip(range(len(string)), string):
        try:
            split_type(character)
            return string[:index], split_type(string[index:])
        except: pass
#        if type(character) == split_type:
#            return string[:index], string[index:]
