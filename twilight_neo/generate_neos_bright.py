
if __name__ == '__main__':

    filter_combos = ['riz', 'iz', 'z']
    repeats = [3,4]
    night_patterns = [1,2,3,4,5,6,7]

    for filters in filter_combos:
        for repeat in repeats:
            for night_pattern in night_patterns:
                print('python twi_neo_brightest.py --night_pattern %i --neo_filters %s --neo_repeat %i' % (night_pattern, filters, repeat))
