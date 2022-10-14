import glob

if __name__ == '__main__':

    dirs = glob.glob('*yrs_ddf')
    for dirname in dirs:
        files = glob.glob(dirname+'/*')
        print(dirname, len(files))

