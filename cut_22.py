from rubin_sim import schema_converter
import glob
import numpy as np

if __name__ == '__main__':

    filenames = glob.glob('*10yrs.db')

    sc = schema_converter()

    outnames = [name.replace('10yrs', '2yrs') for name in filenames]

    for fname, outname in zip(filenames, outnames):
        data = sc.opsim2obs(fname)
        good = np.where(data['night'] < 365.25*2)
        sc.obs2opsim(data[good], filename=outname)
