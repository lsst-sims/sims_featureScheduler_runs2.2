import glob
import os

if __name__ == '__main__':
    filelist = glob.glob('sub_*')
    filelist.sort()

    for filename in filelist:
        slurm_name = 'run_%s.slurm'
        os.system('cp maf_ss_template.slurm %s' % slurm_name)
        os.system("sed -i 's/XXX/%s/g' slurm_name" % filename)
