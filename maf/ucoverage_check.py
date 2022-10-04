from rubin_sim import maf
import glob


if __name__ == "__main__":

    files = glob.glob('*10yrs.db')
    files.sort()

    areas = []
    for filename in files:
        runname = filename.split('/')[-1].replace('.db', '')
        slicer = maf.HealpixSlicer()
        # just count how many observations overlap the point
        metric = maf.CountMetric(col='night')
        # limit the data to just things on one night
        sql = 'filter="u" and night < 365 and note not like "%%DD%%"'
        summarystats = [maf.AreaThresholdMetric(lower_threshold=3)]
        
        bundle = maf.MetricBundle(metric, slicer, sql, runName=runname,
                                  summaryMetrics=summarystats)
        bg = maf.MetricBundleGroup([bundle], filename, 'temp', None)
        bg.runAll()
        bg.plotAll(closefigs=False)
        
        areas.append(bundle.summaryValues)
        
    for a, n in zip(areas, files):
        if a is None:
            print(n, None)
        else:
            print(n, a['AreaThreshold'])
