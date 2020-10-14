# mbf_heatmap


Heatmaps for ChIPseq lanes and delayed DataFrames.

This is the swiss army knife of ChIPseq heatmaps.

Example usage:

```python
from pathlib import Path
import mbf_align
import mbf_genomes
import mbf_genomics
import mbf_heatmap.chipseq as hc
import pypipegraph as ppg


ppg.new_pipegraph()

genome = mbf_genomes.EnsemblGenome('Homo_sapiens', 100)

lane1 = mbf_align.lanes.AlignedSample("one", "chipseq_one.bam", genome, False, None)
lane2 = mbf_align.lanes.AlignedSample("two", "chipseq_two.bam", genome, False, None)

input_regions = mbf_genomics.regions.Regions_FromBed("My_regions", "input.bed", genome)



hm = hc.Heatmap(
	input_ergions, 
	[lane1, lane2],
	region_strategy = hc.regions.RegionsFromCenter(2000) # +- 1000bp,
	smoothing_strategy = hc.Smooth.SmoothExtendedReads(200) # extend reeads *by* 200bp
)

Path('results').mkdir(exists_ok=True)
hm.plot("results/my_first_heatmap.png", 
	norm = hc.norm.AsIs(),
	order = hc.order.AsIs(),
	)
```

Part of the mbf_* suite from https://github.com/IMTMarburg
