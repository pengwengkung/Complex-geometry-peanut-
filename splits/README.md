# Dataset split definitions

These files contain one row per segmented peanut crop. Filter the `split` column for `train` or `test`.

## Files

- `random_stratified_80_20.csv`: individual crops shuffled within each region; this is the ordinary random comparison split.
- `leave_merchants_out.csv`: all crops from a merchant stay together; no merchant is in both train and test.
- `split_summary.csv`: train/test merchant and crop totals.
- `region_balance.csv`: region-by-region counts and merchant overlap.
- `leave_merchants_out_merchants.csv`: the complete merchant assignment for the grouped split.

## Reproducibility

The random split uses seed `42` and a requested test fraction of `0.20`.
The random split is stratified only by `region_label`; it does not keep merchants or parent photos together.

A merchant ID is parsed from the parent-photo stem: `GD1.jpg`, `GD1(2).jpg`, and `GD1(3).jpg` become merchant `GD1`. Full-width parentheses are normalized before parsing. Unparseable source names are kept as separate `UNKNOWN_...` merchants.

For the leave-merchants-out split, whole merchants are selected within each region to get as close as possible to 20% of that region's crops while leaving at least one merchant in training. Regions with only one merchant cannot appear in both train and test without breaking the no-overlap rule; those regions are kept in training and are shown explicitly in `region_balance.csv`.

## Summary

| strategy | split | merchants | crops | crop fraction |
|---|---|---:|---:|---:|
| random_stratified_80_20 | train | 27 | 2333 | 0.7987 |
| random_stratified_80_20 | test | 27 | 588 | 0.2013 |
| leave_merchants_out | train | 20 | 2357 | 0.8069 |
| leave_merchants_out | test | 7 | 564 | 0.1931 |

## Region balance

See `region_balance.csv` for exact values. The leave-merchants-out selection is region-aware for regions with at least two merchants; one-merchant regions are the unavoidable exception described above.

## CNN-ready pixel handling

The locally generated CNN-ready set was checked separately: all 2,921 files were 224x224 RGB PNGs with 8-bit pixel values. They are stored as ordinary image values from 0 to 255, which is the correct file format; they are not supposed to store ImageNet-normalized floating-point values.

When loading these files for an ImageNet-pretrained backbone, convert the pixels to 0-1 and then apply the standard channel normalization: mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). Apply this once at model input, not once when saving the PNG and again when loading it.
