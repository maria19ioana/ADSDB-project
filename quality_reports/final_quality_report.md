# Trusted Zone — Data Quality Assessment

## 1. Scope

The analysis evaluates image, text and audio files processed from the Formatted Zone into the Trusted Zone.

## 2. Integrity and acceptance

| modality   |   formatted_files_inspected |   trusted_files_accepted |   excluded_files |   processing_errors |   cleaned_files |   flagged_files |   acceptance_percentage |
|:-----------|----------------------------:|-------------------------:|-----------------:|--------------------:|----------------:|----------------:|------------------------:|
| Images     |                         948 |                      948 |                0 |                   0 |               0 |               0 |                     100 |
| Text       |                         948 |                      948 |                0 |                   0 |             948 |               0 |                     100 |
| Audio      |                          70 |                       70 |                0 |                   0 |               0 |              20 |                     100 |
## 3. Multimodal completeness

14 of 16 species have at least one file in each modality.

Species without audio:

- 034.Gray_crowned_Rosy_Finch
- 044.Frigatebird
## 4. Audio quality

20 of 70 accepted recordings were flagged as mostly silent.

These recordings were retained because low amplitude does not necessarily imply unusable bird vocalizations.

### Warnings by species

| species                       |   recordings |   flagged |   mean_silence_ratio |   median_silence_ratio |   mean_duration_seconds |   mean_rms |   flagged_percentage |
|:------------------------------|-------------:|----------:|---------------------:|-----------------------:|------------------------:|-----------:|---------------------:|
| 031.Black_billed_Cuckoo       |            5 |         0 |              0.58772 |                 0.6526 |                 89.567  |  0.0174128 |                    0 |
| 032.Mangrove_Cuckoo           |            5 |         0 |              0.44596 |                 0.5234 |                 26.793  |  0.0488206 |                    0 |
| 033.Yellow_billed_Cuckoo      |            5 |         2 |              0.60308 |                 0.5072 |                 45.1206 |  0.0377476 |                   40 |
| 035.Purple_Finch              |            5 |         0 |              0.50894 |                 0.4701 |                 95.0642 |  0.0639684 |                    0 |
| 036.Northern_Flicker          |            5 |         1 |              0.6763  |                 0.8398 |                102.471  |  0.0595866 |                   20 |
| 037.Acadian_Flycatcher        |            5 |         0 |              0.56486 |                 0.6122 |                 56.6044 |  0.026288  |                    0 |
| 038.Great_Crested_Flycatcher  |            5 |         1 |              0.68102 |                 0.6794 |                 67.8516 |  0.0285026 |                   20 |
| 039.Least_Flycatcher          |            5 |         3 |              0.86368 |                 0.9069 |                 79.1578 |  0.013379  |                   60 |
| 040.Olive_sided_Flycatcher    |            5 |         3 |              0.66658 |                 0.9004 |                 58.7874 |  0.0712888 |                   60 |
| 041.Scissor_tailed_Flycatcher |            5 |         2 |              0.63838 |                 0.484  |                 89.7908 |  0.0257888 |                   40 |
| 042.Vermilion_Flycatcher      |            5 |         2 |              0.88644 |                 0.8886 |                402.536  |  0.0159604 |                   40 |
| 043.Yellow_bellied_Flycatcher |            5 |         5 |              0.96696 |                 0.9739 |                 46.2364 |  0.00837   |                  100 |
| 045.Northern_Fulmar           |            5 |         0 |              0.51546 |                 0.494  |                117.834  |  0.0260648 |                    0 |
| 046.Gadwall                   |            5 |         1 |              0.79506 |                 0.7704 |                 58.1656 |  0.021844  |                   20 |

## 5. Text cleaning

948 text files were modified during cleaning.

Net characters removed: 948 (0.129% of original characters).

### Net character changes per file

|   characters_removed |   files |
|---------------------:|--------:|
|                    1 |     948 |

A one-character reduction is consistent with removal of a trailing newline, but this should be verified against original file contents.

## 6. Remaining limitations

- Audio is missing for some bird species.
- Quiet audio may contain useful vocalizations.
