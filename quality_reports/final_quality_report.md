# Data Quality Assessment

## 1. Scope

The analysis evaluates image, text and audio files processed from the Formatted Zone into the Trusted Zone.

## 2. Integrity and acceptance

| modality | formatted_files_inspected | trusted_files_accepted | excluded_files | processing_errors | cleaned_files | flagged_files | acceptance_percentage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Images | 250 | 250 | 0 | 0 | 0 | 0 | 100.0 |
| Text | 250 | 250 | 0 | 0 | 250 | 0 | 100.0 |
| Audio | 195 | 195 | 0 | 0 | 0 | 47 | 100.0 |
## 3. Multimodal completeness

41 of 50 species have at least one file in each modality.

Species without audio:

- 003.Sooty_Albatross
- 009.Brewer_Blackbird
- 017.Cardinal
- 019.Gray_Catbird
- 022.Chuck_will_Widow
- 023.Brandt_Cormorant
- 034.Gray_crowned_Rosy_Finch
- 044.Frigatebird
- 050.Eared_Grebe
## 4. Audio quality

47 of 195 accepted recordings were flagged as mostly silent.

These recordings were retained because low amplitude does not necessarily imply unusable bird vocalizations.

### Warnings by species

| species | recordings | flagged | mean_silence_ratio | median_silence_ratio | mean_duration_seconds | mean_rms | flagged_percentage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 001.Black_footed_Albatross | 5 | 0 | 0.55118 | 0.5221 | 226.2056 | 0.0839594 | 0.0 |
| 002.Laysan_Albatross | 5 | 1 | 0.5258799999999999 | 0.3963 | 69.78720000000001 | 0.0448586 | 20.0 |
| 004.Groove_billed_Ani | 5 | 4 | 0.90238 | 0.9467 | 27.633 | 0.0121252 | 80.0 |
| 005.Crested_Auklet | 3 | 0 | 0.4371 | 0.3452 | 30.886666666666667 | 0.04316066666666666 | 0.0 |
| 006.Least_Auklet | 5 | 0 | 0.5069 | 0.5744 | 64.1064 | 0.06122760000000001 | 0.0 |
| 007.Parakeet_Auklet | 5 | 0 | 0.29528 | 0.3192 | 13.778 | 0.0701596 | 0.0 |
| 008.Rhinoceros_Auklet | 1 | 0 | 0.3466 | 0.3466 | 5.704 | 0.063392 | 0.0 |
| 010.Red_winged_Blackbird | 5 | 0 | 0.7167 | 0.7924 | 46.14 | 0.049390399999999994 | 0.0 |
| 011.Rusty_Blackbird | 5 | 0 | 0.15589999999999998 | 0.0949 | 58.5622 | 0.133514 | 0.0 |
| 012.Yellow_headed_Blackbird | 5 | 2 | 0.72196 | 0.7435 | 238.53119999999998 | 0.0204932 | 40.0 |
| 013.Bobolink | 5 | 0 | 0.39404 | 0.328 | 63.0364 | 0.0738664 | 0.0 |
| 014.Indigo_Bunting | 5 | 2 | 0.6866000000000001 | 0.6338 | 23.200599999999998 | 0.024691800000000003 | 40.0 |
| 015.Lazuli_Bunting | 5 | 0 | 0.8038000000000001 | 0.8006 | 60.07719999999999 | 0.052825 | 0.0 |
| 016.Painted_Bunting | 5 | 0 | 0.5220400000000001 | 0.4441 | 26.897399999999998 | 0.034648200000000004 | 0.0 |
| 018.Spotted_Catbird | 5 | 3 | 0.88582 | 0.9175 | 53.967200000000005 | 0.019437000000000003 | 60.0 |
| 020.Yellow_breasted_Chat | 5 | 2 | 0.82312 | 0.8382 | 63.1352 | 0.031341999999999995 | 40.0 |
| 021.Eastern_Towhee | 5 | 1 | 0.5788 | 0.5801 | 56.697 | 0.044756000000000004 | 20.0 |
| 024.Red_faced_Cormorant | 2 | 1 | 0.8468 | 0.8468 | 27.826999999999998 | 0.03574 | 50.0 |
| 025.Pelagic_Cormorant | 4 | 0 | 0.189625 | 0.18464999999999998 | 412.69375 | 0.11844924999999999 | 0.0 |
| 026.Bronzed_Cowbird | 5 | 2 | 0.8899799999999999 | 0.8724 | 71.75460000000001 | 0.0171624 | 40.0 |
| 027.Shiny_Cowbird | 5 | 1 | 0.7072 | 0.6366 | 122.62259999999999 | 0.0256408 | 20.0 |
| 028.Brown_Creeper | 5 | 4 | 0.85128 | 0.9285 | 79.7448 | 0.0254332 | 80.0 |
| 029.American_Crow | 5 | 1 | 0.58506 | 0.5071 | 52.6894 | 0.040889199999999994 | 20.0 |
| 030.Fish_Crow | 5 | 3 | 0.72954 | 0.9185 | 35.22860000000001 | 0.044599599999999996 | 60.0 |
| 031.Black_billed_Cuckoo | 5 | 0 | 0.58772 | 0.6526 | 89.567 | 0.0174128 | 0.0 |
| 032.Mangrove_Cuckoo | 5 | 0 | 0.44596 | 0.5234 | 26.793 | 0.048820600000000006 | 0.0 |
| 033.Yellow_billed_Cuckoo | 5 | 2 | 0.6030800000000001 | 0.5072 | 45.1206 | 0.037747600000000006 | 40.0 |
| 035.Purple_Finch | 5 | 0 | 0.5089400000000001 | 0.4701 | 95.0642 | 0.0639682 | 0.0 |
| 036.Northern_Flicker | 5 | 1 | 0.63008 | 0.6087 | 80.3664 | 0.0676254 | 20.0 |
| 037.Acadian_Flycatcher | 5 | 0 | 0.56486 | 0.6122 | 56.6044 | 0.026288 | 0.0 |
| 038.Great_Crested_Flycatcher | 5 | 1 | 0.68102 | 0.6794 | 67.85159999999999 | 0.0285026 | 20.0 |
| 039.Least_Flycatcher | 5 | 3 | 0.8636800000000001 | 0.9069 | 79.1578 | 0.013378999999999999 | 60.0 |
| 040.Olive_sided_Flycatcher | 5 | 3 | 0.66658 | 0.9004 | 58.787400000000005 | 0.07128860000000001 | 60.0 |
| 041.Scissor_tailed_Flycatcher | 5 | 2 | 0.63838 | 0.484 | 89.7908 | 0.0257888 | 40.0 |
| 042.Vermilion_Flycatcher | 5 | 1 | 0.6620199999999999 | 0.6731 | 75.9086 | 0.019128399999999997 | 20.0 |
| 043.Yellow_bellied_Flycatcher | 5 | 5 | 0.9502200000000001 | 0.9495 | 36.7686 | 0.0100658 | 100.0 |
| 045.Northern_Fulmar | 5 | 0 | 0.5154599999999999 | 0.494 | 117.8338 | 0.0260648 | 0.0 |
| 046.Gadwall | 5 | 0 | 0.78276 | 0.7924 | 34.3036 | 0.024544 | 0.0 |
| 047.American_Goldfinch | 5 | 0 | 0.6601 | 0.8247 | 48.24239999999999 | 0.040768 | 0.0 |
| 048.European_Goldfinch | 5 | 1 | 0.7026 | 0.8527 | 49.4232 | 0.0445912 | 20.0 |
| 049.Boat_tailed_Grackle | 5 | 1 | 0.53782 | 0.5044 | 30.969 | 0.0440222 | 20.0 |

## 5. Text cleaning

250 text files were modified during cleaning.

Net characters removed: 250 (0.129% of original characters).

### Net character changes per file

| characters_removed | files |
| --- | --- |
| 1 | 250 |

A one-character reduction is consistent with removal of a trailing newline, but this should be verified against original file contents.

## 6. Remaining limitations

- Audio is missing for some bird species.
- Quiet audio may contain useful vocalizations.
