
### dev_colored PRIMARY (index = 0 mod 3): 267 runs

| Subset | n | full | gaussian | ar1 | ar1_gauss | path_gauss | gaussian vs full (W/L) | ar1_gauss vs ar1 (W/L) | path_gauss vs full (W/L) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 267 | 0.760 | 0.805 | 0.760 | 0.805 | 0.805 | 13/1 (p = 0.0018) | 13/1 (p = 0.0018) | 13/1 (p = 0.0018) |
| noise = 0.01 | 67 | 0.925 | 0.925 | 0.925 | 0.925 | 0.925 | 1/1 | 1/1 | 1/1 |
| noise = 0.05 | 66 | 0.773 | 0.848 | 0.773 | 0.848 | 0.848 | 5/0 | 5/0 | 5/0 |
| noise = 0.1 | 67 | 0.761 | 0.836 | 0.761 | 0.836 | 0.836 | 5/0 | 5/0 | 5/0 |
| noise = 0.2 | 67 | 0.582 | 0.612 | 0.582 | 0.612 | 0.612 | 2/0 | 2/0 | 2/0 |
| N = 500 | 134 | 0.746 | 0.769 | 0.746 | 0.769 | 0.769 | 4/1 | 4/1 | 4/1 |
| N = 2000 | 133 | 0.774 | 0.842 | 0.774 | 0.842 | 0.842 | 9/0 | 9/0 | 9/0 |

frozen router (full): {'ar1': 267}
errors: 0

### dev_colored all completed runs: 800 runs

| Subset | n | full | gaussian | ar1 | ar1_gauss | path_gauss | gaussian vs full (W/L) | ar1_gauss vs ar1 (W/L) | path_gauss vs full (W/L) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 800 | 0.750 | 0.792 | 0.750 | 0.792 | 0.792 | 36/2 (p = 5.4e-09) | 36/2 (p = 5.4e-09) | 36/2 (p = 5.4e-09) |
| noise = 0.01 | 200 | 0.915 | 0.930 | 0.915 | 0.930 | 0.930 | 4/1 | 4/1 | 4/1 |
| noise = 0.05 | 200 | 0.785 | 0.855 | 0.785 | 0.855 | 0.855 | 14/0 | 14/0 | 14/0 |
| noise = 0.1 | 200 | 0.725 | 0.775 | 0.725 | 0.775 | 0.775 | 11/1 | 11/1 | 11/1 |
| noise = 0.2 | 200 | 0.575 | 0.610 | 0.575 | 0.610 | 0.610 | 7/0 | 7/0 | 7/0 |
| N = 500 | 400 | 0.738 | 0.755 | 0.738 | 0.755 | 0.755 | 9/2 | 9/2 | 9/2 |
| N = 2000 | 400 | 0.762 | 0.830 | 0.762 | 0.830 | 0.830 | 27/0 | 27/0 | 27/0 |

frozen router (full): {'ar1': 799, 'white': 1}
errors: 0
reproduction vs results_final/dev_colored.json arm ARp: 400/400 exact flags equal, 400/400 coef_err equal to 1e-9

### dev_student PRIMARY (index = 0 mod 3): 267 runs

| Subset | n | full | gaussian | gaussian vs full (W/L) |
| --- | --- | --- | --- | --- |
| all | 267 | 0.918 | 0.843 | 0/20 (p = 1.9e-06) |
| noise = 0.01 | 67 | 0.985 | 0.910 | 0/5 |
| noise = 0.05 | 66 | 0.909 | 0.848 | 0/4 |
| noise = 0.1 | 67 | 0.910 | 0.821 | 0/6 |
| noise = 0.2 | 67 | 0.866 | 0.791 | 0/5 |
| N = 500 | 134 | 0.903 | 0.821 | 0/11 |
| N = 2000 | 133 | 0.932 | 0.865 | 0/9 |

frozen router (full): {'white': 267}
errors: 0

### dev_student all completed runs: 534 runs

| Subset | n | full | gaussian | gaussian vs full (W/L) |
| --- | --- | --- | --- | --- |
| all | 534 | 0.918 | 0.830 | 0/47 (p = 1.4e-14) |
| noise = 0.01 | 134 | 0.985 | 0.858 | 0/17 |
| noise = 0.05 | 133 | 0.902 | 0.842 | 0/8 |
| noise = 0.1 | 133 | 0.910 | 0.820 | 0/12 |
| noise = 0.2 | 134 | 0.873 | 0.799 | 0/10 |
| N = 500 | 267 | 0.906 | 0.813 | 0/25 |
| N = 2000 | 267 | 0.929 | 0.846 | 0/22 |

frozen router (full): {'white': 534}
errors: 0
reproduction vs results_final/dev_student.json arm Cp: 134/134 exact flags equal, 134/134 coef_err equal to 1e-9

### dev_white PRIMARY (index = 0 mod 3): 267 runs

| Subset | n | full | gaussian | gaussian vs full (W/L) |
| --- | --- | --- | --- | --- |
| all | 267 | 0.891 | 0.891 | 0/0 (p = 1) |
| noise = 0.01 | 67 | 0.955 | 0.955 | 0/0 |
| noise = 0.05 | 66 | 0.909 | 0.909 | 0/0 |
| noise = 0.1 | 67 | 0.896 | 0.896 | 0/0 |
| noise = 0.2 | 67 | 0.806 | 0.806 | 0/0 |
| N = 500 | 134 | 0.873 | 0.873 | 0/0 |
| N = 2000 | 133 | 0.910 | 0.910 | 0/0 |

frozen router (full): {'white': 267}
errors: 0
reproduction vs results_final/dev_white.json arm Cp: 267/267 exact flags equal, 267/267 coef_err equal to 1e-9

### dev_cmid all completed runs: 320 runs

| Subset | n | full | gaussian | gaussian vs full (W/L) |
| --- | --- | --- | --- | --- |
| all | 320 | 0.841 | 0.825 | 1/6 (p = 0.12) |
| noise = 0.01 | 80 | 0.912 | 0.900 | 1/2 |
| noise = 0.05 | 80 | 0.875 | 0.863 | 0/1 |
| noise = 0.1 | 80 | 0.838 | 0.812 | 0/2 |
| noise = 0.2 | 80 | 0.738 | 0.725 | 0/1 |
| N = 500 | 160 | 0.812 | 0.781 | 1/6 |
| N = 2000 | 160 | 0.869 | 0.869 | 0/0 |

frozen router (full): {'white': 320}
errors: 0
reproduction vs results_final/dev_cmid.json arm Cp: 320/320 exact flags equal, 320/320 coef_err equal to 1e-9
