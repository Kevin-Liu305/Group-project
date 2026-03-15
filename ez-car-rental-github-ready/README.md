# EZ Car Rental Pricing Project

This repository contains the code, outputs, and paper for the EZ Car Rental pricing project.

## Project summary
This project models EZ Car Rental as a **state-dependent pricing bandit** problem and uses **Thompson Sampling** to balance exploration and exploitation.

- **State**: city, time of day, utilization rate
- **Actions**: low / medium / high price
- **Reward**: rental revenue

## Repository structure
- `code/` — notebook and Python script
- `data/` — compressed input datasets (`.csv.gz`)
- `outputs/` — charts and result tables
- `paper/` — final paper in PDF and Word format

## Main results
- The model approximately learned the simulated rental probabilities.
- The learned policy outperformed simple benchmark strategies.
- Results are based on a simulated environment calibrated from historical data.

## Files to open first
1. `paper/ez_car_rental_paper.pdf`
2. `code/ez_car_rental_thompson_sampling_starter.ipynb`

## Running the code
1. Decompress the files in `data/` if needed.
2. Place the extracted CSV files in the same folder expected by the notebook/script, or update the file paths.
3. Run the notebook:
   - `code/ez_car_rental_thompson_sampling_starter.ipynb`

## Notes
- `journeys.csv.gz` and `utilization.csv.gz` are compressed so the repository remains easy to upload to GitHub.
- The project requirement was to upload code/notebooks to GitHub. The charts, tables, and paper are included for convenience.
