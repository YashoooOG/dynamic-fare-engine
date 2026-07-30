# Dataset Notes

## Ride Datasets

| Dataset | Source | Save As |
|----------|--------|---------|
| Uber & Lyft Cab Prices (Boston) | https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices | `uber_lyft_boston.csv` |
| Uber Fares Dataset | https://www.kaggle.com/datasets/yasserh/uber-fares-dataset | `uber_fares.csv` |

### Final Folder Structure

```
data/raw/ride/
├── uber_lyft_boston.csv
└── uber_fares.csv
```

---

## Delivery Datasets

### Dataset A — Food Delivery (India)

https://www.kaggle.com/datasets/varshinipallerla/food-delivery

→ Save as:

```
food_delivery_india.csv
```

---

### Dataset B — Food Delivery Time

https://www.kaggle.com/datasets/rajatkumar30/food-delivery-time

→ Save as:

```
food_delivery_time.csv
```

---

### Dataset C — Food Demand Forecasting

https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting

Download the complete dataset.

The downloaded ZIP contains **5 CSV files**:

| File | Action |
|------|--------|
| `train.csv` | Rename to `food_demand_train.csv` |
| `fulfilment_center_info.csv` | Keep as is |
| `meal_info.csv` | Keep as is |
| `test.csv` | Delete (not required) |
| `sample_submission.csv` | Delete (not required) |

`test.csv` and `sample_submission.csv` are included only for the original Kaggle competition and are **not required** for this project.

### Final Folder Structure

```
data/raw/delivery/
├── food_delivery_india.csv
├── food_delivery_time.csv
├── food_demand_train.csv
├── fulfilment_center_info.csv
└── meal_info.csv
```

---

## Dataset Usage

| Dataset | Purpose |
|----------|---------|
| `uber_lyft_boston.csv` | Ride demand forecasting, weather features, surge pricing |
| `uber_fares.csv` | Base fare estimation |
| `food_delivery_india.csv` | Order value, distance, traffic features |
| `food_delivery_time.csv` | Delivery time estimation features |
| `food_demand_train.csv` | Weekly demand forecasting (target: `num_orders`) |
| `fulfilment_center_info.csv` | Center metadata (`center_id`) |
| `meal_info.csv` | Meal metadata (`meal_id`) |