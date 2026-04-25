# CLI Usage Guide

All commands are run from the **project root**.

---

## Commands

### List available date ranges

```bash
python cli_GRU.py list --source cleaned
python cli_GRU.py list --source reconstructed
```

### List flights inside a date range

```bash
python cli_GRU.py list --source cleaned --range <range_name>
python cli_GRU.py list --source reconstructed --range <range_name>
```

Example:

```bash
python cli_GRU.py list --source cleaned --range step1_raw_2023-08-10_to_2023-09-10
```

### Check for missing files in processed vs raw

```bash
python cli_GRU.py check-missing
```

### Reconstruct a flight

```bash
python cli_GRU.py reconstruct --range <range_name> --flight <flight_name>
```

Example:

```bash
python cli_GRU.py reconstruct --range step1_raw_2023-08-10_to_2023-09-10 --flight 20230810_4ba959_073209_092245
```

### Display a reconstructed flight

```bash
python cli_GRU.py display --range <range_name> --flight <flight_name>
```

Example:

```bash
python cli_GRU.py display --range step1_raw_2023-08-10_to_2023-09-10 --flight 20230810_4ba959_073209_092245
```

> The flight must be reconstructed before it can be displayed.

---

## Environments

### Without Docker (local)

| Command         | Conda environment  |
| --------------- | ------------------ |
| `list`          | `aero_engineering` |
| `check-missing` | `aero_engineering` |
| `reconstruct`   | `ml_models`        |
| `display`       | `aero_engineering` |

### With Docker

| Command         | Docker command prefix      |
| --------------- | -------------------------- |
| `list`          | `docker compose run core`  |
| `check-missing` | `docker compose run core`  |
| `reconstruct`   | `docker compose run torch` |
| `display`       | `docker compose run core`  |

Example:

```bash
docker compose run torch python cli_GRU.py reconstruct --range step1_raw_2023-08-10_to_2023-09-10 --flight 20230810_4ba959_073209_092245
```

---

## Built-in help

```bash
python cli_GRU.py --help
python cli_GRU.py list --help
python cli_GRU.py reconstruct --help
python cli_GRU.py display --help
```
