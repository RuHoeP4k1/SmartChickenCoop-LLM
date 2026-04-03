## Simulation Mode

Production is the default. If no simulation environment variables are set, the pipeline continues to use the production tables.

To activate simulation mode, set these environment variables:

```env
SENSOR_TABLE=sensor_readings_simulation
RISK_SNAPSHOT_TABLE=risk_snapshots_simulation
CV_COUNT_TABLE=cv_counts_simulation
```

To return to the real pipeline, remove these environment variables or set them back to the production tables.
