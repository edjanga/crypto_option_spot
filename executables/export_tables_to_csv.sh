sqlite3 ../data_centre/crypto_option_spot.db <<EOF
.headers on
.mode csv
.output ../feature_importance.csv
SELECT * FROM feature_importance;
.output stdout
.output ../forecasts.csv
SELECT * FROM forecasts;
.output stdout
.output ../metric.csv
SELECT * FROM metric;
.output stdout
EOF