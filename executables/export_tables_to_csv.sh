sqlite3 /bitbcuket/djanga/data_centre2/crypto_option_spot.db <<EOF
.headers on
.mode csv
.output /bitbucket/djanga/data_centre2/spot_features.csv
SELECT * FROM spot_features;
.output stdout
.output /bitbucket/djanga/data_centre2/option_features.csv
SELECT * FROM option_features;
.output stdout
EOF