#!/usr/bin/env bash
FOLDER="/home/fabio/Desktop/stocker-master"
PARAMETERS="$FOLDER/parameters_default.json"
STOCKS="$FOLDER/sp500.txt"
while IFS= read -r line
do
  content=$line
  stock="$(cut -d'>' -f1 <<<"$content")"
  index="$(cut -d'>' -f2 <<<"$content")"
  python3 $FOLDER/main.py $stock $index $PARAMETERS
done < "$STOCKS"
