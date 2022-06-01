echo 'Downloading and setting up MEH Eye Disease data'
DEST_DIR='data/'
ggID='1DOax6o4zTdsYUZQU0uPAkhCxdNRG62V8'  
ggURL='https://drive.google.com/uc?export=download'  
FILENAME='meh_eyedisease.jsonl'
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${FILENAME}"  
mkdir $DEST_DIR
mv $FILENAME $DEST_DIR
echo 'Done'
