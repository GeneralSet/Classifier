# Classifier

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt 
```

## Run
### 1. Transform
```bash
./transform/svgToPng.sh
python3 transform/groupSets.py
```

### 2. Train
```bash
python3 train/classifier.py
```

### 3. Serve
```bash
cd app
python3 -m flask --app app run
```
