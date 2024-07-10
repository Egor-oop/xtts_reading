# Book synthesizing with coqui xtts-v2
To run the script, create a virtual environment and install dependencies

###### Create environment
```
python3 -m venv venv
```

###### Activate environment
Unix
```commandline
source venv/bin/activate
```
Win
```commandline
venv\Scripts\activate
```

###### Install dependencies
```commandline
pip install -r requirements.txt
```

##### Run script
```commandline
python main.py
```

## Providing reference audio and book to synthesize
You need to edit the `if __name__ == '__main__'` part:
```python
if __name__ == '__main__':
    main('YOUR BOOK NAME',
         ['LIST OF AUDIO SAMPLES'])
```
